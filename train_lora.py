#!/usr/bin/env python3
"""
Train a multi-task VLM: DeiT-Small (frozen) + Qwen2.5-0.5B (LoRA).

Architecture:
- Vision: DeiT-Small (22M, frozen, pretrained ImageNet)
- Projector: Linear(384 → 896) + RMSNorm (trainable)
- Language: Qwen2.5-0.5B with LoRA (rank=16, ~2.6M trainable params)

Uses Qwen's tokenizer (151K vocab) — much better than our custom 5K one.
Special tokens added for tasks and coordinates.
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    # Vision
    vision_model: str = "vit_small_patch16_224"
    vision_dim: int = 384
    image_size: int = 224
    patch_size: int = 16

    # Language
    lm_model: str = "Qwen/Qwen2.5-0.5B"
    lm_dim: int = 896
    max_seq_len: int = 512

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    time_budget: int = 7200  # 60 min

    # Computed
    num_patches: int = 0

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 196


CONFIG = Config()

# ─── Tokenizer Setup ────────────────────────────────────────────────────────

def setup_tokenizer():
    """Load Qwen tokenizer and add special VLM tokens."""
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.lm_model, trust_remote_code=True)

    # Add task tokens and coordinate tokens
    special_tokens = [
        "<caption>", "<vqa>", "<od>", "<seg>", "<describe>",
        "<box>", "</box>", "<poly>", "</poly>", "<sep>",
    ]
    # Add location tokens: <loc000> to <loc999>
    loc_tokens = [f"<loc{i:03d}>" for i in range(1000)]

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens + loc_tokens})
    return tokenizer


# ─── Model ───────────────────────────────────────────────────────────────────

class VLMWithLoRA(nn.Module):
    """DeiT-Small (frozen) + projector + Qwen2.5-0.5B (LoRA)."""

    def __init__(self, config: Config, tokenizer):
        super().__init__()
        self.config = config

        # Vision encoder (frozen)
        self.vision_encoder = timm.create_model(
            config.vision_model, pretrained=True, num_classes=0
        )
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Projector: vision_dim → lm_dim
        self.projector = nn.Sequential(
            nn.Linear(config.vision_dim, config.lm_dim),
            nn.GELU(),
            nn.Linear(config.lm_dim, config.lm_dim),
        )

        # Language model with LoRA
        self.lm = AutoModelForCausalLM.from_pretrained(
            config.lm_model, trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        # Resize embeddings for new special tokens
        self.lm.resize_token_embeddings(len(tokenizer))

        # Apply LoRA — critically, also fully train embed_tokens + lm_head
        # so the model can learn the new <loc000>-<loc999> coordinate tokens
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        self.lm = get_peft_model(self.lm, lora_config)

    def forward(self, images, input_ids, target_ids, attention_mask=None):
        """
        images: (B, 3, H, W)
        input_ids: (B, T_in) — task + optional question tokens
        target_ids: (B, T_tgt) — answer tokens
        """
        B = images.shape[0]

        # 1. Vision features (frozen)
        with torch.no_grad():
            vis_features = self.vision_encoder.forward_features(images)
            vis_features = vis_features[:, 1:, :]  # remove CLS, (B, 196, 384)

        # 2. Project to LM dimension
        vis_tokens = self.projector(vis_features.float())  # (B, 196, 896)
        vis_tokens = vis_tokens.to(torch.bfloat16)

        # 3. Embed text tokens
        all_ids = torch.cat([input_ids, target_ids], dim=1)  # (B, T_text)
        text_embeds = self.lm.get_input_embeddings()(all_ids)  # (B, T_text, 896)

        # 4. Concat: [vis_tokens, text_embeds]
        full_embeds = torch.cat([vis_tokens, text_embeds], dim=1)  # (B, 196+T_text, 896)

        # 5. Create attention mask (all 1s)
        T_total = full_embeds.shape[1]
        attn_mask = torch.ones(B, T_total, device=DEVICE, dtype=torch.long)

        # 6. Forward through LM
        outputs = self.lm(inputs_embeds=full_embeds, attention_mask=attn_mask)
        logits = outputs.logits  # (B, T_total, vocab_size)

        return logits


# ─── Data ────────────────────────────────────────────────────────────────────

class VLMDatasetQwen(Dataset):
    """Dataset that re-tokenizes examples with Qwen tokenizer."""

    def __init__(self, examples, images, tokenizer, max_target_len=200):
        self.tokenizer = tokenizer
        self.images = images
        self.max_target_len = max_target_len
        self.examples = []

        # Re-tokenize with Qwen tokenizer
        # We need to convert from our old token format to text, then re-tokenize
        from prepare_data import load_tokenizer as load_old_tok
        old_tok = load_old_tok()

        for ex in examples:
            if ex["image_id"] not in images:
                continue

            task = ex["task"]
            # Decode old tokens back to text
            old_input = old_tok.decode(ex["input_ids"])
            old_target = old_tok.decode(ex["target_ids"])

            # Re-encode with Qwen tokenizer
            input_ids = tokenizer.encode(old_input, add_special_tokens=False)
            target_ids = tokenizer.encode(old_target, add_special_tokens=False)

            if len(target_ids) > max_target_len:
                target_ids = target_ids[:max_target_len]

            self.examples.append({
                "image_id": ex["image_id"],
                "task": task,
                "input_ids": input_ids,
                "target_ids": target_ids,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = self.images[ex["image_id"]]

        # Convert to PyTorch: (3, H, W), ImageNet normalize
        img_t = torch.tensor(img).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        return {
            "image": img_t,
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(ex["target_ids"], dtype=torch.long),
            "task": ex["task"],
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    max_in = max(b["input_ids"].shape[0] for b in batch)
    max_tgt = max(b["target_ids"].shape[0] for b in batch)

    pad_id = 0
    input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
    target_ids = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)

    for i, b in enumerate(batch):
        input_ids[i, :b["input_ids"].shape[0]] = b["input_ids"]
        target_ids[i, :b["target_ids"].shape[0]] = b["target_ids"]

    tasks = [b["task"] for b in batch]
    return images, input_ids, target_ids, tasks


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_loss(logits, target_ids, num_patches, input_len):
    """CE loss on target tokens only."""
    target_start = num_patches + input_len
    target_end = target_start + target_ids.shape[1]
    pred_logits = logits[:, target_start - 1:target_end - 1, :]
    mask = (target_ids != 0).float()
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ─── Training ────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("Auto-VLM: DeiT-Small + Qwen2.5-0.5B (LoRA)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Tokenizer
    print("\nSetting up Qwen tokenizer...")
    tokenizer = setup_tokenizer()
    print(f"  Vocab size: {len(tokenizer)}")

    # Load data
    print("\nLoading data...")
    from prepare_data import load_data, load_images
    train_examples = load_data("train")
    val_examples = load_data("val")
    images = load_images()

    # Extra train data
    extra_path = Path("data/processed/train_extra.json")
    extra_img_path = Path("data/processed/train_images.npz")
    if extra_path.exists() and extra_img_path.exists():
        with open(extra_path) as f:
            extra = json.load(f)
        extra_imgs = np.load(extra_img_path)
        images.update({int(k): extra_imgs[k] for k in extra_imgs.files})
        train_examples.extend(extra)
        print(f"  +{len(extra)} extra examples, +{len(extra_imgs.files)} images")

    # Upsample
    from collections import Counter
    tc = Counter(ex["task"] for ex in train_examples)
    mx = max(tc.values())
    up = list(train_examples)
    for t, c in tc.items():
        if c < mx:
            te = [e for e in train_examples if e["task"] == t]
            up.extend(te * (max(1, round(mx / c)) - 1))
    train_examples = up

    print(f"  Train: {len(train_examples)} (upsampled)")
    tc2 = Counter(ex["task"] for ex in train_examples)
    for t, c in sorted(tc2.items()):
        print(f"    {t}: {c}")
    print(f"  Val: {len(val_examples)}")

    # Datasets
    print("\nTokenizing with Qwen tokenizer...")
    train_ds = VLMDatasetQwen(train_examples, images, tokenizer)
    val_ds = VLMDatasetQwen(val_examples, images, tokenizer)
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Val examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, drop_last=True, pin_memory=True)

    # Model
    print("\nBuilding model...")
    model = VLMWithLoRA(CONFIG, tokenizer).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params / 1e6:.1f}M")
    print(f"  Trainable: {trainable / 1e6:.1f}M (LoRA + projector)")
    print(f"  Frozen: {(total_params - trainable) / 1e6:.1f}M")
    model.lm.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay,
    )

    total_steps_est = int(len(train_ds) / CONFIG.batch_size * 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps_est, eta_min=CONFIG.learning_rate * 0.1
    )

    # Training
    print(f"\nTraining for {CONFIG.time_budget}s...")
    print(f"  bs={CONFIG.batch_size}, lr={CONFIG.learning_rate}, lora_r={CONFIG.lora_rank}")
    print()

    start = time.time()
    step = 0
    epoch = 0
    losses = []
    # bf16 doesn't need GradScaler (only fp16 does)

    model.train()
    while True:
        epoch += 1
        for batch in train_loader:
            if time.time() - start > CONFIG.time_budget:
                break

            imgs, inp, tgt, tasks = batch
            imgs, inp, tgt = imgs.to(DEVICE), inp.to(DEVICE), tgt.to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs, inp, tgt)
                loss = compute_loss(logits, tgt, CONFIG.num_patches, inp.shape[1])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CONFIG.grad_clip
            )
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 10 == 0:
                avg = np.mean(losses[-10:])
                elapsed = time.time() - start
                print(f"  Step {step:5d} | loss={avg:.4f} | time={elapsed:.0f}s / {CONFIG.time_budget}s")

        if time.time() - start > CONFIG.time_budget:
            break

    elapsed = time.time() - start
    print(f"\nTraining complete: {step} steps, {epoch} epochs, {elapsed:.0f}s")

    # Validation
    print("\nValidation...")
    model.eval()
    val_losses = []
    val_task = {}
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            if n >= 200:
                break
            imgs, inp, tgt, tasks = batch
            imgs, inp, tgt = imgs.to(DEVICE), inp.to(DEVICE), tgt.to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs, inp, tgt)
                loss = compute_loss(logits, tgt, CONFIG.num_patches, inp.shape[1])

            val_losses.append(loss.item())
            for t in tasks:
                val_task.setdefault(t, []).append(loss.item())
            n += 1

    val_loss = np.mean(val_losses) if val_losses else float("inf")
    print(f"\nVal loss: {val_loss:.4f}")
    for t, ls in sorted(val_task.items()):
        print(f"  {t:12s}: {np.mean(ls):.4f} ({len(ls)} batches)")

    # Save
    ckpt = Path("checkpoint_lora")
    ckpt.mkdir(exist_ok=True)
    model.lm.save_pretrained(str(ckpt / "lora"))
    torch.save(model.projector.state_dict(), ckpt / "projector.pt")
    conf = asdict(CONFIG)
    conf["val_loss"] = float(val_loss)
    conf["trainable_params"] = trainable
    conf["total_params"] = total_params
    conf["train_steps"] = step
    with open(ckpt / "config.json", "w") as f:
        json.dump(conf, f, indent=2)

    print(f"\nSaved to {ckpt}/")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Trainable: {trainable / 1e6:.1f}M")

    return val_loss, trainable


if __name__ == "__main__":
    train()
