#!/usr/bin/env python3
"""
Train a tiny multi-task VLM in PyTorch with a pretrained vision encoder.

Uses DeiT-Tiny (pretrained on ImageNet-21k) as the vision backbone,
frozen during training. Only the projector + language decoder are trained.

This gives much better visual features than training from scratch.
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

from prepare_data import load_tokenizer, load_data, load_images, load_metadata

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    # Vision encoder (pretrained DeiT-Tiny)
    image_size: int = 224
    patch_size: int = 16  # DeiT-Tiny uses patch_size=16
    vision_dim: int = 192  # DeiT-Tiny embed_dim
    vision_model: str = "vit_tiny_patch16_224"
    freeze_vision: bool = True

    # Language decoder
    lang_dim: int = 384
    lang_depth: int = 4
    lang_heads: int = 6
    max_seq_len: int = 512  # 196 patches + text

    # Training
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    dropout: float = 0.2
    grad_clip: float = 1.0
    time_budget: int = 3600  # seconds

    # Data
    vocab_size: int = 0  # Set from tokenizer
    num_patches: int = 0

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2


CONFIG = Config()

# ─── Model Components ────────────────────────────────────────────────────────

class PretrainedVisionEncoder(nn.Module):
    """Pretrained DeiT-Tiny as frozen feature extractor."""

    def __init__(self, config: Config):
        super().__init__()
        self.model = timm.create_model(
            config.vision_model, pretrained=True, num_classes=0
        )
        # Remove classification head — we want patch features
        # timm ViT with num_classes=0 outputs (B, num_patches, dim) after forward_features

    def forward(self, images):
        # images: (B, 3, H, W) for PyTorch
        # timm ViT forward_features returns (B, num_patches+1, dim) with CLS token
        features = self.model.forward_features(images)  # (B, 197, 192)
        # Remove CLS token
        features = features[:, 1:, :]  # (B, 196, 192)
        return features


class VisionProjector(nn.Module):
    """Project vision features to language dimension."""

    def __init__(self, vision_dim: int, lang_dim: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, lang_dim)
        self.norm = RMSNorm(lang_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation (LLaMA-style)."""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, mlp_dim)
        self.w2 = nn.Linear(mlp_dim, dim)
        self.w3 = nn.Linear(dim, mlp_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with SwiGLU."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        mlp_dim = int(dim * 4 * 2 / 3)
        self.mlp = SwiGLU(dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with causal mask
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=(mask is None))
        x = x + self.dropout(h)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class LanguageDecoder(nn.Module):
    """Small GPT-style causal decoder."""

    def __init__(self, config: Config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.lang_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.lang_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(config.lang_dim, config.lang_heads, config.dropout)
            for _ in range(config.lang_depth)
        ])
        self.norm = RMSNorm(config.lang_dim)
        self.head = nn.Linear(config.lang_dim, config.vocab_size, bias=False)

        # Init
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(self, x, offset=0):
        T = x.shape[1]
        x = x + self.pos_embed[:, offset:offset + T, :]
        # Generate causal mask
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)

    def embed_tokens(self, token_ids):
        return self.token_embed(token_ids)


class MiniVLM(nn.Module):
    """Multi-task VLM with pretrained vision encoder."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vision_encoder = PretrainedVisionEncoder(config)
        self.projector = VisionProjector(config.vision_dim, config.lang_dim)
        self.decoder = LanguageDecoder(config)

        # Freeze vision encoder
        if config.freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def forward(self, images, input_ids, target_ids):
        # 1. Extract visual features (frozen)
        with torch.no_grad() if self.config.freeze_vision else torch.enable_grad():
            visual_features = self.vision_encoder(images)
        visual_tokens = self.projector(visual_features)

        # 2. Embed text tokens
        all_token_ids = torch.cat([input_ids, target_ids], dim=1)
        text_tokens = self.decoder.embed_tokens(all_token_ids)

        # 3. Concat and decode
        full_sequence = torch.cat([visual_tokens, text_tokens], dim=1)
        logits = self.decoder(full_sequence)

        return logits


# ─── Dataset ─────────────────────────────────────────────────────────────────

class VLMDataset(Dataset):
    """Multi-task VLM dataset for PyTorch."""

    def __init__(self, examples, images, max_input_len=20, max_target_len=256):
        self.examples = []
        self.images = images
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        for ex in examples:
            if ex["image_id"] in images:
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = self.images[ex["image_id"]]  # (224, 224, 3) float32

        # Convert to PyTorch format: (3, 224, 224)
        img = torch.tensor(img).permute(2, 0, 1)

        # Normalize with ImageNet stats (for pretrained ViT)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        input_ids = ex["input_ids"][:self.max_input_len]
        target_ids = ex["target_ids"][:self.max_target_len]

        return {
            "image": img,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "task": ex["task"],
        }


def collate_fn(batch):
    """Collate with padding."""
    images = torch.stack([b["image"] for b in batch])

    max_input = max(b["input_ids"].shape[0] for b in batch)
    max_target = max(b["target_ids"].shape[0] for b in batch)

    input_ids = torch.zeros(len(batch), max_input, dtype=torch.long)
    target_ids = torch.zeros(len(batch), max_target, dtype=torch.long)

    for i, b in enumerate(batch):
        input_ids[i, :b["input_ids"].shape[0]] = b["input_ids"]
        target_ids[i, :b["target_ids"].shape[0]] = b["target_ids"]

    tasks = [b["task"] for b in batch]
    return images, input_ids, target_ids, tasks


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_loss(logits, target_ids, num_patches, input_len):
    """Cross-entropy on target tokens only."""
    target_start = num_patches + input_len
    target_end = target_start + target_ids.shape[1]

    pred_logits = logits[:, target_start - 1:target_end - 1, :]
    mask = (target_ids != 0).float()

    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
        label_smoothing=0.0,
    ).reshape_as(target_ids)

    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum().clamp(min=1)


# ─── Training ────────────────────────────────────────────────────────────────

def train():
    global CONFIG

    print("=" * 60)
    print("Auto-VLM Training (PyTorch + Pretrained ViT)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    tokenizer = load_tokenizer()
    CONFIG.vocab_size = tokenizer.vocab_size
    CONFIG.__post_init__()

    train_examples = load_data("train")
    val_examples = load_data("val")
    images = load_images()

    # Upsample underrepresented tasks
    from collections import Counter
    task_counts = Counter(ex["task"] for ex in train_examples)
    max_count = max(task_counts.values())
    upsampled = list(train_examples)
    for task, count in task_counts.items():
        if count < max_count:
            task_exs = [ex for ex in train_examples if ex["task"] == task]
            repeat = max(1, round(max_count / count)) - 1
            upsampled.extend(task_exs * repeat)
    train_examples = upsampled

    print(f"  Train examples: {len(train_examples)} (after upsampling)")
    task_counts_new = Counter(ex["task"] for ex in train_examples)
    for t, c in sorted(task_counts_new.items()):
        print(f"    {t}: {c}")
    print(f"  Val examples: {len(val_examples)}")
    print(f"  Vocab size: {CONFIG.vocab_size}")

    # Datasets
    train_ds = VLMDataset(train_examples, images)
    val_ds = VLMDataset(val_examples, images)
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )

    # Model
    print("\nBuilding model...")
    model = MiniVLM(CONFIG).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  Frozen (vision): {frozen_params:,} ({frozen_params / 1e6:.2f}M)")

    # Optimizer: differential lr if vision is unfrozen
    if not CONFIG.freeze_vision:
        vision_lr = CONFIG.learning_rate * 0.01  # 100x smaller for pretrained vision
        param_groups = [
            {"params": model.vision_encoder.parameters(), "lr": vision_lr},
            {"params": model.projector.parameters(), "lr": CONFIG.learning_rate},
            {"params": model.decoder.parameters(), "lr": CONFIG.learning_rate},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG.weight_decay)
        print(f"  Vision lr: {vision_lr}, Decoder lr: {CONFIG.learning_rate}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=CONFIG.learning_rate,
            weight_decay=CONFIG.weight_decay,
        )

    # Cosine LR scheduler
    total_steps_est = int(len(train_ds) / CONFIG.batch_size * 3)  # ~3 epochs estimate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps_est, eta_min=CONFIG.learning_rate * 0.1
    )

    # Training loop
    print(f"\nTraining for {CONFIG.time_budget}s...")
    print(f"  Config: bs={CONFIG.batch_size}, lr={CONFIG.learning_rate}, "
          f"wd={CONFIG.weight_decay}, cosine_lr=True")
    print()

    start_time = time.time()
    step = 0
    epoch = 0
    train_losses = []

    model.train()
    while True:
        epoch += 1
        for batch in train_loader:
            if time.time() - start_time > CONFIG.time_budget:
                break

            images_b, input_ids, target_ids, tasks = batch
            images_b = images_b.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            logits = model(images_b, input_ids, target_ids)
            loss = compute_loss(logits, target_ids, CONFIG.num_patches, input_ids.shape[1])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CONFIG.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            step += 1

            if step % 10 == 0:
                avg_loss = np.mean(train_losses[-10:])
                elapsed = time.time() - start_time
                print(f"  Step {step:4d} | loss={avg_loss:.4f} | "
                      f"time={elapsed:.0f}s / {CONFIG.time_budget}s")

        if time.time() - start_time > CONFIG.time_budget:
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {step} steps, {epoch} epochs, {elapsed:.0f}s")

    # Validation
    print("\nRunning validation...")
    model.eval()
    val_losses = []
    val_task_losses = {}
    num_val = 0

    with torch.no_grad():
        for batch in val_loader:
            if num_val >= 200:
                break
            images_b, input_ids, target_ids, tasks = batch
            images_b = images_b.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            logits = model(images_b, input_ids, target_ids)
            loss = compute_loss(logits, target_ids, CONFIG.num_patches, input_ids.shape[1])

            val_losses.append(loss.item())
            for t in tasks:
                if t not in val_task_losses:
                    val_task_losses[t] = []
                val_task_losses[t].append(loss.item())
            num_val += 1

    val_loss = np.mean(val_losses) if val_losses else float("inf")
    print(f"\nVal loss: {val_loss:.4f}")
    for task, losses in sorted(val_task_losses.items()):
        print(f"  {task:12s}: {np.mean(losses):.4f} ({len(losses)} batches)")

    # Save checkpoint
    ckpt_dir = Path("checkpoint_torch")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    config_dict = asdict(CONFIG)
    config_dict["total_params"] = total_params
    config_dict["trainable_params"] = trainable_params
    config_dict["val_loss"] = float(val_loss)
    config_dict["train_steps"] = step
    config_dict["train_time"] = elapsed
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nCheckpoint saved to {ckpt_dir}/")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Params: {trainable_params / 1e6:.2f}M trainable, {total_params / 1e6:.2f}M total")

    return val_loss, trainable_params


if __name__ == "__main__":
    val_loss, params = train()
