#!/usr/bin/env python3
"""
Train a tiny multi-task VLM from scratch in MLX.

THIS FILE IS EDITABLE — the autonomous research agent modifies this file
to experiment with architectures, hyperparameters, and loss functions.

Tasks: Captioning, VQA, Object Detection, Segmentation, Free-text Description
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from prepare_data import load_tokenizer, load_data, load_images, load_metadata

# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    # Vision encoder
    image_size: int = 224
    patch_size: int = 32
    vision_dim: int = 192
    vision_depth: int = 3
    vision_heads: int = 3

    # Language decoder
    lang_dim: int = 256
    lang_depth: int = 4
    lang_heads: int = 4
    max_seq_len: int = 320  # input + target combined

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 200
    dropout: float = 0.1
    grad_clip: float = 1.0
    time_budget: int = 600  # seconds

    # Loss weights
    ce_weight: float = 1.0
    giou_weight: float = 0.0  # Set > 0 to enable GIoU loss for OD
    dice_weight: float = 0.0  # Set > 0 to enable Dice loss for seg

    # Data
    vocab_size: int = 0  # Set from tokenizer
    num_patches: int = 0  # Computed from image_size / patch_size

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2


CONFIG = Config()

# ─── Model Components ────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings using Conv2d."""

    def __init__(self, patch_size: int, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def __call__(self, x):
        # x: (B, H, W, C) — MLX uses channels-last
        x = self.proj(x)  # (B, H/P, W/P, embed_dim)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def attention(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        if self.causal:
            causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            attn = attn + causal_mask

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)

    def __call__(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VisionEncoder(nn.Module):
    """Tiny ViT encoder for image feature extraction."""

    def __init__(self, config: Config):
        super().__init__()
        self.patch_embed = PatchEmbedding(config.patch_size, 3, config.vision_dim)
        self.pos_embed = mx.zeros((1, config.num_patches, config.vision_dim))
        self.blocks = [
            TransformerBlock(config.vision_dim, config.vision_heads,
                           dropout=config.dropout, causal=False)
            for _ in range(config.vision_depth)
        ]
        self.norm = nn.LayerNorm(config.vision_dim)

    def __call__(self, images):
        x = self.patch_embed(images)  # (B, num_patches, vision_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class VisionProjector(nn.Module):
    """Project vision features to language model dimension."""

    def __init__(self, vision_dim: int, lang_dim: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, lang_dim)

    def __call__(self, x):
        return self.proj(x)


class LanguageDecoder(nn.Module):
    """Small GPT-style causal transformer decoder."""

    def __init__(self, config: Config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.lang_dim)
        self.pos_embed = mx.zeros((1, config.max_seq_len, config.lang_dim))
        self.blocks = [
            TransformerBlock(config.lang_dim, config.lang_heads,
                           dropout=config.dropout, causal=True)
            for _ in range(config.lang_depth)
        ]
        self.norm = nn.LayerNorm(config.lang_dim)
        self.head = nn.Linear(config.lang_dim, config.vocab_size)

    def __call__(self, x, offset=0):
        """
        x: (B, T, lang_dim) — already embedded (visual + text tokens combined)
        offset: position offset for the sequence
        """
        T = x.shape[1]
        positions = self.pos_embed[:, offset:offset + T, :]
        x = x + positions
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def embed_tokens(self, token_ids):
        """Embed token IDs."""
        return self.token_embed(token_ids)


class MiniVLM(nn.Module):
    """
    Tiny multi-task Vision-Language Model.

    Architecture:
    - Vision encoder extracts patch features from images
    - Projector maps vision features to language dimension
    - Language decoder autoregressively generates task output

    All tasks use the same architecture with task-specific tokens:
    [visual_tokens] [task_token] [input_text_tokens] → [target_tokens]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.projector = VisionProjector(config.vision_dim, config.lang_dim)
        self.decoder = LanguageDecoder(config)

    def __call__(self, images, input_ids, target_ids):
        """
        Forward pass.

        Args:
            images: (B, H, W, 3) float32 images
            input_ids: (B, T_in) input token IDs (task token + optional text)
            target_ids: (B, T_tgt) target token IDs

        Returns:
            logits: (B, T_total, vocab_size) over the full sequence
        """
        # 1. Encode vision
        visual_features = self.vision_encoder(images)  # (B, num_patches, vision_dim)
        visual_tokens = self.projector(visual_features)  # (B, num_patches, lang_dim)

        # 2. Embed text tokens
        # Combine input and target for teacher forcing
        # Sequence: [visual_tokens, input_ids, target_ids]
        # We predict the next token for each position
        all_token_ids = mx.concatenate([input_ids, target_ids], axis=1)  # (B, T_in + T_tgt)
        text_tokens = self.decoder.embed_tokens(all_token_ids)  # (B, T_text, lang_dim)

        # 3. Concatenate visual and text tokens
        full_sequence = mx.concatenate([visual_tokens, text_tokens], axis=1)  # (B, T_total, lang_dim)

        # 4. Run through decoder (causal attention over full sequence)
        logits = self.decoder(full_sequence)  # (B, T_total, vocab_size)

        return logits

    def count_params(self):
        """Count total trainable parameters."""
        nparams = sum(p.size for _, p in self.parameters().items()
                      if isinstance(p, mx.array))
        # Also count in nested modules
        total = 0
        for k, v in self.trainable_parameters().items():
            if isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for vv in item.values():
                            if isinstance(vv, mx.array):
                                total += vv.size
                            elif isinstance(vv, list):
                                for vvv in vv:
                                    if isinstance(vvv, dict):
                                        for vvvv in vvvv.values():
                                            if isinstance(vvvv, mx.array):
                                                total += vvvv.size
                                    elif isinstance(vvv, mx.array):
                                        total += vvv.size
                    elif isinstance(item, mx.array):
                        total += item.size
        return total


def count_parameters(model):
    """Count all parameters in a model."""
    leaf_params = model.parameters()
    total = 0

    def _count(obj):
        nonlocal total
        if isinstance(obj, mx.array):
            total += obj.size
        elif isinstance(obj, dict):
            for v in obj.values():
                _count(v)
        elif isinstance(obj, list):
            for v in obj:
                _count(v)

    _count(leaf_params)
    return total


# ─── Data Loading ────────────────────────────────────────────────────────────

class MultiTaskDataset:
    """Loads and batches multi-task VLM training data."""

    def __init__(self, examples, images, config: Config, shuffle=True):
        self.examples = examples
        self.images = images
        self.config = config
        self.shuffle = shuffle
        self.pad_id = 0  # <pad>

    def __len__(self):
        return len(self.examples)

    def get_batches(self, rng=None):
        """Yield batches of (images, input_ids, target_ids, labels)."""
        if rng is None:
            rng = np.random.RandomState()

        indices = np.arange(len(self.examples))
        if self.shuffle:
            rng.shuffle(indices)

        batch_size = self.config.batch_size
        num_patches = self.config.num_patches

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if len(batch_indices) < batch_size:
                continue  # Skip incomplete last batch

            batch_images = []
            batch_input_ids = []
            batch_target_ids = []
            batch_tasks = []

            for idx in batch_indices:
                ex = self.examples[idx]
                img_id = ex["image_id"]

                if img_id not in self.images:
                    continue

                batch_images.append(self.images[img_id])
                batch_input_ids.append(ex["input_ids"])
                batch_target_ids.append(ex["target_ids"])
                batch_tasks.append(ex["task"])

            if len(batch_images) < batch_size:
                continue

            # Pad sequences
            max_input_len = max(len(ids) for ids in batch_input_ids)
            max_target_len = max(len(ids) for ids in batch_target_ids)

            # Limit total sequence length
            max_text_len = self.config.max_seq_len - num_patches
            if max_input_len + max_target_len > max_text_len:
                max_target_len = min(max_target_len, max_text_len - max_input_len)
                if max_target_len <= 0:
                    continue

            padded_inputs = np.full((batch_size, max_input_len), self.pad_id, dtype=np.int32)
            padded_targets = np.full((batch_size, max_target_len), self.pad_id, dtype=np.int32)

            for i in range(batch_size):
                inp = batch_input_ids[i][:max_input_len]
                tgt = batch_target_ids[i][:max_target_len]
                padded_inputs[i, :len(inp)] = inp
                padded_targets[i, :len(tgt)] = tgt

            images_arr = np.stack(batch_images)  # (B, H, W, 3)

            yield (
                mx.array(images_arr),
                mx.array(padded_inputs),
                mx.array(padded_targets),
                batch_tasks,
            )


# ─── Loss Functions ──────────────────────────────────────────────────────────

def compute_ce_loss(logits, targets, num_patches, input_len):
    """
    Compute cross-entropy loss on target tokens only.

    logits: (B, T_total, vocab_size) where T_total = num_patches + input_len + target_len
    targets: (B, target_len) target token IDs
    """
    # The target tokens start at position (num_patches + input_len)
    # But we predict the NEXT token, so logits at position t predict token at t+1
    # For the text portion: logits at [num_patches + 0 ... num_patches + input_len + target_len - 2]
    # predict tokens [input[1], ..., input[-1], target[0], ..., target[-1]]
    # We only care about predicting target tokens.

    target_start = num_patches + input_len  # logits position that predicts first target token
    target_end = target_start + targets.shape[1]

    # Get logits for target prediction (shifted by 1)
    pred_logits = logits[:, target_start - 1:target_end - 1, :]  # (B, target_len, vocab_size)

    # Mask out padding (target_id == 0 means pad)
    mask = (targets != 0).astype(mx.float32)

    # Cross entropy
    log_probs = nn.losses.cross_entropy(pred_logits, targets, reduction="none")  # (B, target_len)
    masked_loss = log_probs * mask

    # Mean over non-pad tokens
    num_tokens = mx.maximum(mask.sum(), mx.array(1.0))
    loss = masked_loss.sum() / num_tokens

    return loss


def loss_fn(model, images, input_ids, target_ids):
    """Compute the training loss."""
    logits = model(images, input_ids, target_ids)

    num_patches = (CONFIG.image_size // CONFIG.patch_size) ** 2
    input_len = input_ids.shape[1]

    ce_loss = compute_ce_loss(logits, target_ids, num_patches, input_len)

    return ce_loss


# ─── Learning Rate Schedule ──────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_lr, total_steps):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Training Loop ───────────────────────────────────────────────────────────

def train():
    """Main training function."""
    global CONFIG

    print("=" * 60)
    print("Auto-VLM Training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    tokenizer = load_tokenizer()
    CONFIG.vocab_size = tokenizer.vocab_size
    CONFIG.__post_init__()

    train_examples = load_data("train")
    val_examples = load_data("val")
    images = load_images()

    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")
    print(f"  Images loaded: {len(images)}")
    print(f"  Vocab size: {CONFIG.vocab_size}")

    # Create datasets
    train_dataset = MultiTaskDataset(train_examples, images, CONFIG, shuffle=True)
    val_dataset = MultiTaskDataset(val_examples, images, CONFIG, shuffle=True)

    # Build model
    print("\nBuilding model...")
    model = MiniVLM(CONFIG)

    # Initialize parameters
    mx.eval(model.parameters())
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    if num_params > 10_000_000:
        print(f"  WARNING: Model exceeds 10M parameter limit ({num_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )

    # Compile loss + grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training
    print(f"\nTraining for {CONFIG.time_budget}s...")
    print(f"  Config: bs={CONFIG.batch_size}, lr={CONFIG.learning_rate}, "
          f"wd={CONFIG.weight_decay}, dropout={CONFIG.dropout}")
    print(f"  Vision: dim={CONFIG.vision_dim}, depth={CONFIG.vision_depth}, "
          f"heads={CONFIG.vision_heads}, patch={CONFIG.patch_size}")
    print(f"  Language: dim={CONFIG.lang_dim}, depth={CONFIG.lang_depth}, "
          f"heads={CONFIG.lang_heads}")
    print()

    start_time = time.time()
    step = 0
    epoch = 0
    best_val_loss = float("inf")
    train_losses = []

    while True:
        epoch += 1
        rng = np.random.RandomState(epoch)

        for batch in train_dataset.get_batches(rng):
            if time.time() - start_time > CONFIG.time_budget:
                break

            images_batch, input_ids, target_ids, tasks = batch

            # Forward + backward
            loss, grads = loss_and_grad_fn(model, images_batch, input_ids, target_ids)

            # Gradient clipping
            grad_norm = optim.clip_grad_norm(grads, CONFIG.grad_clip)

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            train_losses.append(loss_val)
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
    val_losses = []
    val_task_losses = {}
    model.eval()

    num_val_batches = 0
    for batch in val_dataset.get_batches(rng=np.random.RandomState(42)):
        if num_val_batches >= 200:  # Cap validation
            break
        images_batch, input_ids, target_ids, tasks = batch
        logits = model(images_batch, input_ids, target_ids)
        num_patches = (CONFIG.image_size // CONFIG.patch_size) ** 2
        input_len = input_ids.shape[1]
        loss = compute_ce_loss(logits, target_ids, num_patches, input_len)
        mx.eval(loss)
        loss_val = loss.item()
        val_losses.append(loss_val)

        # Track per-task losses
        for task in tasks:
            if task not in val_task_losses:
                val_task_losses[task] = []
            val_task_losses[task].append(loss_val)

        num_val_batches += 1

    val_loss = np.mean(val_losses) if val_losses else float("inf")
    print(f"\nVal loss: {val_loss:.4f}")

    # Per-task validation losses
    for task, losses in sorted(val_task_losses.items()):
        print(f"  {task:12s}: {np.mean(losses):.4f} ({len(losses)} batches)")

    # Save checkpoint
    checkpoint_dir = Path("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)

    # Save model weights
    model.save_weights(str(checkpoint_dir / "model.safetensors"))

    # Save config
    config_dict = asdict(CONFIG)
    config_dict["num_params"] = num_params
    config_dict["val_loss"] = float(val_loss)
    config_dict["train_steps"] = step
    config_dict["train_time"] = elapsed
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nCheckpoint saved to {checkpoint_dir}/")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Params: {num_params / 1e6:.2f}M")

    return val_loss, num_params


if __name__ == "__main__":
    val_loss, num_params = train()
