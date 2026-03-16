#!/usr/bin/env python3
"""
Convert pretrained ViT-Tiny weights from timm/PyTorch to MLX format.
Saves weights that can be loaded into our VisionEncoder.

DeiT-Tiny: dim=192, depth=12, heads=3, patch_size=16
Our model: dim=192, depth=4, heads=3, patch_size=32 (or 16)

We load the patch embedding + first N layers from the pretrained model.
"""

import json
from pathlib import Path

import numpy as np

SAVE_DIR = Path("pretrained")


def convert_vit_tiny(num_layers=4, patch_size=16):
    """Convert DeiT-Tiny weights for our VisionEncoder."""
    import timm
    import torch

    print("Loading pretrained ViT-Tiny from timm...")
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model.eval()

    weights = {}

    # Patch embedding: Conv2d(3, 192, kernel_size=16, stride=16)
    # Our model: Conv2d(3, 192, kernel_size=patch_size, stride=patch_size)
    # timm Conv2d weight shape: (out_channels, in_channels, kH, kW)
    # MLX Conv2d weight shape: (out_channels, kH, kW, in_channels)
    patch_w = model.patch_embed.proj.weight.detach().numpy()  # (192, 3, 16, 16)
    patch_b = model.patch_embed.proj.bias.detach().numpy()    # (192,)

    if patch_size != 16:
        # Resize patch embedding if needed (e.g., 16→32)
        # Simple approach: average pool the kernel
        import torch.nn.functional as F
        pw_tensor = model.patch_embed.proj.weight.detach()
        pw_resized = F.adaptive_avg_pool2d(pw_tensor, (patch_size, patch_size))
        patch_w = pw_resized.numpy()

    # Convert from PyTorch (O, I, H, W) to MLX (O, H, W, I)
    patch_w = np.transpose(patch_w, (0, 2, 3, 1))
    weights["vision_encoder.patch_embed.proj.weight"] = patch_w
    weights["vision_encoder.patch_embed.proj.bias"] = patch_b

    # Position embedding: (1, num_patches + 1, 192) in DeiT (includes CLS token)
    # Our model: (1, num_patches, 192) — no CLS token
    pos_embed = model.pos_embed.detach().numpy()  # (1, 197, 192) for patch16
    # Skip CLS token position (index 0)
    pos_embed_no_cls = pos_embed[:, 1:, :]  # (1, 196, 192)

    if patch_size != 16:
        # Interpolate position embeddings for different number of patches
        num_patches_src = pos_embed_no_cls.shape[1]  # 196 for patch16
        h_src = int(num_patches_src ** 0.5)  # 14
        num_patches_tgt = (224 // patch_size) ** 2
        h_tgt = int(num_patches_tgt ** 0.5)

        # Reshape to 2D, interpolate, reshape back
        import torch.nn.functional as F
        pe = torch.tensor(pos_embed_no_cls)  # (1, 196, 192)
        pe = pe.reshape(1, h_src, h_src, 192).permute(0, 3, 1, 2)  # (1, 192, 14, 14)
        pe = F.interpolate(pe, size=(h_tgt, h_tgt), mode="bilinear", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, num_patches_tgt, 192)
        pos_embed_no_cls = pe.numpy()

    weights["vision_encoder.pos_embed"] = pos_embed_no_cls

    # Transformer blocks: load first num_layers blocks
    for i in range(num_layers):
        src_block = model.blocks[i]
        prefix = f"vision_encoder.blocks.{i}"

        # Attention: DeiT uses combined qkv projection
        # Our model uses separate q, k, v projections
        qkv_w = src_block.attn.qkv.weight.detach().numpy()  # (3*192, 192)
        qkv_b = src_block.attn.qkv.bias.detach().numpy()    # (3*192,)
        dim = 192
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)

        # PyTorch Linear: y = xW^T + b, weight shape (out, in)
        # MLX Linear: y = xW^T + b, weight shape (out, in) — same!
        weights[f"{prefix}.q_proj.weight"] = q_w
        weights[f"{prefix}.q_proj.bias"] = q_b
        weights[f"{prefix}.k_proj.weight"] = k_w
        weights[f"{prefix}.k_proj.bias"] = k_b
        weights[f"{prefix}.v_proj.weight"] = v_w
        weights[f"{prefix}.v_proj.bias"] = v_b

        # Output projection
        proj_w = src_block.attn.proj.weight.detach().numpy()
        proj_b = src_block.attn.proj.bias.detach().numpy()
        weights[f"{prefix}.out_proj.weight"] = proj_w
        weights[f"{prefix}.out_proj.bias"] = proj_b

        # MLP: DeiT uses Linear→GELU→Linear
        # Our model uses SwiGLU: w1, w2, w3
        # We can't directly map GELU MLP to SwiGLU, so we'll initialize
        # w1 and w3 from the first linear, and w2 from the second
        # This is approximate but better than random
        mlp_fc1_w = src_block.mlp.fc1.weight.detach().numpy()  # (768, 192)
        mlp_fc1_b = src_block.mlp.fc1.bias.detach().numpy()
        mlp_fc2_w = src_block.mlp.fc2.weight.detach().numpy()  # (192, 768)
        mlp_fc2_b = src_block.mlp.fc2.bias.detach().numpy()

        # SwiGLU: out = w2(silu(w1(x)) * w3(x))
        # Our SwiGLU mlp_dim = int(dim * 4.0 * 2 / 3) = 512
        # DeiT MLP hidden dim = 768
        # We need to truncate/pad to match our dimensions
        our_mlp_dim = int(192 * 4.0 * 2 / 3)  # 512

        # Truncate DeiT MLP weights to our SwiGLU dim
        w1_init = mlp_fc1_w[:our_mlp_dim, :]  # (512, 192)
        w1_b_init = mlp_fc1_b[:our_mlp_dim]
        w3_init = mlp_fc1_w[:our_mlp_dim, :]  # Same init for gate
        w3_b_init = mlp_fc1_b[:our_mlp_dim]
        w2_init = mlp_fc2_w[:, :our_mlp_dim]  # (192, 512)
        w2_b_init = mlp_fc2_b

        weights[f"{prefix}.w1.weight"] = w1_init
        weights[f"{prefix}.w1.bias"] = w1_b_init
        weights[f"{prefix}.w3.weight"] = w3_init
        weights[f"{prefix}.w3.bias"] = w3_b_init
        weights[f"{prefix}.w2.weight"] = w2_init
        weights[f"{prefix}.w2.bias"] = w2_b_init

        # Layer norms → RMSNorm (just use the weight, ignore bias)
        # DeiT norm1/norm2 have weight and bias; RMSNorm only has weight
        norm1_w = src_block.norm1.weight.detach().numpy()
        norm2_w = src_block.norm2.weight.detach().numpy()
        weights[f"{prefix}.norm1.weight"] = norm1_w
        weights[f"{prefix}.norm2.weight"] = norm2_w

    # Final norm
    final_norm_w = model.norm.weight.detach().numpy()
    weights["vision_encoder.norm.weight"] = final_norm_w

    # Save
    SAVE_DIR.mkdir(exist_ok=True)
    save_path = SAVE_DIR / "vit_tiny_weights.npz"
    np.savez(save_path, **weights)

    meta = {
        "source": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "dim": 192,
        "depth_loaded": num_layers,
        "depth_total": 12,
        "heads": 3,
        "patch_size": patch_size,
        "num_weights": len(weights),
    }
    with open(SAVE_DIR / "vit_tiny_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(weights)} weight tensors to {save_path}")
    print(f"Shapes:")
    for k, v in sorted(weights.items()):
        print(f"  {k}: {v.shape}")

    return weights


if __name__ == "__main__":
    import sys
    ps = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    nl = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    convert_vit_tiny(num_layers=nl, patch_size=ps)
