#!/usr/bin/env python3
"""
Generate sample predictions from the trained MiniVLM model.
Saves predictions as JSON for display on GitHub Pages.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from prepare_data import (
    load_tokenizer, load_data, load_images, load_metadata,
    CAPTION_TOKEN, VQA_TOKEN, OD_TOKEN, SEG_TOKEN, DESCRIBE_TOKEN,
    EOS_TOKEN, SEP_TOKEN, PAD_TOKEN,
)


def load_model():
    """Load trained model from checkpoint."""
    from train_mlx import MiniVLM, Config

    config_path = Path("checkpoint/config.json")
    if not config_path.exists():
        print("No checkpoint found.")
        return None, None

    with open(config_path) as f:
        config_dict = json.load(f)

    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.__post_init__()

    model = MiniVLM(config)
    model.load_weights("checkpoint/model.safetensors")
    model.eval()
    mx.eval(model.parameters())

    return model, config


def greedy_decode(model, image, input_ids, tokenizer, config, max_new_tokens=64):
    """Greedy autoregressive decoding."""
    eos_id = tokenizer.token_to_id[EOS_TOKEN]
    pad_id = tokenizer.token_to_id[PAD_TOKEN]

    # Prepare image: (1, H, W, 3)
    image_mx = mx.array(image[np.newaxis])

    generated = []
    current_input = mx.array([input_ids])

    for _ in range(max_new_tokens):
        # Build target (what we've generated so far, or just a dummy start)
        if not generated:
            target = mx.array([[pad_id]])
        else:
            target = mx.array([generated])

        logits = model(image_mx, current_input, target)

        # Get the last token prediction
        # logits shape: (1, num_patches + input_len + target_len, vocab_size)
        next_logits = logits[0, -1, :]
        next_token = mx.argmax(next_logits).item()

        if next_token == eos_id:
            break

        generated.append(next_token)

    return generated


def format_prediction(token_ids, tokenizer):
    """Convert token IDs to readable text."""
    tokens = []
    for tid in token_ids:
        token = tokenizer.id_to_token.get(tid, "<unk>")
        tokens.append(token)

    # Join and clean up
    text = " ".join(tokens)
    # Clean up location tokens for display
    import re
    text = re.sub(r'<loc(\d{3})>', r'[\1]', text)
    text = text.replace("<box>", "[BOX: ").replace("</box>", "] ")
    text = text.replace("<poly>", "[POLY: ").replace("</poly>", "] ")
    text = text.replace("<sep>", " | ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_samples(num_per_task=3):
    """Generate sample predictions for each task."""
    print("Loading model...")
    model, config = load_model()
    if model is None:
        return {}

    print("Loading data...")
    tokenizer = load_tokenizer()
    val_examples = load_data("val")
    images = load_images()

    # Group examples by task
    task_examples = {}
    for ex in val_examples:
        task = ex["task"]
        if task not in task_examples:
            task_examples[task] = []
        if ex["image_id"] in images and len(task_examples[task]) < num_per_task * 3:
            task_examples[task].append(ex)

    samples = {}
    task_tokens = {
        "caption": CAPTION_TOKEN,
        "vqa": VQA_TOKEN,
        "od": OD_TOKEN,
        "seg": SEG_TOKEN,
        "describe": DESCRIBE_TOKEN,
    }

    for task, examples in task_examples.items():
        print(f"\nGenerating {task} samples...")
        task_samples = []
        count = 0

        for ex in examples:
            if count >= num_per_task:
                break

            img_id = ex["image_id"]
            if img_id not in images:
                continue

            image = images[img_id]
            input_ids = ex["input_ids"]

            # Ground truth
            gt_text = format_prediction(ex["target_ids"], tokenizer)

            # Model prediction
            try:
                pred_ids = greedy_decode(model, image, input_ids, tokenizer, config)
                pred_text = format_prediction(pred_ids, tokenizer)
            except Exception as e:
                pred_text = f"(error: {e})"

            # Build input description
            if task == "vqa":
                input_text = format_prediction(input_ids, tokenizer)
            else:
                input_text = f"<{task}>"

            sample = {
                "image_id": img_id,
                "task": task,
                "input": input_text,
                "ground_truth": gt_text,
                "prediction": pred_text,
            }
            task_samples.append(sample)
            count += 1
            print(f"  [{task}] img={img_id}: pred='{pred_text[:80]}...'")

        samples[task] = task_samples

    return samples


def save_samples(samples, path="samples.json"):
    """Save samples to JSON."""
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved {sum(len(v) for v in samples.values())} samples to {path}")


if __name__ == "__main__":
    samples = generate_samples(num_per_task=3)
    save_samples(samples)
