#!/usr/bin/env python3
"""
Generate sample predictions from the trained MiniVLM model.
Saves predictions as JSON for display on GitHub Pages.
Includes base64-encoded images and supports per-experiment snapshots.
"""

import base64
import io
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from prepare_data import (
    load_tokenizer, load_data, load_images,
    EOS_TOKEN, PAD_TOKEN, IMAGE_SIZE,
)

# Fixed set of image IDs for consistent comparison across experiments.
# These are chosen after data prep; we pick the first valid ID per task.
FIXED_SAMPLE_IDS_FILE = Path("data/processed/sample_ids.json")


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


def image_to_base64(arr):
    """Convert a float32 [0,1] numpy image to a base64 JPEG string."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def greedy_decode(model, image, input_ids, tokenizer, config, max_new_tokens=64):
    """Greedy autoregressive decoding."""
    eos_id = tokenizer.token_to_id[EOS_TOKEN]
    pad_id = tokenizer.token_to_id[PAD_TOKEN]

    image_mx = mx.array(image[np.newaxis])
    generated = []
    current_input = mx.array([input_ids])

    for _ in range(max_new_tokens):
        if not generated:
            target = mx.array([[pad_id]])
        else:
            target = mx.array([generated])

        logits = model(image_mx, current_input, target)
        next_logits = logits[0, -1, :]
        next_token = mx.argmax(next_logits).item()

        if next_token == eos_id:
            break
        generated.append(next_token)

    return generated


def format_prediction(token_ids, tokenizer):
    """Convert token IDs to readable text."""
    tokens = [tokenizer.id_to_token.get(tid, "<unk>") for tid in token_ids]
    text = " ".join(tokens)
    text = re.sub(r'<loc(\d{3})>', r'[\1]', text)
    text = text.replace("<box>", "[BOX: ").replace("</box>", "]")
    text = text.replace("<poly>", "[POLY: ").replace("</poly>", "]")
    text = text.replace("<sep>", " | ")
    text = text.replace("<eos>", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_fixed_sample_ids(val_examples, images, num_per_task=3):
    """
    Return a fixed set of (image_id, example_index) per task so every
    experiment is evaluated on the exact same inputs.
    """
    if FIXED_SAMPLE_IDS_FILE.exists():
        with open(FIXED_SAMPLE_IDS_FILE) as f:
            return json.load(f)

    # First time: pick examples deterministically
    task_picks = {}
    for idx, ex in enumerate(val_examples):
        task = ex["task"]
        if task not in task_picks:
            task_picks[task] = []
        if ex["image_id"] in images and len(task_picks[task]) < num_per_task:
            task_picks[task].append({"index": idx, "image_id": ex["image_id"]})

    FIXED_SAMPLE_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXED_SAMPLE_IDS_FILE, "w") as f:
        json.dump(task_picks, f, indent=2)
    return task_picks


def generate_samples(num_per_task=3, run_num=None):
    """
    Generate sample predictions for each task.
    If run_num is given, saves to samples_NNN.json and updates samples.json
    (which always points to the latest).
    """
    print("Loading model...")
    model, config = load_model()
    if model is None:
        return {}

    print("Loading data...")
    tokenizer = load_tokenizer()
    val_examples = load_data("val")
    images = load_images()

    fixed_ids = get_fixed_sample_ids(val_examples, images, num_per_task)

    samples = {}

    for task, picks in fixed_ids.items():
        print(f"\nGenerating {task} samples...")
        task_samples = []

        for pick in picks:
            idx = pick["index"]
            img_id = pick["image_id"]

            if img_id not in images:
                continue

            ex = val_examples[idx]
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

            # Input description
            if task == "vqa":
                input_text = format_prediction(input_ids, tokenizer)
            else:
                input_text = f"<{task}>"

            # Image as base64
            img_b64 = image_to_base64(image)

            sample = {
                "image_id": img_id,
                "task": task,
                "input": input_text,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "image_b64": img_b64,
            }
            task_samples.append(sample)
            print(f"  [{task}] img={img_id}: pred='{pred_text[:80]}...'")

        samples[task] = task_samples

    return samples


def save_samples(samples, run_num=None):
    """
    Save samples.  Always writes samples.json (latest).
    If run_num given, also writes samples_NNN.json and updates the index.
    """
    # Always overwrite latest
    with open("samples.json", "w") as f:
        json.dump({"run": run_num, "samples": samples}, f)

    # Per-experiment snapshot
    if run_num is not None:
        fname = f"samples_{run_num:03d}.json"
        with open(fname, "w") as f:
            json.dump({"run": run_num, "samples": samples}, f)

        # Update the manifest (list of available experiment samples)
        manifest_path = Path("samples_manifest.json")
        manifest = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        if run_num not in manifest:
            manifest.append(run_num)
            manifest.sort()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

    total = sum(len(v) for v in samples.values())
    print(f"\nSaved {total} samples" +
          (f" (run {run_num:03d})" if run_num else ""))


if __name__ == "__main__":
    import sys
    run = int(sys.argv[1]) if len(sys.argv) > 1 else None
    samples = generate_samples(num_per_task=3, run_num=run)
    save_samples(samples, run_num=run)
