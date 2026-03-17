#!/usr/bin/env python3
"""
Download COCO train2017 subset and add to training data.
This supplements the val2017 data with more diverse examples,
especially important for OD and Segmentation tasks.
"""

import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from prepare_data import (
    SimpleTokenizer, load_tokenizer, load_metadata,
    coord_to_token, bbox_to_tokens, polygon_to_tokens,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN,
    CAPTION_TOKEN, VQA_TOKEN, OD_TOKEN, SEG_TOKEN, DESCRIBE_TOKEN,
    BOX_OPEN, BOX_CLOSE, POLY_OPEN, POLY_CLOSE,
    IMAGE_SIZE,
)

DATA_DIR = Path("data")
TRAIN_IMAGE_DIR = DATA_DIR / "images" / "train2017"
ANN_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"

MAX_TRAIN_IMAGES = 20000  # Use 20K out of 118K


def download_coco_train():
    """Download COCO train2017 images."""
    images_zip = DATA_DIR / "train2017.zip"
    if TRAIN_IMAGE_DIR.exists() and len(list(TRAIN_IMAGE_DIR.glob("*.jpg"))) > 1000:
        print(f"  train2017 images already exist ({len(list(TRAIN_IMAGE_DIR.glob('*.jpg')))} images)")
        return

    if not images_zip.exists():
        print(f"  Downloading COCO train2017 (~18GB)...")
        response = requests.get(COCO_TRAIN_URL, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(images_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (100 * 1024 * 1024) < 65536:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="", flush=True)
        print()

    print("  Extracting train2017...")
    with zipfile.ZipFile(images_zip) as zf:
        zf.extractall(DATA_DIR / "images")
    print("  Done.")


def download_train_annotations():
    """Ensure train annotations are available."""
    captions_file = ANN_DIR / "captions_train2017.json"
    instances_file = ANN_DIR / "instances_train2017.json"

    if captions_file.exists() and instances_file.exists():
        print("  Train annotations already exist.")
        return

    ann_zip = DATA_DIR / "annotations.zip"
    if not ann_zip.exists():
        print("  Downloading annotations...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(ann_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (50 * 1024 * 1024) < 65536:
                    print(f"\r  {downloaded // 1024 // 1024}MB", end="", flush=True)
        print()

    print("  Extracting train annotations...")
    with zipfile.ZipFile(ann_zip) as zf:
        for name in zf.namelist():
            if "train2017" in name:
                zf.extract(name, DATA_DIR)
    print("  Done.")


def process_train_subset():
    """Process COCO train2017 subset and add to training data."""
    print("\n=== Preparing COCO train2017 subset ===")

    # Download
    download_coco_train()
    download_train_annotations()

    # Load annotations
    print("\nLoading train annotations...")
    with open(ANN_DIR / "captions_train2017.json") as f:
        captions_data = json.load(f)
    with open(ANN_DIR / "instances_train2017.json") as f:
        instances_data = json.load(f)

    id_to_cat = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    id_to_filename = {img["id"]: img["file_name"] for img in instances_data["images"]}
    img_dims = {img["id"]: (img["width"], img["height"]) for img in instances_data["images"]}

    # Select subset of images (deterministic)
    all_train_ids = sorted(id_to_filename.keys())
    rng = np.random.RandomState(123)
    rng.shuffle(all_train_ids)
    subset_ids = set(all_train_ids[:MAX_TRAIN_IMAGES])
    print(f"  Selected {len(subset_ids)} train images")

    # Preprocess images
    cache_path = PROCESSED_DIR / "train_images.npz"
    if cache_path.exists():
        print("  Loading cached train images...")
        data = np.load(cache_path)
        train_images = {int(k): data[k] for k in data.files}
        print(f"  Loaded {len(train_images)} images")
    else:
        print(f"  Processing {len(subset_ids)} train images...")
        train_images = {}
        for i, img_id in enumerate(sorted(subset_ids)):
            fname = id_to_filename[img_id]
            img_path = TRAIN_IMAGE_DIR / fname
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                train_images[img_id] = np.array(img, dtype=np.float32) / 255.0
            except Exception:
                continue
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(subset_ids)}")

        np.savez_compressed(cache_path, **{str(k): v for k, v in train_images.items()})
        print(f"  Saved {len(train_images)} images")

    # Load existing tokenizer
    tokenizer = load_tokenizer()

    # Generate examples for train subset
    print("\nGenerating task examples for train subset...")

    examples = []

    # Captions
    for ann in captions_data["annotations"]:
        if ann["image_id"] not in subset_ids:
            continue
        input_ids = tokenizer.encode([CAPTION_TOKEN])
        target_ids = tokenizer.encode(ann["caption"].strip()) + [tokenizer.token_to_id[EOS_TOKEN]]
        if len(target_ids) > 256:
            continue
        examples.append({
            "image_id": ann["image_id"], "task": "caption",
            "input_ids": input_ids, "target_ids": target_ids,
        })

    # OD
    img_anns = {}
    for ann in instances_data["annotations"]:
        if ann.get("iscrowd", 0) or ann["image_id"] not in subset_ids:
            continue
        img_anns.setdefault(ann["image_id"], []).append(ann)

    for img_id, anns in img_anns.items():
        if img_id not in img_dims:
            continue
        w, h = img_dims[img_id]
        target_tokens = []
        for ann in anns[:10]:
            cat_name = id_to_cat[ann["category_id"]]
            box_tokens = bbox_to_tokens(ann["bbox"], w, h)
            if target_tokens:
                target_tokens.append(SEP_TOKEN)
            target_tokens.append(cat_name)
            target_tokens.append(BOX_OPEN)
            target_tokens.extend(box_tokens)
            target_tokens.append(BOX_CLOSE)
        target_tokens.append(EOS_TOKEN)
        target_ids = tokenizer.encode(target_tokens)
        if len(target_ids) > 256:
            continue
        examples.append({
            "image_id": img_id, "task": "od",
            "input_ids": tokenizer.encode([OD_TOKEN]), "target_ids": target_ids,
        })

    # Segmentation
    for img_id, anns in img_anns.items():
        if img_id not in img_dims:
            continue
        w, h = img_dims[img_id]
        target_tokens = []
        for ann in anns[:5]:
            if not ann.get("segmentation") or isinstance(ann["segmentation"], dict):
                continue
            cat_name = id_to_cat[ann["category_id"]]
            poly_tokens = polygon_to_tokens(ann["segmentation"], w, h, max_vertices=12)
            if not poly_tokens:
                continue
            if target_tokens:
                target_tokens.append(SEP_TOKEN)
            target_tokens.append(cat_name)
            target_tokens.append(POLY_OPEN)
            target_tokens.extend(poly_tokens)
            target_tokens.append(POLY_CLOSE)
        if not target_tokens:
            continue
        target_tokens.append(EOS_TOKEN)
        target_ids = tokenizer.encode(target_tokens)
        if len(target_ids) > 256:
            continue
        examples.append({
            "image_id": img_id, "task": "seg",
            "input_ids": tokenizer.encode([SEG_TOKEN]), "target_ids": target_ids,
        })

    # VQA (generated)
    for img_id, anns in img_anns.items():
        cats = [id_to_cat[a["category_id"]] for a in anns]
        cat_counts = Counter(cats)
        for cat, count in list(cat_counts.items())[:3]:
            q = f"how many {cat} are in the image?"
            input_ids = tokenizer.encode([VQA_TOKEN]) + tokenizer.encode(q) + [tokenizer.token_to_id[SEP_TOKEN]]
            target_ids = tokenizer.encode(str(count)) + [tokenizer.token_to_id[EOS_TOKEN]]
            examples.append({
                "image_id": img_id, "task": "vqa",
                "input_ids": input_ids, "target_ids": target_ids,
            })

    # Describe (longest caption)
    img_caps = {}
    for ann in captions_data["annotations"]:
        if ann["image_id"] not in subset_ids:
            continue
        img_caps.setdefault(ann["image_id"], []).append(ann["caption"].strip())
    for img_id, caps in img_caps.items():
        desc = max(caps, key=len)
        input_ids = tokenizer.encode([DESCRIBE_TOKEN])
        target_ids = tokenizer.encode(desc) + [tokenizer.token_to_id[EOS_TOKEN]]
        if len(target_ids) > 256:
            continue
        examples.append({
            "image_id": img_id, "task": "describe",
            "input_ids": input_ids, "target_ids": target_ids,
        })

    # Filter to images we actually have
    examples = [ex for ex in examples if ex["image_id"] in train_images]

    task_counts = Counter(ex["task"] for ex in examples)
    print(f"\n  Train subset examples: {len(examples)}")
    for t, c in sorted(task_counts.items()):
        print(f"    {t}: {c}")

    # Save
    with open(PROCESSED_DIR / "train_extra.json", "w") as f:
        json.dump(examples, f)

    print(f"\n  Saved to {PROCESSED_DIR / 'train_extra.json'}")
    print(f"  Images: {PROCESSED_DIR / 'train_images.npz'}")
    return examples, train_images


if __name__ == "__main__":
    process_train_subset()
