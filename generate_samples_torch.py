#!/usr/bin/env python3
"""Generate sample predictions from the PyTorch VLM model."""

import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from prepare_data import (
    load_tokenizer, load_data, load_images,
    EOS_TOKEN, PAD_TOKEN,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FIXED_SAMPLE_IDS_FILE = Path("data/processed/sample_ids.json")


def load_model():
    from train_torch import MiniVLM, Config
    config_path = Path("checkpoint_torch/config.json")
    if not config_path.exists():
        return None, None
    with open(config_path) as f:
        cd = json.load(f)
    config = Config()
    for k, v in cd.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.__post_init__()
    model = MiniVLM(config).to(DEVICE)
    model.load_state_dict(torch.load("checkpoint_torch/model.pt", map_location=DEVICE, weights_only=False))
    model.eval()
    return model, config


def image_to_base64(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_image(arr):
    """Convert (224,224,3) float [0,1] to PyTorch tensor with ImageNet normalization."""
    t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((t - mean) / std).to(DEVICE)


@torch.no_grad()
def greedy_decode(model, image_tensor, input_ids, tokenizer, config, max_new=64):
    eos_id = tokenizer.token_to_id[EOS_TOKEN]
    pad_id = tokenizer.token_to_id[PAD_TOKEN]
    inp = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    generated = []
    for _ in range(max_new):
        tgt = torch.tensor([generated or [pad_id]], dtype=torch.long, device=DEVICE)
        logits = model(image_tensor, inp, tgt)
        next_tok = logits[0, -1, :].argmax().item()
        if next_tok == eos_id:
            break
        generated.append(next_tok)
    return generated


def format_prediction(ids, tokenizer):
    tokens = [tokenizer.id_to_token.get(i, "<unk>") for i in ids]
    text = " ".join(tokens)
    text = re.sub(r'<loc(\d{3})>', r'[\1]', text)
    text = text.replace("<box>", "[BOX: ").replace("</box>", "]")
    text = text.replace("<poly>", "[POLY: ").replace("</poly>", "]")
    text = text.replace("<sep>", " | ").replace("<eos>", "")
    return re.sub(r'\s+', ' ', text).strip()


def parse_boxes(ids, tokenizer):
    """Parse predicted token IDs into list of (class_name, [x1,y1,x2,y2]) tuples."""
    tokens = [tokenizer.id_to_token.get(i, "") for i in ids]
    objects = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Skip special tokens
        if tok in ("<sep>", "<eos>", "<od>", "<pad>", ""):
            i += 1; continue
        # Check if this is a class name followed by <box>
        if not tok.startswith("<loc") and not tok.startswith("<"):
            class_name = tok
            i += 1
            # Look for <box> then 4 loc tokens then </box>
            if i < len(tokens) and tokens[i] == "<box>":
                i += 1
                coords = []
                while i < len(tokens) and tokens[i] != "</box>" and len(coords) < 4:
                    m = re.match(r'<loc(\d{3})>', tokens[i])
                    if m:
                        coords.append(int(m.group(1)) / 999.0 * 224)
                    i += 1
                if tokens[i:i+1] == ["</box>"]:
                    i += 1
                if len(coords) == 4:
                    objects.append((class_name, coords))
            continue
        i += 1
    return objects


def parse_polygons(ids, tokenizer):
    """Parse predicted token IDs into list of (class_name, [(x,y),...]) tuples."""
    tokens = [tokenizer.id_to_token.get(i, "") for i in ids]
    objects = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("<sep>", "<eos>", "<seg>", "<pad>", ""):
            i += 1; continue
        if not tok.startswith("<loc") and not tok.startswith("<"):
            class_name = tok
            i += 1
            if i < len(tokens) and tokens[i] == "<poly>":
                i += 1
                coords = []
                while i < len(tokens) and tokens[i] != "</poly>":
                    m = re.match(r'<loc(\d{3})>', tokens[i])
                    if m:
                        coords.append(int(m.group(1)) / 999.0 * 224)
                    i += 1
                if tokens[i:i+1] == ["</poly>"]:
                    i += 1
                points = [(coords[j], coords[j+1]) for j in range(0, len(coords)-1, 2)]
                if len(points) >= 3:
                    objects.append((class_name, points))
            continue
        i += 1
    return objects


COLORS = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),(255,100,255),(100,255,255),(200,150,50),(50,200,150)]

def render_boxes_on_image(img_arr, objects, title=""):
    """Draw bounding boxes on image, return base64."""
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray((img_arr * 255).astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img)
    for idx, (cls, box) in enumerate(objects):
        color = COLORS[idx % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1+2, max(0, y1-12)), cls, fill=color)
    if title:
        draw.text((2, 2), title, fill=(255,255,255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_polygons_on_image(img_arr, objects, title=""):
    """Draw polygons on image, return base64."""
    from PIL import ImageDraw
    img = Image.fromarray((img_arr * 255).astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    for idx, (cls, points) in enumerate(objects):
        color = COLORS[idx % len(COLORS)]
        if len(points) >= 3:
            flat = [(int(x), int(y)) for x, y in points]
            draw.polygon(flat, outline=color, fill=(*color, 50))
            draw.text((int(points[0][0]), int(points[0][1])-12), cls, fill=color)
    if title:
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), title, fill=(255,255,255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_samples(num_per_task=3, run_num=None):
    print("Loading model...")
    model, config = load_model()
    if model is None:
        print("No checkpoint found.")
        return {}

    tokenizer = load_tokenizer()
    val_examples = load_data("val")
    images = load_images()

    # Use same fixed sample IDs as MLX track
    if FIXED_SAMPLE_IDS_FILE.exists():
        with open(FIXED_SAMPLE_IDS_FILE) as f:
            fixed_ids = json.load(f)
    else:
        fixed_ids = {}
        for ex_i, ex in enumerate(val_examples):
            t = ex["task"]
            if t not in fixed_ids:
                fixed_ids[t] = []
            if ex["image_id"] in images and len(fixed_ids[t]) < num_per_task:
                fixed_ids[t].append({"index": ex_i, "image_id": ex["image_id"]})

    samples = {}
    for task, picks in fixed_ids.items():
        print(f"  {task}...")
        task_samples = []
        for pick in picks:
            idx, img_id = pick["index"], pick["image_id"]
            if img_id not in images:
                continue
            ex = val_examples[idx]
            image = images[img_id]
            gt_text = format_prediction(ex["target_ids"], tokenizer)
            try:
                img_tensor = prepare_image(image)
                pred_ids = greedy_decode(model, img_tensor, ex["input_ids"], tokenizer, config)
                pred_text = format_prediction(pred_ids, tokenizer)
            except Exception as e:
                pred_text = f"(error: {e})"
            inp_text = format_prediction(ex["input_ids"], tokenizer) if task == "vqa" else f"<{task}>"
            sample = {
                "image_id": img_id, "task": task, "input": inp_text,
                "ground_truth": gt_text, "prediction": pred_text,
                "image_b64": image_to_base64(image),
            }

            # Render visualizations for spatial tasks
            if task == "od":
                gt_boxes = parse_boxes(ex["target_ids"], tokenizer)
                pred_boxes = parse_boxes(pred_ids, tokenizer)
                if gt_boxes:
                    sample["gt_viz_b64"] = render_boxes_on_image(image, gt_boxes, "Ground Truth")
                if pred_boxes:
                    sample["pred_viz_b64"] = render_boxes_on_image(image, pred_boxes, "Prediction")
            elif task == "seg":
                gt_polys = parse_polygons(ex["target_ids"], tokenizer)
                pred_polys = parse_polygons(pred_ids, tokenizer)
                if gt_polys:
                    sample["gt_viz_b64"] = render_polygons_on_image(image, gt_polys, "Ground Truth")
                if pred_polys:
                    sample["pred_viz_b64"] = render_polygons_on_image(image, pred_polys, "Prediction")

            task_samples.append(sample)
            print(f"    img={img_id}: pred='{pred_text[:60]}...'")
        samples[task] = task_samples

    # Save
    data = {"run": run_num, "samples": samples}
    with open("samples_torch.json", "w") as f:
        json.dump(data, f)
    if run_num:
        with open(f"samples_torch_{run_num:03d}.json", "w") as f:
            json.dump(data, f)
    print(f"Saved {sum(len(v) for v in samples.values())} samples")
    return samples


if __name__ == "__main__":
    run = int(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_samples(run_num=run)
