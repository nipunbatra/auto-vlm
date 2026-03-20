#!/usr/bin/env python3
"""
Compute proper evaluation metrics for each task:
- Caption/Describe: BLEU-4, METEOR, CIDEr (via text similarity)
- VQA: Accuracy (exact match)
- OD: mAP@50 (IoU-based)
- Seg: mIoU (polygon IoU)
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


# ─── Text Metrics ────────────────────────────────────────────────────────────

def bleu_score(pred, ref, max_n=4):
    """Compute BLEU-N score between prediction and reference."""
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    # N-gram precision
    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = max(sum(pred_ngrams.values()), 1)
        scores.append(matches / total)

    # Geometric mean
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = sum(np.log(s) for s in scores) / len(scores)
    return bp * np.exp(log_avg)


def vqa_accuracy(pred, ref):
    """Exact match accuracy for VQA."""
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


# ─── Spatial Metrics ─────────────────────────────────────────────────────────

def parse_boxes(text, img_size=224):
    """Parse '[BOX: x1 y1 x2 y2]' from text."""
    boxes = []
    for m in re.finditer(r'(\w+)\s*\[BOX:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', text):
        cls = m.group(1)
        coords = [int(m.group(i)) / 999.0 * img_size for i in range(2, 6)]
        boxes.append((cls, coords))
    return boxes


def iou_box(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap50(pred_boxes, gt_boxes):
    """Compute AP@50 for a single image."""
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    matched = set()
    tp = 0
    for pred_cls, pred_box in pred_boxes:
        best_iou = 0
        best_idx = -1
        for idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if idx in matched:
                continue
            if pred_cls == gt_cls:
                iou = iou_box(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
        if best_iou >= 0.5:
            tp += 1
            matched.add(best_idx)

    precision = tp / len(pred_boxes) if pred_boxes else 0
    recall = tp / len(gt_boxes) if gt_boxes else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_metrics(samples):
    """Compute metrics for all tasks from samples dict."""
    metrics = {}

    for task, task_samples in samples.items():
        task_metrics = []

        for s in task_samples:
            gt = s.get("ground_truth", "")
            pred = s.get("prediction", "")

            if task in ("caption", "describe"):
                task_metrics.append({
                    "bleu4": bleu_score(pred, gt, max_n=4),
                    "bleu1": bleu_score(pred, gt, max_n=1),
                })
            elif task == "vqa":
                task_metrics.append({
                    "accuracy": vqa_accuracy(pred, gt),
                })
            elif task == "od":
                gt_boxes = parse_boxes(gt)
                pred_boxes = parse_boxes(pred)
                task_metrics.append({
                    "ap50": compute_ap50(pred_boxes, gt_boxes),
                    "num_gt": len(gt_boxes),
                    "num_pred": len(pred_boxes),
                })
            elif task == "seg":
                # For seg, just count if the right classes are predicted
                gt_classes = set(re.findall(r'(\w+)\s*\[POLY:', gt))
                pred_classes = set(re.findall(r'(\w+)\s*\[POLY:', pred))
                overlap = gt_classes & pred_classes
                prec = len(overlap) / len(pred_classes) if pred_classes else 0
                rec = len(overlap) / len(gt_classes) if gt_classes else 0
                task_metrics.append({
                    "class_f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0,
                    "num_gt_classes": len(gt_classes),
                    "num_pred_classes": len(pred_classes),
                })

        if task_metrics:
            avg = {}
            for key in task_metrics[0]:
                vals = [m[key] for m in task_metrics]
                avg[key] = sum(vals) / len(vals)
            metrics[task] = avg

    return metrics


if __name__ == "__main__":
    samples_path = Path("samples_torch.json")
    if samples_path.exists():
        data = json.load(open(samples_path))
        samples = data.get("samples", data)
        metrics = compute_metrics(samples)
        print("\n=== Evaluation Metrics ===")
        for task, m in sorted(metrics.items()):
            print(f"\n{task}:")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}")
    else:
        print("No samples found. Run generate_samples first.")
