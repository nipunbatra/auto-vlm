# Auto-VLM: Autonomous Multi-Task Vision-Language Model Research

## Project Overview
Train a tiny (<10M param) multi-task VLM from scratch using MLX on Apple Silicon.
The model handles: VQA, Object Detection, Segmentation, Captioning, and Free-text Description.

## Key Files
- `prepare_data.py` — **FIXED, DO NOT EDIT.** Downloads COCO data, builds tokenizer, creates task examples.
- `train_mlx.py` — **EDITABLE.** Model architecture, training loop, losses. This is what experiments modify.
- `run_experiments.py` — Experiment harness. Runs train, parses results, keeps/discards, plots, pushes.
- `program.md` — Instructions for the autonomous research agent.
- `index.html` — GitHub Pages dashboard showing experiment progress.
- `results.tsv` — Tab-separated experiment log.

## Architecture
- **Vision Encoder**: Tiny ViT (patch_size=32, 224x224 input → 49 visual tokens)
- **Projector**: Linear layer mapping vision features to language dimension
- **Language Decoder**: Small GPT-style causal transformer
- **Task tokens**: `<caption>`, `<vqa>`, `<od>`, `<seg>`, `<describe>`
- **Coordinate tokens**: `<loc000>`–`<loc999>` for bounding boxes and polygon vertices

## Multi-Task Loss Strategy
- All tasks: Cross-entropy on predicted tokens (autoregressive)
- OD: CE + optional GIoU loss on parsed bounding boxes
- Segmentation: CE + optional Dice loss on rasterized polygon masks
- VQA/Caption/Describe: CE only

## Data
- COCO val2017 (5000 images): captions, object detection, instance segmentation
- Generated VQA pairs from COCO annotations
- Split: 4000 train / 1000 val

## Running
```bash
pip install -e .
python prepare_data.py   # Download and preprocess data (~5 min)
python train_mlx.py      # Train the model (~10 min)
python run_experiments.py # Start autonomous experiment loop
```

## Constraints
- Model must be <10M parameters
- Each experiment has a 10-minute wall-clock budget
- Only `train_mlx.py` may be modified by the agent
- Target hardware: Apple M3 with 64GB RAM
