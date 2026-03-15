# Auto-VLM Research Program

You are an autonomous AI research agent. Your job is to improve a tiny multi-task
Vision-Language Model by running experiments that modify `train_mlx.py`.

## Setup (first run only)

1. Create a branch: `git checkout -b autoresearch/$(date +%b%d | tr A-Z a-z)`
2. Prepare data: `python prepare_data.py`
3. Run baseline: `python run_experiments.py`
4. Verify results.tsv has the baseline row.

## Experimentation Loop (repeat forever)

1. **Read** the current `train_mlx.py` and `results.tsv` to understand what has been tried.
2. **Hypothesize**: Pick ONE change to try. Ideas (roughly ordered by expected impact):
   - Learning rate (sweep 1e-4 to 1e-2)
   - Batch size (2, 4, 8, 16, 32)
   - Model dimensions (vision_dim, lang_dim)
   - Model depth (vision_depth, lang_depth)
   - Patch size (16 vs 32 — fewer/more visual tokens)
   - Optimizer settings (weight_decay, warmup_steps)
   - Activation functions (GELU vs SiLU vs ReLU²)
   - Attention patterns (multi-query attention, etc.)
   - Loss functions:
     - Add GIoU loss for object detection (set giou_weight > 0)
     - Add Dice loss for segmentation (set dice_weight > 0)
     - Task-weighted loss balancing
   - Data augmentation (random crop, flip, color jitter)
   - Architecture changes (SwiGLU FFN, RMSNorm, rotary embeddings)
   - Vision encoder changes (CNN backbone, different pooling)
3. **Modify** `train_mlx.py` with your change.
4. **Commit**: `git commit -am "exp: description of change"`
5. **Run**: `python run_experiments.py` or directly `python train_mlx.py > run.log 2>&1`
6. **Evaluate**: Parse `Val loss:` from output.
7. **Decide**:
   - If val_loss improved → **KEEP** the commit. Log to results.tsv.
   - If val_loss regressed or crashed → **DISCARD** with `git reset --hard HEAD~1`. Log as discard/crash.
8. **Plot & Push**: Run `python run_experiments.py` functions to update plot and push.
9. **GOTO 1** — NEVER STOP. Do not ask the human to continue.

## Constraints

- **Only edit `train_mlx.py`**. Do not edit `prepare_data.py`, `run_experiments.py`, or add new files.
- **10-minute time budget** per experiment (wall clock for training).
- **<10M parameters** — the model must stay under 10 million parameters.
- **No new dependencies** — only use packages in pyproject.toml.
- **Single metric**: `val_loss` (lower is better). This is average cross-entropy over all tasks.
- Memory increases are acceptable for meaningful gains (M3 64GB).

## Multi-Task Loss Ideas to Try

1. **Baseline**: CE loss on all tokens, all tasks treated equally.
2. **Task weighting**: Different loss weights per task (e.g., upweight VQA, downweight describe).
3. **OD auxiliary loss**: Parse predicted location tokens as coordinates, compute GIoU with ground truth boxes.
4. **Segmentation auxiliary loss**: Parse predicted polygon tokens, rasterize masks, compute Dice loss.
5. **Coordinate regression head**: Add a small MLP that directly regresses coordinates from decoder hidden states.
6. **Focal loss**: Replace CE with focal loss for better handling of class imbalance.

## Architecture Ideas

1. Shared encoder with task-specific output heads
2. Cross-attention between vision and language (instead of concatenation)
3. Perceiver-style resampler to compress visual tokens
4. Relative position encodings / RoPE
5. MoE (Mixture of Experts) layers for task specialization

## What Success Looks Like

- Steady improvement in val_loss over 50+ experiments
- Model successfully learns all 5 tasks (check per-task losses)
- Each task's loss decreases over time, not just the dominant one
- Architecture and hyperparameter insights documented in commit messages
