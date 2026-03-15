"""
Auto-VLM: Streamlit Dashboard
Live monitoring of autonomous VLM research experiments.

Deploy: streamlit run app.py
Or deploy to Streamlit Community Cloud from GitHub.
"""

import json
import csv
from pathlib import Path

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Auto-VLM — Autonomous Multi-Task VLM Research",
    page_icon="🔬",
    layout="wide",
)

# ─── Config ──────────────────────────────────────────────────────────────────

RESULTS_FILE = Path("results.tsv")
PLOT_FILE = Path("experiments_plot.png")
CHECKPOINT_CONFIG = Path("checkpoint/config.json")

TASK_DESCRIPTIONS = {
    "caption": ("📝 Captioning", "Generate a natural language description of the image"),
    "vqa": ("❓ VQA", "Answer questions about the image"),
    "od": ("📦 Object Detection", "Detect objects with bounding boxes: class <box>x1 y1 x2 y2</box>"),
    "seg": ("🎭 Segmentation", "Segment objects with polygons: class <poly>vertices</poly>"),
    "describe": ("💬 Free Text", "Generate a detailed description of the scene"),
}


# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_results():
    """Load experiment results from TSV."""
    results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                results.append(row)
    return results


@st.cache_data(ttl=30)
def load_config():
    """Load model config from checkpoint."""
    if CHECKPOINT_CONFIG.exists():
        with open(CHECKPOINT_CONFIG) as f:
            return json.load(f)
    return None


# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🔬 Auto-VLM")
st.markdown(
    "Autonomous research: training a tiny (<10M param) multi-task "
    "Vision-Language Model from scratch in **MLX** on Apple Silicon"
)

# Task badges
cols = st.columns(5)
for i, (task, (emoji_name, desc)) in enumerate(TASK_DESCRIPTIONS.items()):
    with cols[i]:
        st.markdown(f"**{emoji_name}**")
        st.caption(desc)

st.divider()

# ─── Results ─────────────────────────────────────────────────────────────────

results = load_results()

if not results:
    st.info("No experiments yet. Run `python run_experiments.py` to start.")
    st.stop()

# Parse numeric values
for r in results:
    try:
        r["val_loss_f"] = float(r["val_loss"])
    except (ValueError, KeyError):
        r["val_loss_f"] = None
    try:
        r["params_f"] = float(r["params_M"])
    except (ValueError, KeyError):
        r["params_f"] = None
    try:
        r["improv_f"] = float(r["improv_%"])
    except (ValueError, KeyError):
        r["improv_f"] = None

# ─── Stats ───────────────────────────────────────────────────────────────────

kept = [r for r in results if r["status"] == "keep"]
best = min(kept, key=lambda r: r["val_loss_f"]) if kept else None
baseline = results[0] if results else None

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Experiments", len(results))
with col2:
    st.metric("Kept", len(kept))
with col3:
    discarded = sum(1 for r in results if r["status"] == "discard")
    st.metric("Discarded", discarded)
with col4:
    if best:
        st.metric("Best Val Loss", f"{best['val_loss_f']:.4f}")
with col5:
    if best and best["improv_f"] is not None:
        st.metric("Improvement", f"{best['improv_f']:.1f}%")

st.divider()

# ─── Charts ──────────────────────────────────────────────────────────────────

tab_chart, tab_table, tab_arch, tab_about = st.tabs(
    ["📈 Progress", "📋 Results Table", "🏗️ Architecture", "ℹ️ About"]
)

with tab_chart:
    # Show the matplotlib plot if available
    if PLOT_FILE.exists():
        st.image(str(PLOT_FILE), use_container_width=True)
    else:
        # Build a simple chart with streamlit
        import pandas as pd

        chart_data = []
        best_so_far = float("inf")
        for r in results:
            vl = r["val_loss_f"]
            if vl is None:
                continue
            run = int(r["run"])
            if r["status"] == "keep" and vl < best_so_far:
                best_so_far = vl
            chart_data.append({
                "Experiment": run,
                "Val Loss": vl,
                "Best So Far": best_so_far,
                "Status": r["status"],
            })

        if chart_data:
            df = pd.DataFrame(chart_data)
            st.line_chart(df, x="Experiment", y=["Val Loss", "Best So Far"])

    # Per-task loss breakdown (if available from config)
    config = load_config()
    if config and "val_loss" in config:
        st.subheader("Latest Model Stats")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Val Loss", f"{config['val_loss']:.4f}")
        with c2:
            st.metric("Parameters", f"{config.get('num_params', 0) / 1e6:.2f}M")
        with c3:
            st.metric("Training Steps", config.get("train_steps", "N/A"))

with tab_table:
    import pandas as pd

    df = pd.DataFrame(results)
    display_cols = ["run", "val_loss", "params_M", "status", "time_s", "improv_%", "description"]
    df_display = df[[c for c in display_cols if c in df.columns]]

    # Color the status column
    def color_status(val):
        if val == "keep":
            return "background-color: #2ea04320"
        elif val == "discard":
            return "background-color: #f8514920"
        elif val == "crash":
            return "background-color: #d2992220"
        return ""

    styled = df_display.iloc[::-1].style.map(color_status, subset=["status"])
    st.dataframe(styled, use_container_width=True, height=500)

with tab_arch:
    st.subheader("Model Architecture")

    config = load_config()
    if config:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Vision Encoder (Tiny ViT)")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Image Size | {config.get('image_size', 224)} |
            | Patch Size | {config.get('patch_size', 32)} |
            | Num Patches | {config.get('num_patches', '?')} |
            | Dimension | {config.get('vision_dim', 192)} |
            | Depth | {config.get('vision_depth', 3)} |
            | Heads | {config.get('vision_heads', 3)} |
            """)

        with col2:
            st.markdown("### Language Decoder (Small GPT)")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Vocab Size | {config.get('vocab_size', '?')} |
            | Dimension | {config.get('lang_dim', 256)} |
            | Depth | {config.get('lang_depth', 4)} |
            | Heads | {config.get('lang_heads', 4)} |
            | Max Seq Len | {config.get('max_seq_len', 320)} |
            """)

        st.markdown("### Training Config")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Batch Size | {config.get('batch_size', 8)} |
        | Learning Rate | {config.get('learning_rate', 3e-4)} |
        | Weight Decay | {config.get('weight_decay', 0.1)} |
        | Dropout | {config.get('dropout', 0.1)} |
        | Time Budget | {config.get('time_budget', 600)}s |
        | Total Params | {config.get('num_params', 0):,} |
        """)
    else:
        st.info("No checkpoint found yet. Run training first.")

    st.subheader("Task Output Formats")
    st.markdown("""
    All tasks use a unified autoregressive token format (Florence-2 style):

    ```
    Caption:  <caption> a dog sitting on green grass <eos>
    VQA:      <vqa> how many dogs? <sep> 2 <eos>
    OD:       <od> dog <box><loc120><loc450><loc670><loc890></box> <sep> cat <box>...</box> <eos>
    Seg:      <seg> dog <poly><loc120><loc450><loc130><loc460>...</poly> <eos>
    Describe: <describe> a detailed scene showing a dog sitting... <eos>
    ```

    **Coordinates**: 1000 location bins (`<loc000>` to `<loc999>`), each representing
    0.1% of the image dimension. A `<loc500>` token = 50% of image width/height.

    **Loss**: Pure cross-entropy on all tokens — same as Florence-2, Pix2Seq, and Kosmos-2.
    The model learns spatial relationships implicitly through token embeddings.
    """)

with tab_about:
    st.markdown("""
    ### About Auto-VLM

    This project trains a tiny (<10M parameter) multi-task Vision-Language Model
    entirely from scratch using [MLX](https://ml-explore.github.io/mlx/) on Apple Silicon.

    **The autonomous research loop**:
    1. An AI agent modifies the training script (`train_mlx.py`)
    2. Trains the model for 10 minutes
    3. If val_loss improves → keep the change
    4. If not → discard and try something else
    5. Repeat indefinitely

    **Supported tasks**:
    - **Captioning**: Generate image descriptions
    - **VQA**: Answer questions about images
    - **Object Detection**: Locate objects with bounding boxes
    - **Segmentation**: Outline objects with polygons
    - **Free Text**: Generate detailed scene descriptions

    **Data**: COCO val2017 (5000 images, ~51K training examples)

    **Inspired by**: [autoresearch](https://github.com/nipunbatra/autoresearch)

    ---

    [GitHub Repository](https://github.com/nipunbatra/auto-vlm) ·
    [GitHub Pages Dashboard](https://nipunbatra.github.io/auto-vlm/)
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Auto-VLM — Trained on Apple M3 64GB with MLX · "
    "[GitHub](https://github.com/nipunbatra/auto-vlm) · "
    "Powered by autonomous AI research"
)
