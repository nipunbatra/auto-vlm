#!/usr/bin/env python3
"""
Experiment harness for Auto-VLM.
Runs training, parses results, generates plots, commits and pushes to GitHub.
"""

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_FILE = Path("results.tsv")
PLOT_FILE = Path("experiments_plot.png")
REMOTE_NAME = "origin"
BRANCH_NAME = "master"
TIME_BUDGET = 600  # 10 minutes per experiment


def run_training():
    """Run train_mlx.py and capture output."""
    print("\n" + "=" * 60)
    print("Running training...")
    print("=" * 60)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "train_mlx.py"],
            capture_output=True, text=True,
            timeout=TIME_BUDGET + 120,  # Extra buffer
        )
        elapsed = time.time() - start
        output = result.stdout + "\n" + result.stderr

        # Save log
        with open("run.log", "w") as f:
            f.write(output)

        print(output[-2000:])  # Print last 2000 chars

        return output, elapsed, result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return "TIMEOUT", elapsed, False
    except Exception as e:
        elapsed = time.time() - start
        return f"ERROR: {e}", elapsed, False


def parse_results(output):
    """Parse val_loss and params from training output."""
    val_loss = None
    params_m = None

    # Look for "Val loss: X.XXXX"
    match = re.search(r"Val loss:\s*([\d.]+)", output)
    if match:
        val_loss = float(match.group(1))

    # Look for "Parameters: X,XXX,XXX (Y.YYM)"
    match = re.search(r"Parameters:.*?\(([\d.]+)M\)", output)
    if match:
        params_m = float(match.group(1))

    # Also try "Params: X.XXM"
    if params_m is None:
        match = re.search(r"Params:\s*([\d.]+)M", output)
        if match:
            params_m = float(match.group(1))

    return val_loss, params_m


def load_results():
    """Load existing results from TSV."""
    results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                results.append(row)
    return results


def save_result(run_num, val_loss, params_m, status, time_s, improv_pct, description):
    """Append a result to the TSV file."""
    file_exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["run", "val_loss", "params_M", "status", "time_s", "improv_%", "description"])
        writer.writerow([
            f"{run_num:03d}",
            f"{val_loss:.4f}" if val_loss else "N/A",
            f"{params_m:.1f}" if params_m else "N/A",
            status,
            f"{time_s:.0f}",
            f"{improv_pct:.1f}" if improv_pct is not None else "N/A",
            description,
        ])


def generate_plot():
    """Generate experiment progress plot."""
    results = load_results()
    if not results:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")

    # ─── Top: Val loss over experiments ───
    ax = axes[0]
    ax.set_facecolor("#0d1117")

    runs = []
    val_losses_keep = []
    val_losses_discard = []
    best_so_far = []
    current_best = float("inf")

    for r in results:
        run_num = int(r["run"])
        try:
            vl = float(r["val_loss"])
        except (ValueError, KeyError):
            continue

        runs.append(run_num)
        if r["status"] == "keep":
            val_losses_keep.append((run_num, vl))
            if vl < current_best:
                current_best = vl
        else:
            val_losses_discard.append((run_num, vl))

        best_so_far.append((run_num, current_best))

    if val_losses_keep:
        x, y = zip(*val_losses_keep)
        ax.scatter(x, y, c="#2ea043", s=60, zorder=3, label="Keep", marker="o")
    if val_losses_discard:
        x, y = zip(*val_losses_discard)
        ax.scatter(x, y, c="#f85149", s=40, zorder=2, label="Discard", marker="x")
    if best_so_far:
        x, y = zip(*best_so_far)
        ax.plot(x, y, c="#58a6ff", linewidth=2, label="Best so far", zorder=1)

    ax.set_xlabel("Experiment", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Val Loss", color="#c9d1d9", fontsize=12)
    ax.set_title("Auto-VLM: Multi-Task Vision-Language Model Training",
                 color="#c9d1d9", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.1, color="#8b949e")

    # ─── Bottom: Improvement % ───
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")

    improvements = []
    for r in results:
        run_num = int(r["run"])
        try:
            imp = float(r["improv_%"])
        except (ValueError, KeyError):
            imp = 0
        improvements.append((run_num, imp, r["status"]))

    if improvements:
        x_vals = [i[0] for i in improvements]
        y_vals = [i[1] for i in improvements]
        colors = ["#2ea043" if i[2] == "keep" else "#f85149" for i in improvements]
        ax2.bar(x_vals, y_vals, color=colors, alpha=0.7)

    ax2.set_xlabel("Experiment", color="#c9d1d9", fontsize=12)
    ax2.set_ylabel("Improvement %", color="#c9d1d9", fontsize=12)
    ax2.tick_params(colors="#8b949e")
    ax2.spines["bottom"].set_color("#30363d")
    ax2.spines["left"].set_color("#30363d")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.1, color="#8b949e")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {PLOT_FILE}")


def update_index_html(results):
    """Update the GitHub Pages index.html with latest results."""
    if not results:
        return

    best_result = None
    baseline_loss = None
    for r in results:
        try:
            vl = float(r["val_loss"])
        except (ValueError, KeyError):
            continue
        if baseline_loss is None:
            baseline_loss = vl
        if r["status"] == "keep":
            if best_result is None or vl < float(best_result["val_loss"]):
                best_result = r

    total_exp = len(results)
    kept_exp = sum(1 for r in results if r["status"] == "keep")
    best_loss = float(best_result["val_loss"]) if best_result else 0
    improv = float(best_result["improv_%"]) if best_result and best_result["improv_%"] != "N/A" else 0

    # Read current index.html and update stats
    # The index.html has placeholder spans for dynamic content
    html_path = Path("index.html")
    if html_path.exists():
        html = html_path.read_text()
        html = re.sub(r'id="total-exp">[^<]*<', f'id="total-exp">{total_exp}<', html)
        html = re.sub(r'id="kept-exp">[^<]*<', f'id="kept-exp">{kept_exp}<', html)
        html = re.sub(r'id="best-loss">[^<]*<', f'id="best-loss">{best_loss:.4f}<', html)
        html = re.sub(r'id="improvement">[^<]*<', f'id="improvement">{improv:.1f}%<', html)
        html_path.write_text(html)


def git_commit_and_push(run_num, description, val_loss, improv_pct):
    """Commit results and push to GitHub."""
    msg = f"exp {run_num:03d}: {description}"
    if val_loss:
        msg += f" (val={val_loss:.4f}"
        if improv_pct is not None:
            msg += f", {improv_pct:.1f}%"
        msg += ")"

    files_to_add = [
        "results.tsv",
        "experiments_plot.png",
        "train_mlx.py",
        "index.html",
    ]
    if Path("checkpoint/config.json").exists():
        files_to_add.append("checkpoint/config.json")

    for f in files_to_add:
        if Path(f).exists():
            subprocess.run(["git", "add", f], check=False)

    subprocess.run(["git", "commit", "-m", msg], check=False)
    subprocess.run(["git", "push", REMOTE_NAME, BRANCH_NAME], check=False)


def get_next_run_num():
    """Get the next experiment number."""
    results = load_results()
    if not results:
        return 1
    return max(int(r["run"]) for r in results) + 1


def get_best_val_loss():
    """Get the best val_loss from kept experiments."""
    results = load_results()
    best = float("inf")
    for r in results:
        if r["status"] == "keep":
            try:
                vl = float(r["val_loss"])
                best = min(best, vl)
            except (ValueError, KeyError):
                pass
    return best


def run_single_experiment(description="baseline"):
    """Run a single experiment: train, evaluate, log, plot, push."""
    run_num = get_next_run_num()
    best_val = get_best_val_loss()
    baseline_loss = None

    results = load_results()
    if results:
        try:
            baseline_loss = float(results[0]["val_loss"])
        except (ValueError, KeyError):
            pass

    print(f"\n{'=' * 60}")
    print(f"Experiment {run_num:03d}: {description}")
    print(f"Best val_loss so far: {best_val:.4f}")
    print(f"{'=' * 60}")

    # Run training
    output, elapsed, success = run_training()
    val_loss, params_m = parse_results(output)

    if not success or val_loss is None:
        save_result(run_num, val_loss or 0, params_m or 0, "crash", elapsed, None, description)
        generate_plot()
        update_index_html(load_results())
        git_commit_and_push(run_num, description, val_loss, None)
        return False

    # Compute improvement
    if baseline_loss is None:
        baseline_loss = val_loss
    improv_pct = 100.0 * (baseline_loss - val_loss) / baseline_loss if baseline_loss > 0 else 0

    # Keep or discard
    if val_loss < best_val:
        status = "keep"
        print(f"\n✓ NEW BEST: {val_loss:.4f} (was {best_val:.4f}, {improv_pct:.1f}% improvement)")
    else:
        status = "discard"
        print(f"\n✗ No improvement: {val_loss:.4f} >= {best_val:.4f}")

    save_result(run_num, val_loss, params_m, status, elapsed, improv_pct, description)
    generate_plot()
    update_index_html(load_results())
    git_commit_and_push(run_num, description, val_loss, improv_pct)

    return status == "keep"


if __name__ == "__main__":
    # Run baseline experiment
    run_single_experiment("baseline: patch32 vdim192 vd3 ldim256 ld4 bs8 lr3e-4")
