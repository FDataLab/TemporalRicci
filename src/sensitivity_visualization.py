#!/usr/bin/env python3
"""
Simplified Alpha-Beta Sensitivity Analysis for GraphPulse.

This script produces:

1) For each dataset and each task:
   - Best-bin AUC surface (α, β -> max AUC over bins 1–10)
     * Interactive Plotly HTML
     * Static PNG 3D surface

2) For each dataset:
   - Combined PNG with 3 surfaces (Task 1, 2, 3 best-bin AUC side by side)

3) Global across all tasks (within one dataset):
   - Mean best-bin AUC across tasks per (α, β)
   - Interactive 3D surface (HTML)
   - Static 3D surface (PNG)
   - Heatmap (PNG) with best (α, β) highlighted

Input:
  ../GraphPulseResults/data/output/RNNResultsAllTasks_{DATASET}_Sensitivity.csv

Expected columns:
  - Task          (e.g., "task1", "task2", "task3")
  - Network       (string containing aX.X_bY.Y and optional _binK)
  - ROC_AUC       (float)
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# =====================================================
# CONFIG
# =====================================================

DATASET_CODES = ["MIR"]  # extend if needed
INPUT_TEMPLATE = "../GraphPulseResults/data/output/RNNResultsAllTasks_{code}_Sensitivity.csv"

IMAGES_ROOT = Path("../GraphPulseResults/sensitivity_bestbin")
IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


# =====================================================
# HELPERS
# =====================================================

def extract_alpha_beta(name: str):
    """Extract alpha,beta from network name like MIR_TFR_a3.00_b5.00_bin2."""
    m = re.search(r"a(\d+\.?\d*)_b(\d+\.?\d*)", name)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def extract_bin_number(name: str):
    """Extract bin number K from suffix _binK; original graphs return 0."""
    m = re.search(r"_bin(\d+)$", name)
    return int(m.group(1)) if m else 0


def build_auc_grid(df: pd.DataFrame, value_col="ROC_AUC"):
    """
    Create an (alphas, betas, Z) grid for surface plotting.

    df must have columns: alpha, beta, value_col
    """
    if df.empty:
        return [], [], np.array([[]])

    grouped = (
        df.groupby(["alpha", "beta"])[value_col]
        .mean()
        .reset_index()
    )

    alphas = sorted(grouped["alpha"].unique())
    betas = sorted(grouped["beta"].unique())
    A, B = np.meshgrid(alphas, betas)
    Z = np.full_like(A, np.nan, dtype=float)

    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            row = grouped[(grouped["alpha"] == a) & (grouped["beta"] == b)]
            if len(row):
                Z[i, j] = row[value_col].values[0]

    return alphas, betas, Z


def find_best_alpha_beta(alphas, betas, Z):
    """Return best (alpha, beta, value) from grid (skip if empty or all NaN)."""
    if Z.size == 0 or len(alphas) == 0 or len(betas) == 0:
        return None, None, None

    if np.all(np.isnan(Z)):
        return None, None, None

    max_idx = np.nanargmax(Z)
    i_best, j_best = np.unravel_index(max_idx, Z.shape)
    best_alpha = alphas[j_best]
    best_beta = betas[i_best]
    best_val = Z[i_best, j_best]
    return best_alpha, best_beta, best_val


def plot_surface_plotly(alphas, betas, Z, title, html_path: Path, zlabel="AUC"):
    """Interactive 3D surface with automatic best (alpha,beta) highlight."""
    if not HAS_PLOTLY:
        print(f"[WARN] Plotly missing → cannot render: {html_path}")
        return

    if Z.size == 0 or len(alphas) == 0 or len(betas) == 0:
        print(f"[WARN] Empty grid → skipping Plotly surface: {html_path}")
        return

    A, B = np.meshgrid(alphas, betas)

    best_alpha, best_beta, best_val = find_best_alpha_beta(alphas, betas, Z)
    if best_alpha is None:
        print(f"[WARN] All-NaN grid → skipping Plotly surface: {html_path}")
        return

    print(f"  → Best (Plotly) α={best_alpha}, β={best_beta}, {zlabel}={best_val:.4f}")

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=A,
            y=B,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title=zlabel),
            opacity=0.95,
            hovertemplate="α=%{x}<br>β=%{y}<br>" + zlabel + "=%{z}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[best_alpha],
            y=[best_beta],
            z=[best_val],
            mode="markers+text",
            marker=dict(size=7, color="red"),
            text=[f"BEST ({best_val:.3f})"],
            textposition="top center",
            name="Best point",
        )
    )

    fig.update_layout(
        title=f"{title}<br><sup>Best α={best_alpha}, β={best_beta}, {zlabel}={best_val:.4f}</sup>",
        scene=dict(
            xaxis_title="Alpha (α)",
            yaxis_title="Beta (β)",
            zaxis_title=zlabel,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"[3D HTML SAVED] {html_path}")


def plot_surface_png(alphas, betas, Z, title, png_path: Path, zlabel="AUC"):
    """Static Matplotlib 3D surface with best (alpha,beta) marked."""
    if Z.size == 0 or len(alphas) == 0 or len(betas) == 0:
        print(f"[WARN] Empty grid → skipping PNG surface: {png_path}")
        return

    A, B = np.meshgrid(alphas, betas)
    best_alpha, best_beta, best_val = find_best_alpha_beta(alphas, betas, Z)
    if best_alpha is None:
        print(f"[WARN] All-NaN grid → skipping PNG surface: {png_path}")
        return

    print(f"  → Best (Matplotlib) α={best_alpha}, β={best_beta}, {zlabel}={best_val:.4f}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.scatter(best_alpha, best_beta, best_val, color="red", s=40, label=f"Best ({best_val:.3f})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Beta (β)")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=25, azim=35)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[3D PNG SAVED] {png_path}")


def plot_multi_task_surfaces_png(dataset_code, summary, out_dir: Path):
    """
    Make one PNG with 3 subplots (Task 1–3) best-bin AUC surfaces side by side.
    """
    tasks = sorted(summary["Task"].unique())
    if len(tasks) == 0:
        print("[WARN] No tasks in summary for multi-surface figure.")
        return

    fig = plt.figure(figsize=(18, 5))

    for idx, task in enumerate(tasks, start=1):
        sub = summary[summary["Task"] == task].copy()
        if sub.empty:
            continue

        alphas, betas, Z = build_auc_grid(
            sub.rename(columns={"best_bin_auc": "value"}), value_col="value"
        )
        if Z.size == 0 or len(alphas) == 0 or len(betas) == 0:
            continue

        A, B = np.meshgrid(alphas, betas)
        best_alpha, best_beta, best_val = find_best_alpha_beta(alphas, betas, Z)
        if best_alpha is None:
            continue

        ax = fig.add_subplot(1, 3, idx, projection="3d")

        surf = ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
        ax.scatter(best_alpha, best_beta, best_val, color="red", s=40)

        ax.set_title(f"Task {task} – Best-bin AUC", fontsize=11)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_zlabel("AUC")
        ax.view_init(elev=25, azim=35)

    fig.suptitle(f"{dataset_code}: Best-bin AUC Surfaces for All Tasks", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / f"{dataset_code}_all_tasks_bestbin_surfaces.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[COMBINED 3-SURFACE PNG SAVED] {out_path}")


# =====================================================
# PER-DATASET ANALYSIS
# =====================================================

def analyze_dataset(dataset_code: str):
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_code}")
    print("=" * 80)

    csv_path = INPUT_TEMPLATE.format(code=dataset_code)
    if not os.path.exists(csv_path):
        print(f"[ERROR] Missing CSV: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Extract α,β,bin
    df["alpha"], df["beta"] = zip(*df["Network"].apply(extract_alpha_beta))
    df = df.dropna(subset=["alpha", "beta"]).copy()
    df["alpha"] = df["alpha"].astype(float)
    df["beta"] = df["beta"].astype(float)
    df["bin_num"] = df["Network"].apply(extract_bin_number)

    tasks = sorted(df["Task"].unique())
    print("Tasks found:", tasks)

    out_dir = IMAGES_ROOT / dataset_code
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Build (alpha, beta) summary for best bin per task
    # --------------------------------------------------------
    print("Computing best-bin AUC per (α, β, Task)...")

    records = []
    grouped = df.groupby(["Task", "alpha", "beta"])
    for (task, a, b), sub in grouped:
        bin_rows = sub[sub["bin_num"] > 0]

        if bin_rows.empty:
            continue

        idx = bin_rows["ROC_AUC"].idxmax()
        best_auc = float(bin_rows.loc[idx, "ROC_AUC"])
        best_bin = int(bin_rows.loc[idx, "bin_num"])

        records.append({
            "Task": task,
            "alpha": a,
            "beta": b,
            "best_bin_auc": best_auc,
            "best_bin_num": best_bin,
            "dataset": dataset_code
        })

    summary = pd.DataFrame(records)
    if summary.empty:
        print("[WARN] No best-bin summary constructed (no bins?).")
        return summary

    summary_path = out_dir / f"{dataset_code}_bestbin_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved per-task best-bin summary: {summary_path}")

    # --------------------------------------------------------
    # Per-task surfaces (interactive + PNG)
    # --------------------------------------------------------
    print("\nGenerating per-task best-bin AUC surfaces...")

    for task in tasks:
        sub = summary[summary["Task"] == task].copy()
        if sub.empty:
            print(f"[Task {task}] No entries → skipping.")
            continue

        alphas, betas, Z = build_auc_grid(
            sub.rename(columns={"best_bin_auc": "value"}),
            value_col="value"
        )

        # Interactive Plotly
        html_path = out_dir / f"surface_bestbin_task{task}.html"
        plot_surface_plotly(
            alphas, betas, Z,
            title=f"{dataset_code} – Task {task}: Best-bin AUC Surface",
            html_path=html_path,
            zlabel="Best-bin AUC"
        )

        # Static PNG
        png_path = out_dir / f"surface_bestbin_task{task}.png"
        plot_surface_png(
            alphas, betas, Z,
            title=f"{dataset_code} – Task {task}: Best-bin AUC Surface",
            png_path=png_path,
            zlabel="Best-bin AUC"
        )

    # --------------------------------------------------------
    # Combined PNG with 3 surfaces side by side
    # --------------------------------------------------------
    print("\nGenerating combined 3-surface PNG (all tasks)...")
    plot_multi_task_surfaces_png(dataset_code, summary, out_dir)

    # --------------------------------------------------------
    # Global α–β performance across all tasks
    # --------------------------------------------------------
    print("\nComputing GLOBAL α–β average performance across all tasks...")

    global_ab = (
        summary.groupby(["alpha", "beta"])["best_bin_auc"]
        .mean()
        .reset_index(name="mean_auc")
    )

    global_ab_path = out_dir / f"{dataset_code}_global_alpha_beta_mean_auc.csv"
    global_ab.to_csv(global_ab_path, index=False)
    print(f"[GLOBAL TABLE SAVED] {global_ab_path}")

    # Build grid for global mean AUC
    alphas_g, betas_g, Z_g = build_auc_grid(
        global_ab.rename(columns={"mean_auc": "value"}),
        value_col="value"
    )

    best_alpha, best_beta, best_val = find_best_alpha_beta(alphas_g, betas_g, Z_g)
    if best_alpha is not None:
        print(f"\n[GLOBAL BEST α–β ACROSS TASKS]")
        print(f"   α={best_alpha}, β={best_beta}, mean AUC={best_val:.4f}")

    # Interactive 3D surface
    html_global = out_dir / "surface_global_alpha_beta.html"
    plot_surface_plotly(
        alphas_g, betas_g, Z_g,
        title=f"{dataset_code}: GLOBAL α–β Performance (Mean Best-bin AUC Across Tasks)",
        html_path=html_global,
        zlabel="Mean Best-bin AUC"
    )

    # Static 3D surface PNG
    png_global_surface = out_dir / "surface_global_alpha_beta.png"
    plot_surface_png(
        alphas_g, betas_g, Z_g,
        title=f"{dataset_code}: GLOBAL α–β Performance (Mean Across Tasks)",
        png_path=png_global_surface,
        zlabel="Mean Best-bin AUC"
    )

    # Heatmap PNG
    if Z_g.size > 0 and len(alphas_g) > 0 and len(betas_g) > 0 and best_alpha is not None:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            Z_g,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[min(alphas_g), max(alphas_g), min(betas_g), max(betas_g)],
        )
        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("Beta (β)")
        ax.set_title(f"{dataset_code}: GLOBAL Mean Best-bin AUC Across Tasks")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Best-bin AUC", fontsize=11)

        # Mark global best
        ax.scatter([best_alpha], [best_beta], color="red", s=80)
        ax.text(best_alpha, best_beta, f"{best_val:.3f}",
                color="red", fontsize=9, ha="left", va="bottom")

        heatmap_path = out_dir / "heatmap_global_alpha_beta.png"
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[GLOBAL HEATMAP SAVED] {heatmap_path}")

    return summary


# =====================================================
# GLOBAL DRIVER
# =====================================================

if __name__ == "__main__":
    all_summaries = []

    for code in DATASET_CODES:
        s = analyze_dataset(code)
        if s is not None and not s.empty:
            all_summaries.append(s)

    if all_summaries:
        global_summary = pd.concat(all_summaries, ignore_index=True)
        global_path = IMAGES_ROOT / "GLOBAL_bestbin_summary.csv"
        global_summary.to_csv(global_path, index=False)
        print(f"\nSaved GLOBAL summary across datasets: {global_path}")
    else:
        print("\nNo summaries produced. Check input CSV paths and contents.")

    print("\nDone.")
