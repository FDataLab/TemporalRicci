#!/usr/bin/env python3
"""
Visualize RNN experiment results and processing times.

1️⃣ Bar charts for each dataset (ROC_AUC, Accuracy, Loss, etc.)
2️⃣ Bar charts for processing time (in minutes) per dataset variant
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PATH CONFIGURATION
# ============================================================

RESULTS_FILE = r"C:\Users\azadp\PycharmProjects\TemporalRicci\GraphPulseResults\data\output\RNNResultsAllTasks.csv"
TIME_FILE = r"C:\Users\azadp\PycharmProjects\TemporalRicci\GraphPulseResults\process_data_time.csv"
OUTPUT_DIR = r"C:\Users\azadp\PycharmProjects\TemporalRicci\images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# COMMON HELPERS
# ============================================================

def clean_label(name: str) -> str:
    """
    Convert 'BEPRO_TFR_a3.00_b1.00' -> 'BEPRO (original)'
    and 'BEPRO_TFR_a3.00_b1.00_bin3' -> 'BEPRO_bin3'
    """
    base_match = re.match(r"^(.*)_TFR_a[\d.]+_b[\d.]+(_bin\d+)?$", name)
    if not base_match:
        return name
    dataset = base_match.group(1)
    bin_part = base_match.group(2)
    if bin_part:
        return f"{dataset}{bin_part}"
    else:
        return f"{dataset} (original)"


# ============================================================
# VISUALIZE RNN METRICS
# ============================================================

def visualize_rnn_results(metric: str = "ROC_AUC"):
    print(f"\n=== Generating {metric} bar graphs ===")

    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} rows from {RESULTS_FILE}")

    # Extract dataset root and bin info
    df["dataset_root"] = df["Network"].apply(lambda x: x.split("_TFR")[0])
    df["bin_num"] = df["Network"].apply(lambda x: int(x.split("bin")[-1]) if "bin" in x else 0)
    df["clean_name"] = df["Network"].apply(clean_label)
    df = df.sort_values(["dataset_root", "bin_num", "Task"])

    # Task display names
    TASK_NAMES = {
        "task1": "Network Growth Prediction",
        "task2": "Influential Node Prediction",
        "task3": "Connected Component Prediction"
    }

    datasets = df["dataset_root"].unique()

    for dataset in datasets:
        subset = df[df["dataset_root"] == dataset]
        pivot = subset.pivot(index="clean_name", columns="Task", values=metric)
        pivot = pivot.reindex(columns=["task1", "task2", "task3"], fill_value=0)

        # Plot
        x = np.arange(len(pivot))
        width = 0.25
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, task in enumerate(pivot.columns):
            ax.bar(
                x + i * width,
                pivot[task],
                width,
                label=TASK_NAMES.get(task, task),
                color=colors[i],
            )

        ax.set_title(f"{dataset} — {metric} by Bin and Task", fontsize=14, pad=15)
        ax.set_xlabel("Dataset Variant", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{dataset}_{metric}_bars.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {out_path}")

    print(f"✅ Finished creating {metric} graphs.\n")


# ============================================================
# VISUALIZE PROCESSING TIME
# ============================================================

def visualize_processing_times():
    print(f"\n=== Generating processing time bar graphs ===")

    df = pd.read_csv(TIME_FILE)
    print(f"Loaded {len(df)} rows from {TIME_FILE}")

    # Convert to minutes
    df["duration_min"] = df["total_duration_sec"] / 60.0

    # Clean variant names for readability
    df["clean_name"] = df["variant"].apply(clean_label)
    df["bin_num"] = df["variant"].apply(lambda x: int(x.split("bin")[-1]) if "bin" in x else 0)
    df = df.sort_values(["dataset", "bin_num"])

    datasets = df["dataset"].unique()

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        x = np.arange(len(subset))
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x, subset["duration_min"], color="#1f77b4", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["clean_name"], rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{dataset} — Sequence Preparation Time", fontsize=14, pad=15)
        ax.set_xlabel("Dataset Variant", fontsize=12)
        ax.set_ylabel("Time (minutes)", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{dataset}_processing_time.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_path}")

    print("✅ Finished creating processing time graphs.\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    visualize_rnn_results(metric="ROC_AUC")     # main performance metric
    visualize_processing_times()                 # runtime visualization
