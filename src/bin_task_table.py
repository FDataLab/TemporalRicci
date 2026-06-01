import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parent.parent
SENS_DIR = ROOT / "GraphPulseResultsWindowed" / "data" / "output"

METRIC_COL = "ROC_AUC"
TASK_COL = "Task"
NETWORK_COL = "Network"

TASKS = ["task1", "task2", "task3"]
METHODS = ["Full", "Bin1", "Bin2", "Bin3", "Bin4", "Bin5"]
BINS = ["Bin1", "Bin2", "Bin3", "Bin4", "Bin5"]

OUT_COUNTS = SENS_DIR / "bestbin_counts_all_datasets.csv"
OUT_DETAILS = SENS_DIR / "bestbin_winners_all_datasets.csv"
OUT_VALUES_TABLE = SENS_DIR / "bestbin_values_vs_full_all_datasets.csv"

FIG_DIR = SENS_DIR / "bestbin_visualizations"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRESENTATION_PLOT = FIG_DIR / "winning_bin_distribution_by_task.png"
OUT_TABLE_FIG = FIG_DIR / "auc_table_vs_full.png"



def parse_network(net: str):
    s = str(net).strip()
    s = s.replace("\\", "/").split("/")[-1]

    m = re.match(r"^(?P<dataset>.+)_(?P<suffix>full|bin[1-5])$", s, re.IGNORECASE)

    if not m:
        return None

    dataset = m.group("dataset").upper()
    suffix = m.group("suffix").lower()

    if suffix == "full":
        return {
            "dataset": dataset,
            "bin": np.nan,
            "Method": "Full"
        }

    bin_num = int(suffix.replace("bin", ""))

    return {
        "dataset": dataset,
        "bin": bin_num,
        "Method": f"Bin{bin_num}"
    }


# =========================
# Plot 1: Winning-bin distribution
# =========================
def save_presentation_winning_bin_plot(counts_df, output_path):
    plot_df = counts_df.drop(index="Total", errors="ignore").copy()
    plot_df = plot_df.drop(columns="Total", errors="ignore")

    pct_df = plot_df.div(plot_df.sum(axis=1), axis=0) * 100
    pct_df = pct_df.fillna(0)

    fig, ax = plt.subplots(figsize=(11, 4.8))

    left = np.zeros(len(pct_df))

    colors = {
        "Bin1": "#d73027",
        "Bin2": "#fc8d59",
        "Bin3": "#fee08b",
        "Bin4": "#91cf60",
        "Bin5": "#1a9850",
    }

    for bin_name in BINS:
        values = pct_df[bin_name].values

        ax.barh(
            pct_df.index,
            values,
            left=left,
            label=bin_name,
            color=colors[bin_name],
            edgecolor="white",
            linewidth=1.2
        )

        for i, v in enumerate(values):
            if v >= 8:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black"
                )

        left += values

    ax.set_xlim(0, 100)
    ax.set_xlabel(
        "Share of dataset-task cases where each bin gives the highest ROC-AUC (%)",
        fontsize=11
    )
    ax.set_ylabel("")

    ax.set_title(
        "High-Curvature Edges Dominate the Best Predictive Subgraphs",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    ax.legend(
        title="Winning curvature bin",
        ncol=5,
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        frameon=False
    )

    ax.grid(axis="x", linestyle="--", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Presentation plot -> {output_path}")


# =========================
# Plot 2: table vs full graph
# =========================
def save_auc_table_figure(values_table, output_path):
    table_df = values_table.copy()

    # Sort by dataset and task order
    table_df["task_order"] = table_df["task"].apply(lambda x: TASKS.index(x) if x in TASKS else 99)
    table_df = table_df.sort_values(["dataset", "task_order"])
    table_df = table_df.drop(columns=["task_order"])

    display_df = table_df.copy()
    display_df.insert(0, "Dataset / Task", display_df["dataset"] + " / " + display_df["task"])
    display_df = display_df.drop(columns=["dataset", "task"])

    numeric_cols = METHODS

    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    n_rows = len(display_df)
    fig_height = max(5, 0.38 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.35)

    # Header style
    for j in range(len(display_df.columns)):
        cell = table[0, j]
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")

    # Body style
    for i in range(1, n_rows + 1):
        row_color = "#f9fafb" if i % 2 == 0 else "white"

        for j in range(len(display_df.columns)):
            cell = table[i, j]
            cell.set_facecolor(row_color)
            cell.set_edgecolor("#d1d5db")

    col_positions = {col: idx for idx, col in enumerate(display_df.columns)}

    original_numeric = table_df.copy()

    for i, row in original_numeric.iterrows():
        row_num = list(original_numeric.index).index(i) + 1

        bin_values = row[BINS].dropna()

        if bin_values.empty:
            continue

        best_bin = bin_values.astype(float).idxmax()

        if best_bin in col_positions:
            j = col_positions[best_bin]
            cell = table[row_num, j]
            cell.set_facecolor("#bbf7d0")
            cell.set_text_props(weight="bold", color="#14532d")

    ax.set_title(
        "ROC-AUC of Full Graph vs. Curvature Bins",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] AUC table figure -> {output_path}")


# =========================
# Load all sensitivity CSVs
# =========================
files = sorted(SENS_DIR.glob("RNNResultsAllTasks_*_Sensitivity.csv"))

if not files:
    raise FileNotFoundError(f"No sensitivity files found in: {SENS_DIR}")

all_rows = []
winners_rows = []

for fp in files:
    df = pd.read_csv(fp)

    missing = [c for c in [TASK_COL, NETWORK_COL, METRIC_COL] if c not in df.columns]

    if missing:
        print(f"[SKIP] {fp.name}: missing columns {missing}")
        continue

    parsed = df[NETWORK_COL].apply(parse_network)
    ok = parsed.notna()

    if not ok.any():
        print(f"[SKIP] {fp.name}: could not parse any Network values")
        print("Sample Network values:")
        print(df[NETWORK_COL].dropna().astype(str).head(10).to_string(index=False))
        continue

    df = df.loc[ok].copy()
    parsed_df = pd.DataFrame([p for p in parsed[ok]])

    df = pd.concat(
        [df.reset_index(drop=True), parsed_df.reset_index(drop=True)],
        axis=1
    )

    df = df[df[METRIC_COL].notna()].copy()
    df = df[df["Method"].isin(METHODS)].copy()

    if df.empty:
        print(f"[SKIP] {fp.name}: no Full/Bin rows after parsing")
        continue

    # Keep max ROC-AUC for each dataset, task, method
    best_method_values = (
        df.groupby(["dataset", TASK_COL, "Method"], as_index=False)[METRIC_COL]
        .max()
    )

    all_rows.append(best_method_values)

    # Only bins for winners
    bin_df = best_method_values[best_method_values["Method"].isin(BINS)].copy()

    if bin_df.empty:
        continue

    best_per_task = (
        bin_df.sort_values(METRIC_COL, ascending=False)
        .groupby(["dataset", TASK_COL], as_index=False)
        .first()
        .rename(columns={"Method": "best_bin", METRIC_COL: "best_bin_auc"})
    )

    for r in best_per_task.itertuples(index=False):
        winners_rows.append({
            "dataset": r.dataset,
            "task": getattr(r, TASK_COL),
            "best_bin": r.best_bin,
            "best_bin_auc": float(r.best_bin_auc),
            "source_file": fp.name
        })


if not all_rows:
    raise RuntimeError("No valid rows found. Check parsing and file formats.")

all_values = pd.concat(all_rows, ignore_index=True)

winners = pd.DataFrame(winners_rows)

if winners.empty:
    raise RuntimeError("No winners computed. Check parsing and file formats.")


# =========================
# Count table
# =========================
counts = (
    winners.pivot_table(
        index="task",
        columns="best_bin",
        values="dataset",
        aggfunc="count",
        fill_value=0
    )
    .reindex(index=TASKS, fill_value=0)
    .reindex(columns=BINS, fill_value=0)
)

counts["Total"] = counts.sum(axis=1)
counts.loc["Total"] = counts.sum(axis=0)


# =========================
# Values table: Full vs Bin1..Bin5
# =========================
values_table = (
    all_values.pivot_table(
        index=["dataset", TASK_COL],
        columns="Method",
        values=METRIC_COL,
        aggfunc="max"
    )
    .reset_index()
    .rename(columns={TASK_COL: "task"})
)

for col in METHODS:
    if col not in values_table.columns:
        values_table[col] = np.nan

values_table = values_table[["dataset", "task"] + METHODS]


# =========================
# Save outputs
# =========================
counts.to_csv(OUT_COUNTS)
winners.to_csv(OUT_DETAILS, index=False)
values_table.to_csv(OUT_VALUES_TABLE, index=False)

print(f"[SAVED] Counts table -> {OUT_COUNTS}")
print(f"[SAVED] Winners detail -> {OUT_DETAILS}")
print(f"[SAVED] Values table -> {OUT_VALUES_TABLE}")

print("\nCounts preview:\n", counts)
print("\nValues table preview:\n", values_table.head())


# =========================
# Save presentation figures
# =========================
save_presentation_winning_bin_plot(counts, OUT_PRESENTATION_PLOT)
save_auc_table_figure(values_table, OUT_TABLE_FIG)

print(f"\nPresentation figures saved in:\n{FIG_DIR}")