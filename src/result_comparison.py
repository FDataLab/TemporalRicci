#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent

OUTPUT_DIR = _BASE_DIR / "comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TFRC_RESULTS_OVERRIDE: "Path | None" = None

# ============================================================
# DATASET GROUPS
# ============================================================

DATASET_GROUPS = {
    "token": [
        "ADX",
        "BAG",
        "BEPRO",
        "DERC",
        "DINO",
        "ETH2x-FLI",
        "EVERMOON",
        "GLM",
        "HOICHI",
    ],
    "tgbl": [
        "tgbl-coin",
        "tgbl-review",
        "tgbl-comment"
    ],
}

TASKS = {
    "task1": "Network Activity Growth",
    "task2": "Influential Node Turnover",
    "task3": "Network Participation Expansion",
}

# ============================================================
# METHODS
# ============================================================

METHODS = {
    "TFRC": {
        "label": "TRicci",
        "color": "#6A51A3",
        "results_dir": _BASE_DIR / "GraphPulseResultsWindowed" / "data" / "output",
        "file_tpl": "RNNResultsAllTasks_{dataset}_Sensitivity.csv",
        "net_suffix": "bin",
    },
    "MoG": {
        "label": "MoG",
        "color": "#4C78A8",
        "results_dir": _BASE_DIR / "GraphPulseResultsMoG" / "results",
        "file_tpl": "RNNResults_MoG_{dataset}.csv",
        "net_suffix": "mog_er",
    },
    "TEDDY": {
        "label": "TEDDY",
        "color": "#72B7B2",
        "results_dir": _BASE_DIR / "GraphPulseResultsTeddy" / "results",
        "file_tpl": "RNNResults_Teddy_{dataset}.csv",
        "net_suffix": "teddy",
    },
    "SEM": {
        "label": "SEM",
        "color": "#9ECAE1",
        "results_dir": _BASE_DIR / "GraphPulseResultsSEM" / "results",
        "file_tpl": "RNNResults_SEM_{dataset}.csv",
        "net_suffix": "sem_bfc",
    },
}

METHOD_KEYS = list(METHODS.keys())

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# ============================================================
# HELPERS
# ============================================================

def pretty_dataset_name(name: str) -> str:
    return name.replace("tgbl-", "TGBL-")


def _candidate_paths(method_key: str, dataset: str) -> list[Path]:
    cfg = METHODS[method_key]
    fname = cfg["file_tpl"].format(dataset=dataset)

    if method_key != "TFRC":
        return [cfg["results_dir"] / fname]

    root = _BASE_DIR / "GraphPulseResultsWindowed"

    candidates = []

    if TFRC_RESULTS_OVERRIDE is not None:
        candidates.append(Path(TFRC_RESULTS_OVERRIDE) / fname)

    candidates += [
        root / "data" / "output" / fname,
        root / "data" / "output" / dataset / fname,
        root / "results" / fname,
        root / "results" / dataset / fname,
        root / "output" / fname,
        root / "output" / dataset / fname,
        root / fname,
    ]

    return candidates


def _normalize_task(task_raw_val) -> str | None:
    task_raw = str(task_raw_val).strip().lower()

    if task_raw in ("task1", "1"):
        return "task1"
    if task_raw in ("task2", "2"):
        return "task2"
    if task_raw in ("task3", "3"):
        return "task3"

    return None


# ============================================================
# DATA LOADING
# ============================================================

def load_method_results(method_key: str, datasets: list[str]) -> pd.DataFrame:
    cfg = METHODS[method_key]
    rows = []
    suffix = cfg["net_suffix"]

    for ds in datasets:
        csv_path = None
        tried = []

        for p in _candidate_paths(method_key, ds):
            tried.append(p)

            if p.exists():
                csv_path = p
                break

        if csv_path is None:
            if method_key == "TFRC":
                print(f"  [DIAG] TFRC: no file found for {ds}. Tried:")
                for p in tried:
                    print(f"           {p}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] Could not read {csv_path}: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]

        if "Network" not in df.columns or "Task" not in df.columns or "ROC_AUC" not in df.columns:
            continue

        if method_key == "TFRC":
            df = df[df["Network"].astype(str).str.contains("bin", regex=False)]
        else:
            df = df[df["Network"].astype(str).str.contains(suffix, regex=False)]

        if df.empty:
            continue

        for task_raw_val in df["Task"].unique():
            task = _normalize_task(task_raw_val)

            if task is None:
                continue

            task_rows = df[
                df["Task"].astype(str).str.strip().str.lower()
                .isin([str(task_raw_val).strip().lower()])
            ].copy()

            task_rows["_roc"] = pd.to_numeric(task_rows["ROC_AUC"], errors="coerce")
            task_rows = task_rows.dropna(subset=["_roc"])

            if task_rows.empty:
                continue

            best_row = task_rows.sort_values("_roc", ascending=False).iloc[0]

            rows.append({
                "dataset": ds,
                "task": task,
                "method": method_key,
                "roc_auc": float(best_row["_roc"]),
                "auc_avg": float(best_row.get("AUC_AVG", np.nan)),
                "auc_std": float(best_row.get("AUC_STD", np.nan)),
                "num_samples": int(best_row.get("NumSamples", 0)),
                "network": str(best_row.get("Network", "")),
            })

    return pd.DataFrame(rows)


def load_full_graph_results(datasets: list[str]) -> pd.DataFrame:
    rows = []

    for ds in datasets:
        csv_path = None

        for p in _candidate_paths("TFRC", ds):
            if p.exists():
                csv_path = p
                break

        if csv_path is None:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        df.columns = [c.strip() for c in df.columns]

        if "Network" not in df.columns or "Task" not in df.columns or "ROC_AUC" not in df.columns:
            continue

        full_df = df[df["Network"].astype(str).str.endswith("_full")].copy()

        if full_df.empty:
            continue

        for task_raw_val in full_df["Task"].unique():
            task = _normalize_task(task_raw_val)

            if task is None:
                continue

            task_rows = full_df[
                full_df["Task"].astype(str).str.strip().str.lower()
                .isin([str(task_raw_val).strip().lower()])
            ].copy()

            task_rows["_roc"] = pd.to_numeric(task_rows["ROC_AUC"], errors="coerce")
            task_rows = task_rows.dropna(subset=["_roc"])

            if task_rows.empty:
                continue

            best_row = task_rows.sort_values("_roc", ascending=False).iloc[0]

            rows.append({
                "dataset": ds,
                "task": task,
                "method": "FULL",
                "roc_auc": float(best_row["_roc"]),
                "auc_avg": np.nan,
                "auc_std": np.nan,
                "num_samples": int(best_row.get("NumSamples", 0)),
                "network": str(best_row.get("Network", "")),
            })

    return pd.DataFrame(rows)


def load_all(datasets: list[str]) -> pd.DataFrame:
    frames = []

    for mk in METHOD_KEYS:
        df = load_method_results(mk, datasets)

        if not df.empty:
            frames.append(df)
        else:
            print(f"  [INFO] No results found for method: {mk}")

    full_df = load_full_graph_results(datasets)

    if not full_df.empty:
        frames.append(full_df)
        print("  [INFO] Loaded full graph ROC-AUC reference rows.")

    if not frames:
        print("[FATAL] No result files found.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    print("\n  Loading summary:")

    for mk in METHOD_KEYS + ["FULL"]:
        sub = combined[combined["method"] == mk]

        if sub.empty:
            print(f"    {mk:8s}: NOT FOUND")
        else:
            ds_list = sorted(sub["dataset"].unique())
            print(
                f"    {mk:8s}: {len(sub):3d} rows, "
                f"{len(ds_list)} datasets: {ds_list}"
            )

    print()

    return combined


def filter_to_tfrc_valid(df: pd.DataFrame) -> pd.DataFrame:
    tfrc = df[df["method"] == "TFRC"][["dataset", "task", "roc_auc"]].copy()
    tfrc = tfrc.dropna(subset=["roc_auc"])
    tfrc = tfrc[np.isfinite(tfrc["roc_auc"])]

    valid_pairs = set(zip(tfrc["dataset"], tfrc["task"]))

    if not valid_pairs:
        print("  [WARN] Our method has no valid results.")
        return df

    mask = df.apply(
        lambda r: (r["dataset"], r["task"]) in valid_pairs,
        axis=1
    )

    dropped = (~mask).sum()

    if dropped:
        print(f"  [INFO] Dropped {dropped} rows where our method had no result.")

    return df[mask].copy()


def pivot_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = df[
        (df["task"] == task) &
        (df["method"].isin(METHOD_KEYS))
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    tbl = sub.pivot_table(
        index="dataset",
        columns="method",
        values="roc_auc",
        aggfunc="mean"
    )

    cols = [m for m in METHOD_KEYS if m in tbl.columns]

    return tbl[cols]


def get_full_auc(df: pd.DataFrame, dataset: str, task: str) -> float:
    cell = df[
        (df["dataset"] == dataset) &
        (df["task"] == task) &
        (df["method"] == "FULL")
    ]["roc_auc"]

    if cell.empty:
        return np.nan

    return float(cell.iloc[0])


def active_datasets(df: pd.DataFrame, datasets: list[str]) -> list[str]:
    found = set(df["dataset"].unique())
    return [d for d in datasets if d in found]


# ============================================================
# FIGURE -- GROUPED BAR CHART
# ============================================================

def plot_grouped_bar(
    df: pd.DataFrame,
    task: str,
    task_label: str,
    group_name: str
) -> None:
    tbl = pivot_task(df, task)

    if tbl.empty:
        return

    datasets = tbl.index.tolist()

    if group_name == "tgbl":
        dataset_width = 0.95
        left_margin_units = 0.45
        right_margin_units = 0.45
        fig_height = 2.5
        fixed_bar_width = 0.14
        group_spacing = 0.17
        min_fig_width = 4.6
    else:
        dataset_width = 1.05
        left_margin_units = 0.55
        right_margin_units = 0.55
        fig_height = 3
        fixed_bar_width = 0.16
        group_spacing = 0.18
        min_fig_width = 4.8

    fig_width = left_margin_units + len(datasets) * dataset_width + right_margin_units
    fig_width = max(fig_width, min_fig_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x = np.arange(len(datasets)) * dataset_width
    used_labels = set()

    for i, ds in enumerate(datasets):
        available_methods = [
            m for m in METHOD_KEYS
            if m in tbl.columns and not pd.isna(tbl.loc[ds, m])
        ]

        n_m = len(available_methods)
        if n_m == 0:
            continue

        for j, m in enumerate(available_methods):
            v = tbl.loc[ds, m]
            offset = x[i] + (j - (n_m - 1) / 2) * group_spacing

            ax.bar(
                offset,
                v,
                width=fixed_bar_width,
                color=METHODS[m]["color"],
                label=METHODS[m]["label"] if m not in used_labels else None,
                alpha=0.96,
                linewidth=0,
                edgecolor="none",
                zorder=3
            )
            used_labels.add(m)

        full_auc = get_full_auc(df, ds, task)
        if not np.isnan(full_auc):
            line_half_width = 0.34 if group_name == "tgbl" else 0.36

            ax.hlines(
                y=full_auc,
                xmin=x[i] - line_half_width,
                xmax=x[i] + line_half_width,
                colors="#A8A8A8",
                linestyles=(0, (4, 2)),
                linewidth=1.1,
                label="Full graph" if "FULL" not in used_labels else None,
                zorder=5
            )
            used_labels.add("FULL")

    pretty_datasets = [pretty_dataset_name(d) for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_datasets, rotation=0, ha="center", fontsize=9)

    all_vals = []
    for col in tbl.columns:
        all_vals.extend(tbl[col].dropna().values.tolist())

    for ds in datasets:
        full_auc = get_full_auc(df, ds, task)
        if not np.isnan(full_auc):
            all_vals.append(full_auc)

    if group_name == "tgbl":
        ax.set_ylim(0.0, 0.85)
    else:
        ax.set_ylim(0.0, min(1.0, max(all_vals) + 0.025))

    ax.set_ylabel("ROC-AUC")

    ax.set_xlim(x[0] - left_margin_units, x[-1] + right_margin_units)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.0
    )

    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", visible=False)

    out = OUTPUT_DIR / f"grouped_bar_{group_name}_{task}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"  Saved: {out.name}")

def compute_and_save_preservation(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """
    Computes preservation ratio:
        sparse ROC-AUC / full graph ROC-AUC

    Important:
      Uses only common dataset-task pairs where ALL methods have valid results.
      If one method is missing for a dataset-task pair, that pair is removed
      for every method to keep the comparison fair.

    Saves:
      1. Raw dataset-task-method ratios
      2. Average preservation per method
      3. Common valid pairs used in the comparison
    """

    # ------------------------------------------------------------
    # Full graph reference
    # ------------------------------------------------------------

    full_df = (
        df[df["method"] == "FULL"]
        [["dataset", "task", "roc_auc"]]
        .rename(columns={"roc_auc": "full_roc_auc"})
        .dropna(subset=["full_roc_auc"])
    )

    full_df = full_df[full_df["full_roc_auc"] > 0]

    # ------------------------------------------------------------
    # Sparse-method results
    # ------------------------------------------------------------

    sparse_df = df[df["method"].isin(METHOD_KEYS)].copy()
    sparse_df = sparse_df.dropna(subset=["roc_auc"])

    # Attach full graph ROC-AUC
    merged = sparse_df.merge(
        full_df,
        on=["dataset", "task"],
        how="inner"
    )

    merged = merged.dropna(subset=["roc_auc", "full_roc_auc"])
    merged = merged[merged["full_roc_auc"] > 0]

    if merged.empty:
        print(f"  [WARN] No valid preservation rows for group: {group_name}")
        return merged

    # ------------------------------------------------------------
    # Keep only common dataset-task pairs across ALL methods
    # ------------------------------------------------------------

    required_methods = set(METHOD_KEYS)

    pair_method_counts = (
        merged.groupby(["dataset", "task"])["method"]
        .apply(lambda x: set(x.dropna()))
        .reset_index(name="available_methods")
    )

    pair_method_counts["has_all_methods"] = pair_method_counts["available_methods"].apply(
        lambda s: required_methods.issubset(s)
    )

    common_pairs_df = pair_method_counts[pair_method_counts["has_all_methods"]][
        ["dataset", "task"]
    ].copy()

    if common_pairs_df.empty:
        print(
            f"  [WARN] No common dataset-task pairs where all methods "
            f"have valid results for group: {group_name}"
        )
        return pd.DataFrame()

    # Save the common pairs for transparency
    common_pairs_out = OUTPUT_DIR / f"performance_preservation_common_pairs_{group_name}.csv"
    common_pairs_df.to_csv(common_pairs_out, index=False)

    before_rows = len(merged)
    before_pairs = merged[["dataset", "task"]].drop_duplicates().shape[0]

    merged = merged.merge(
        common_pairs_df,
        on=["dataset", "task"],
        how="inner"
    )

    after_rows = len(merged)
    after_pairs = merged[["dataset", "task"]].drop_duplicates().shape[0]

    print(
        f"  Common-pair filter: {before_pairs} dataset-task pairs -> "
        f"{after_pairs} common pairs"
    )
    print(
        f"  Rows after common-pair filter: {before_rows} -> {after_rows}"
    )

    # ------------------------------------------------------------
    # Compute preservation
    # ------------------------------------------------------------

    merged["preservation_ratio"] = merged["roc_auc"] / merged["full_roc_auc"]
    merged["preservation_percent"] = merged["preservation_ratio"] * 100.0
    merged["method_label"] = merged["method"].map(
        {k: METHODS[k]["label"] for k in METHOD_KEYS}
    )

    raw_cols = [
        "dataset",
        "task",
        "method",
        "method_label",
        "network",
        "full_roc_auc",
        "roc_auc",
        "preservation_ratio",
        "preservation_percent",
    ]

    raw_out = OUTPUT_DIR / f"performance_preservation_raw_{group_name}.csv"
    merged[raw_cols].to_csv(raw_out, index=False)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = (
        merged.groupby(["method", "method_label"], as_index=False)
        .agg(
            valid_pairs=("preservation_ratio", "count"),
            avg_preservation_ratio=("preservation_ratio", "mean"),
            avg_preservation_percent=("preservation_percent", "mean"),
            std_preservation_percent=("preservation_percent", "std"),
        )
    )

    # Make sure all methods have the same number of valid pairs
    expected_pairs = after_pairs

    summary["expected_valid_pairs"] = expected_pairs
    summary["same_pair_count"] = summary["valid_pairs"] == expected_pairs

    summary = summary.sort_values("avg_preservation_percent", ascending=False)

    summary_out = OUTPUT_DIR / f"performance_preservation_summary_{group_name}.csv"
    summary.to_csv(summary_out, index=False)

    # ------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------

    latex_out = OUTPUT_DIR / f"performance_preservation_summary_{group_name}.tex"

    with open(latex_out, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Average predictive-performance preservation relative to the full graph.}\n")
        f.write(f"\\label{{tab:preservation_{group_name}}}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Method & Common pairs & Avg. Preservation (\\%) \\\\\n")
        f.write("\\midrule\n")

        for _, row in summary.iterrows():
            f.write(
                f"{row['method_label']} & "
                f"{int(row['valid_pairs'])} & "
                f"{row['avg_preservation_percent']:.2f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  Saved common pairs: {common_pairs_out.name}")
    print(f"  Saved preservation raw: {raw_out.name}")
    print(f"  Saved preservation summary: {summary_out.name}")
    print(f"  Saved LaTeX table: {latex_out.name}")

    print("\n  Preservation summary using common dataset-task pairs:")
    print(summary.to_string(index=False))

    return merged
# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("Our Method vs Baselines -- Comparison Plots")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    all_preservation_rows = []

    for group_name, datasets in DATASET_GROUPS.items():
        print("\n" + "=" * 60)
        print(f"Processing dataset group: {group_name}")
        print(f"Datasets: {datasets}")
        print("=" * 60)

        print("\n[1/3] Loading results ...")

        df = load_all(datasets)

        print(
            f"       Loaded {len(df)} result rows across "
            f"{df['dataset'].nunique()} datasets, "
            f"{df['method'].nunique()} methods."
        )

        print(
            "\n  Filtering: dropping (dataset, task) pairs "
            "where our method has no valid result ..."
        )

        df = filter_to_tfrc_valid(df)

        print(
            f"       After filter: {len(df)} rows, "
            f"{df['dataset'].nunique()} datasets.\n"
        )

        ads = active_datasets(df, datasets)
        print(f"Active datasets ({len(ads)}): {ads}\n")

        print("[2/3] Generating grouped bar figures ...")

        for tk, tl in TASKS.items():
            print(f"\n  Task: {tl}")
            plot_grouped_bar(df, tk, tl, group_name)

        print("\n[3/3] Computing preservation ratios y/x ...")

        preservation_df = compute_and_save_preservation(df, group_name)

        if not preservation_df.empty:
            preservation_df["dataset_group"] = group_name
            all_preservation_rows.append(preservation_df)

    # ============================================================
    # Combined preservation over token + TGBL datasets
    # ============================================================

    if all_preservation_rows:
        all_preservation = pd.concat(all_preservation_rows, ignore_index=True)

        all_raw_out = OUTPUT_DIR / "performance_preservation_raw_all.csv"
        all_preservation.to_csv(all_raw_out, index=False)

        all_summary = (
            all_preservation.groupby(["method", "method_label"], as_index=False)
            .agg(
                valid_pairs=("preservation_ratio", "count"),
                avg_preservation_ratio=("preservation_ratio", "mean"),
                avg_preservation_percent=("preservation_percent", "mean"),
                std_preservation_percent=("preservation_percent", "std"),
            )
        )

        all_summary = all_summary.sort_values(
            "avg_preservation_percent",
            ascending=False
        )

        all_summary_out = OUTPUT_DIR / "performance_preservation_summary_all.csv"
        all_summary.to_csv(all_summary_out, index=False)

        all_latex_out = OUTPUT_DIR / "performance_preservation_summary_all.tex"

        with open(all_latex_out, "w", encoding="utf-8") as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write("\\caption{Average predictive-performance preservation relative to the full graph.}\n")
            f.write("\\label{tab:preservation_all}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Method & Valid pairs & Avg. Preservation (\\%) \\\\\n")
            f.write("\\midrule\n")

            for _, row in all_summary.iterrows():
                f.write(
                    f"{row['method_label']} & "
                    f"{int(row['valid_pairs'])} & "
                    f"{row['avg_preservation_percent']:.2f} \\\\\n"
                )

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print("\n" + "=" * 60)
        print("Combined preservation summary over all dataset groups")
        print("=" * 60)
        print(all_summary.to_string(index=False))
        print(f"\nSaved combined raw file: {all_raw_out}")
        print(f"Saved combined summary file: {all_summary_out}")
        print(f"Saved combined LaTeX table: {all_latex_out}")

    print("\n" + "=" * 60)
    print(f"Done. All figures and preservation files saved to:\n  {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()