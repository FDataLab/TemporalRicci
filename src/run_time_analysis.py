from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\azadp\PycharmProjects\TemporalRicci")

GRAPH_PULSE_ROOT = PROJECT_ROOT / "GraphPulseResultsWindowed"

PROCESS_TIME_CSV = GRAPH_PULSE_ROOT / "process_data_time.csv"
RNN_RUNTIME_DIR = GRAPH_PULSE_ROOT / "data" / "output"

RICCI_RUNTIME_CSV = PROJECT_ROOT / "RicciResults" / "runtime" / "tfrc_sparsification_runtime.csv"

OUT_DIR = PROJECT_ROOT / "results" / "runtime_comparison_all_methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "tgbl-comment",
    "tgbl_review",
    "tgbl_coin"
]

TARGET_BIN = "bin5"

BASELINE_METHODS = {
    "MoG": {
        "results_dir": PROJECT_ROOT / "GraphPulseResultsMoG" / "results",
        "runtime_file_patterns": [
            "RNN_Runtime_MoG_{dataset}.csv",
            "RNN_PredictionRuntime_MoG_{dataset}.csv",
        ],
        "process_time_csv": PROJECT_ROOT / "GraphPulseResultsMoG" / "run_times_mog.csv",
        "network_key": "mog_er",
        "sparsification_runtime_csv": PROJECT_ROOT / "mog_sparsified" / "mog_sparsification_runtime.csv",
    },
    "SEM": {
        "results_dir": PROJECT_ROOT / "GraphPulseResultsSEM" / "results",
        "runtime_file_patterns": [
            "RNN_Runtime_SEM_{dataset}.csv",
            "RNN_PredictionRuntime_SEM_{dataset}.csv",
        ],
        "process_time_csv": PROJECT_ROOT / "GraphPulseResultsSEM" / "run_times_sem.csv",
        "network_key": "sem_bfc",
        "sparsification_runtime_csv": PROJECT_ROOT / "sem_sparsified" / "sem_sparsification_runtime.csv",
    },
    "TEDDY": {
        "results_dir": PROJECT_ROOT / "GraphPulseResultsTeddy" / "results",
        "runtime_file_patterns": [
            "RNN_Runtime_Teddy_{dataset}.csv",
            "RNN_Runtime_TEDDY_{dataset}.csv",
            "RNN_PredictionRuntime_Teddy_{dataset}.csv",
            "RNN_PredictionRuntime_TEDDY_{dataset}.csv",
        ],
        "process_time_csv": PROJECT_ROOT / "GraphPulseResultsTeddy" / "run_times_teddy.csv",
        "network_key": "teddy",
        "sparsification_runtime_csv": PROJECT_ROOT / "teddy_sparsified" / "teddy_sparsification_runtime.csv",
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace("_", "-").replace(" ", "-")


def seconds_to_min_sec(seconds):
    if pd.isna(seconds):
        return ""

    seconds = float(seconds)

    if seconds < 0:
        return f"-{seconds_to_min_sec(abs(seconds))}"

    minutes = int(seconds // 60)
    sec = seconds % 60

    if minutes == 0:
        return f"{sec:.1f}s"

    return f"{minutes}m {sec:.1f}s"


def dataset_variants(dataset: str) -> list[str]:
    return list(dict.fromkeys([
        dataset,
        dataset.upper(),
        dataset.lower(),
        dataset.replace("-", "_"),
        dataset.replace("-", "_").upper(),
        dataset.replace("-", "_").lower(),
        dataset.replace("_", "-"),
        dataset.replace("_", "-").upper(),
        dataset.replace("_", "-").lower(),
    ]))


def find_file_with_patterns(base_dir: Path, patterns: list[str], dataset: str) -> Path | None:
    for variant in dataset_variants(dataset):
        for pattern in patterns:
            p = base_dir / pattern.format(dataset=variant)
            if p.exists():
                return p
    return None


def get_time_col(df: pd.DataFrame) -> str | None:
    for c in [
        "TotalSeconds",
        "total_seconds",
        "total_sec",
        "runtime_sec",
        "seconds",
        "duration_sec",
        "total_duration_sec",
    ]:
        if c in df.columns:
            return c
    return None


def extract_graph_type_from_network(network: str) -> str | None:
    s = str(network).strip().lower()

    if "full" in s:
        return "full"

    if TARGET_BIN.lower() in s:
        return TARGET_BIN.lower()

    return None


def extract_dataset_and_variant_from_process(row: pd.Series):
    dataset = str(row["dataset"]).strip()
    variant = str(row["variant"]).strip().lower()

    if "full" in variant:
        graph_type = "full"
    elif TARGET_BIN.lower() in variant:
        graph_type = TARGET_BIN.lower()
    else:
        graph_type = None

    return dataset, graph_type


def safe_pct(saved, base):
    if pd.isna(saved) or pd.isna(base) or base == 0:
        return np.nan

    return 100.0 * saved / base


def load_sparsification_runtime(csv_path: Path, selected_dataset_norms: set[str]) -> dict:
    if not csv_path.exists():
        print(f"[WARN] Missing sparsification runtime CSV: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    if "dataset" not in df.columns:
        print(f"[WARN] No dataset column in sparsification runtime CSV: {csv_path}")
        return {}

    time_col = get_time_col(df)

    if time_col is None:
        print(f"[WARN] No time column found in sparsification runtime CSV: {csv_path}")
        print(f"       Columns: {list(df.columns)}")
        return {}

    df["dataset_norm"] = df["dataset"].apply(normalize_name)
    df = df[df["dataset_norm"].isin(selected_dataset_norms)].copy()

    if df.empty:
        return {}

    df["_row_order"] = np.arange(len(df))

    latest = (
        df.sort_values("_row_order")
        .groupby("dataset_norm", as_index=False)
        .tail(1)
    )

    return dict(
        zip(
            latest["dataset_norm"],
            pd.to_numeric(latest[time_col], errors="coerce").fillna(0.0),
        )
    )


def load_baseline_processing_runtime(
    csv_path: Path,
    selected_dataset_norms: set[str],
) -> dict:
    """
    Reads run_times_mog.csv / run_times_sem.csv / run_times_teddy.csv.
    Expected columns are flexible, but process_mog.py writes:
        dataset, variant, phase, seconds
    We sum all rows for each dataset variant.
    """
    if not csv_path.exists():
        print(f"[WARN] Missing baseline process time CSV: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    if "dataset" not in df.columns:
        print(f"[WARN] No dataset column in baseline process CSV: {csv_path}")
        return {}

    time_col = get_time_col(df)

    if time_col is None:
        print(f"[WARN] No time column found in baseline process CSV: {csv_path}")
        print(f"       Columns: {list(df.columns)}")
        return {}

    df["dataset_norm"] = df["dataset"].apply(normalize_name)

    # For rows like ADX_mog_er, recover ADX by checking containment.
    fixed_norms = []
    for raw_norm in df["dataset_norm"]:
        matched = None
        for dn in selected_dataset_norms:
            if raw_norm == dn or raw_norm.startswith(dn + "-") or raw_norm.startswith(dn + "_"):
                matched = dn
                break
        fixed_norms.append(matched)

    df["dataset_norm_fixed"] = fixed_norms
    df = df[df["dataset_norm_fixed"].notna()].copy()

    if df.empty:
        return {}

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)

    summary = (
        df.groupby("dataset_norm_fixed", as_index=False)
        .agg(processing_time_sec=(time_col, "sum"))
    )

    return dict(zip(summary["dataset_norm_fixed"], summary["processing_time_sec"]))


def load_baseline_prediction_runtime(
    runtime_file: Path,
    network_key: str,
) -> float:
    df = pd.read_csv(runtime_file)

    time_col = get_time_col(df)

    if time_col is None:
        raise ValueError(
            f"No runtime column found in {runtime_file}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "Network" in df.columns:
        sub = df[
            df["Network"].astype(str)
            .str.lower()
            .str.contains(network_key.lower(), regex=False)
        ].copy()
    else:
        sub = df.copy()

    return pd.to_numeric(sub[time_col], errors="coerce").fillna(0.0).sum()


# ============================================================
# SELECTED DATASETS
# ============================================================

selected_dataset_norms = {normalize_name(d) for d in DATASETS}


# ============================================================
# LOAD FULL + OUR METHOD PROCESSING TIME
# ============================================================

if not PROCESS_TIME_CSV.exists():
    raise FileNotFoundError(f"Could not find process time CSV:\n{PROCESS_TIME_CSV}")

process_df = pd.read_csv(PROCESS_TIME_CSV)

required_process_cols = {"dataset", "variant", "total_duration_sec"}
missing = required_process_cols - set(process_df.columns)

if missing:
    raise ValueError(
        f"process_data_time.csv is missing columns: {missing}\n"
        f"Available columns: {list(process_df.columns)}"
    )

process_rows = []

for _, row in process_df.iterrows():
    dataset, graph_type = extract_dataset_and_variant_from_process(row)

    if normalize_name(dataset) not in selected_dataset_norms:
        continue

    if graph_type not in {"full", TARGET_BIN.lower()}:
        continue

    process_rows.append({
        "dataset": dataset,
        "dataset_norm": normalize_name(dataset),
        "graph_type": graph_type,
        "processing_time_sec": float(row["total_duration_sec"]),
    })

process_clean = pd.DataFrame(process_rows)

if process_clean.empty:
    process_summary = pd.DataFrame(columns=[
        "dataset",
        "dataset_norm",
        "graph_type",
        "processing_time_sec",
    ])
else:
    process_summary = (
        process_clean
        .groupby(["dataset", "dataset_norm", "graph_type"], as_index=False)
        .agg(processing_time_sec=("processing_time_sec", "sum"))
    )


# ============================================================
# LOAD FULL + OUR METHOD PREDICTOR TIME
# ============================================================

rnn_rows = []

for dataset in DATASETS:
    runtime_file = find_file_with_patterns(
        RNN_RUNTIME_DIR,
        ["RNN_PredictionRuntime_{dataset}.csv", "RNN_Runtime_{dataset}.csv"],
        dataset,
    )

    if runtime_file is None:
        print(f"[SKIP] Missing full/bin5 RNN runtime file for {dataset}")
        continue

    df = pd.read_csv(runtime_file)

    time_col = get_time_col(df)

    if time_col is None:
        raise ValueError(
            f"No runtime column found in {runtime_file}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "Network" not in df.columns:
        raise ValueError(
            f"{runtime_file} is missing Network column.\n"
            f"Available columns: {list(df.columns)}"
        )

    for _, row in df.iterrows():
        graph_type = extract_graph_type_from_network(row["Network"])

        if graph_type not in {"full", TARGET_BIN.lower()}:
            continue

        rnn_rows.append({
            "dataset": dataset,
            "dataset_norm": normalize_name(dataset),
            "method": "Full graph" if graph_type == "full" else "Our method",
            "graph_type": graph_type,
            "prediction_time_sec": float(row[time_col]),
        })

rnn_clean = pd.DataFrame(rnn_rows)

if rnn_clean.empty:
    rnn_summary = pd.DataFrame(columns=[
        "dataset",
        "dataset_norm",
        "method",
        "graph_type",
        "prediction_time_sec",
    ])
else:
    rnn_summary = (
        rnn_clean
        .groupby(["dataset", "dataset_norm", "method", "graph_type"], as_index=False)
        .agg(prediction_time_sec=("prediction_time_sec", "sum"))
    )


# ============================================================
# LOAD OUR METHOD SPARSIFICATION TIME
# ============================================================

if not RICCI_RUNTIME_CSV.exists():
    raise FileNotFoundError(
        f"Could not find Ricci runtime CSV:\n{RICCI_RUNTIME_CSV}"
    )

ricci_df = pd.read_csv(RICCI_RUNTIME_CSV)

required_ricci_cols = {"dataset", "total_seconds"}
missing = required_ricci_cols - set(ricci_df.columns)

if missing:
    raise ValueError(
        f"{RICCI_RUNTIME_CSV} is missing columns: {missing}\n"
        f"Available columns: {list(ricci_df.columns)}"
    )

ricci_df["dataset_norm"] = ricci_df["dataset"].apply(normalize_name)
ricci_df = ricci_df[ricci_df["dataset_norm"].isin(selected_dataset_norms)].copy()

ricci_df["sparsification_time_sec"] = pd.to_numeric(
    ricci_df["total_seconds"],
    errors="coerce"
).fillna(0.0)

ricci_df["_row_order"] = np.arange(len(ricci_df))

if ricci_df.empty:
    ricci_summary = pd.DataFrame(columns=[
        "dataset",
        "dataset_norm",
        "sparsification_time_sec",
        "method",
    ])
else:
    ricci_summary = (
        ricci_df
        .sort_values("_row_order")
        .groupby("dataset_norm", as_index=False)
        .tail(1)
        [["dataset", "dataset_norm", "sparsification_time_sec"]]
        .reset_index(drop=True)
    )
    ricci_summary["method"] = "Our method"


# ============================================================
# BUILD FULL + OUR METHOD RECORDS
# ============================================================

records = []

for dataset in DATASETS:
    dn = normalize_name(dataset)

    full_process = process_summary[
        (process_summary["dataset_norm"] == dn) &
        (process_summary["graph_type"] == "full")
    ]["processing_time_sec"].sum()

    bin_process = process_summary[
        (process_summary["dataset_norm"] == dn) &
        (process_summary["graph_type"] == TARGET_BIN.lower())
    ]["processing_time_sec"].sum()

    full_pred = rnn_summary[
        (rnn_summary["dataset_norm"] == dn) &
        (rnn_summary["method"] == "Full graph")
    ]["prediction_time_sec"].sum()

    bin_pred = rnn_summary[
        (rnn_summary["dataset_norm"] == dn) &
        (rnn_summary["method"] == "Our method")
    ]["prediction_time_sec"].sum()

    ricci_time = ricci_summary[
        ricci_summary["dataset_norm"] == dn
    ]["sparsification_time_sec"].sum()

    records.append({
        "dataset": dataset,
        "dataset_norm": dn,
        "method": "Full graph",
        "sparsification_time_sec": 0.0,
        "processing_time_sec": full_process,
        "prediction_time_sec": full_pred,
        "total_time_sec": full_process + full_pred,
    })

    records.append({
        "dataset": dataset,
        "dataset_norm": dn,
        "method": "Our method",
        "sparsification_time_sec": ricci_time,
        "processing_time_sec": bin_process,
        "prediction_time_sec": bin_pred,
        "total_time_sec": ricci_time + bin_process + bin_pred,
    })


# ============================================================
# LOAD BASELINE METHODS
# ============================================================

for method_name, cfg in BASELINE_METHODS.items():
    baseline_processing_runtime = load_baseline_processing_runtime(
        cfg["process_time_csv"],
        selected_dataset_norms,
    )

    baseline_sparsification_runtime = load_sparsification_runtime(
        cfg["sparsification_runtime_csv"],
        selected_dataset_norms,
    )

    for dataset in DATASETS:
        dn = normalize_name(dataset)

        runtime_file = find_file_with_patterns(
            cfg["results_dir"],
            cfg["runtime_file_patterns"],
            dataset,
        )

        prediction_time = 0.0

        if runtime_file is None:
            print(f"[SKIP] Missing predictor runtime file for {method_name} / {dataset}")
        else:
            prediction_time = load_baseline_prediction_runtime(
                runtime_file,
                cfg["network_key"],
            )

        records.append({
            "dataset": dataset,
            "dataset_norm": dn,
            "method": method_name,
            "sparsification_time_sec": baseline_sparsification_runtime.get(dn, 0.0),
            "processing_time_sec": baseline_processing_runtime.get(dn, 0.0),
            "prediction_time_sec": prediction_time,
            "total_time_sec": (
                baseline_sparsification_runtime.get(dn, 0.0) +
                baseline_processing_runtime.get(dn, 0.0) +
                prediction_time
            ),
        })


runtime_df = pd.DataFrame(records)


# ============================================================
# SAVING PERCENT RELATIVE TO FULL GRAPH
# ============================================================

full_totals = (
    runtime_df[runtime_df["method"] == "Full graph"]
    [["dataset_norm", "total_time_sec"]]
    .rename(columns={"total_time_sec": "full_total_time_sec"})
)

runtime_df = runtime_df.merge(full_totals, on="dataset_norm", how="left")

runtime_df["saved_sec_vs_full"] = (
    runtime_df["full_total_time_sec"] -
    runtime_df["total_time_sec"]
)

runtime_df["saved_pct_vs_full"] = runtime_df.apply(
    lambda r: safe_pct(r["saved_sec_vs_full"], r["full_total_time_sec"]),
    axis=1,
)


# ============================================================
# SAVE RAW NUMBERS TABLE
# ============================================================

method_order = ["Full graph", "Our method", "MoG", "SEM", "TEDDY"]

raw_table = runtime_df.copy()

raw_table = raw_table[[
    "dataset",
    "method",
    "sparsification_time_sec",
    "processing_time_sec",
    "prediction_time_sec",
    "total_time_sec",
    "saved_sec_vs_full",
    "saved_pct_vs_full",
]]

raw_table = raw_table.rename(columns={
    "dataset": "Dataset",
    "method": "Method",
    "sparsification_time_sec": "Sparsification (s)",
    "processing_time_sec": "Processing (s)",
    "prediction_time_sec": "Predictor (s)",
    "total_time_sec": "Total (s)",
    "saved_sec_vs_full": "Saved (s)",
    "saved_pct_vs_full": "Saved (%)",
})

raw_table["Method"] = pd.Categorical(
    raw_table["Method"],
    categories=method_order,
    ordered=True
)

raw_table = raw_table.sort_values(["Dataset", "Method"]).reset_index(drop=True)

raw_table = raw_table.round({
    "Sparsification (s)": 3,
    "Processing (s)": 3,
    "Predictor (s)": 3,
    "Total (s)": 3,
    "Saved (s)": 3,
    "Saved (%)": 2,
})

raw_table_out = OUT_DIR / "runtime_all_methods_raw_numbers_table.csv"
raw_table.to_csv(raw_table_out, index=False)


# ============================================================
# SAVE FORMATTED TABLE
# ============================================================

formatted_table = raw_table.copy()

for c in [
    "Sparsification (s)",
    "Processing (s)",
    "Predictor (s)",
    "Total (s)",
    "Saved (s)",
]:
    formatted_table[c.replace(" (s)", "")] = formatted_table[c].apply(seconds_to_min_sec)
    formatted_table = formatted_table.drop(columns=[c])

formatted_table = formatted_table[[
    "Dataset",
    "Method",
    "Sparsification",
    "Processing",
    "Predictor",
    "Total",
    "Saved",
    "Saved (%)",
]]

formatted_table_out = OUT_DIR / "runtime_all_methods_formatted_table.csv"
formatted_table.to_csv(formatted_table_out, index=False)


# ============================================================
# AVERAGE RAW TABLE
# ============================================================

avg_df = (
    runtime_df
    .groupby("method", as_index=False)
    .agg(
        sparsification_time_sec=("sparsification_time_sec", "mean"),
        processing_time_sec=("processing_time_sec", "mean"),
        prediction_time_sec=("prediction_time_sec", "mean"),
        total_time_sec=("total_time_sec", "mean"),
        saved_sec_vs_full=("saved_sec_vs_full", "mean"),
        saved_pct_vs_full=("saved_pct_vs_full", "mean"),
    )
)

avg_df["method"] = pd.Categorical(avg_df["method"], categories=method_order, ordered=True)
avg_df = avg_df.sort_values("method").reset_index(drop=True)

avg_raw_table = avg_df.rename(columns={
    "method": "Method",
    "sparsification_time_sec": "Avg Sparsification (s)",
    "processing_time_sec": "Avg Processing (s)",
    "prediction_time_sec": "Avg Predictor (s)",
    "total_time_sec": "Avg Total (s)",
    "saved_sec_vs_full": "Avg Saved (s)",
    "saved_pct_vs_full": "Avg Saved (%)",
})

avg_raw_table = avg_raw_table.round({
    "Avg Sparsification (s)": 3,
    "Avg Processing (s)": 3,
    "Avg Predictor (s)": 3,
    "Avg Total (s)": 3,
    "Avg Saved (s)": 3,
    "Avg Saved (%)": 2,
})

avg_raw_out = OUT_DIR / "runtime_all_methods_average_raw_numbers_table.csv"
avg_raw_table.to_csv(avg_raw_out, index=False)


# ============================================================
# AVERAGE FORMATTED TABLE
# ============================================================

avg_formatted_table = avg_raw_table.copy()

for c in [
    "Avg Sparsification (s)",
    "Avg Processing (s)",
    "Avg Predictor (s)",
    "Avg Total (s)",
    "Avg Saved (s)",
]:
    avg_formatted_table[c.replace("Avg ", "").replace(" (s)", "")] = (
        avg_formatted_table[c].apply(seconds_to_min_sec)
    )
    avg_formatted_table = avg_formatted_table.drop(columns=[c])

avg_formatted_table = avg_formatted_table.rename(columns={
    "Avg Saved (%)": "Saved (%)"
})

avg_formatted_table = avg_formatted_table[[
    "Method",
    "Sparsification",
    "Processing",
    "Predictor",
    "Total",
    "Saved",
    "Saved (%)",
]]

avg_formatted_out = OUT_DIR / "runtime_all_methods_average_formatted_table.csv"
avg_formatted_table.to_csv(avg_formatted_out, index=False)


print(f"[SAVE] Raw numbers table -> {raw_table_out}")
print(f"[SAVE] Formatted table -> {formatted_table_out}")
print(f"[SAVE] Average raw numbers table -> {avg_raw_out}")
print(f"[SAVE] Average formatted table -> {avg_formatted_out}")

print("\nAverage raw runtime table:")
print(avg_raw_table.to_string(index=False))


# ============================================================
# PLOT 1: STACKED AVERAGE RUNTIME BREAKDOWN
# ============================================================

plot_avg = avg_df[avg_df["method"] != "Full graph"].copy()

fig, ax = plt.subplots(figsize=(7.2, 4.5))

x = np.arange(len(plot_avg))
bar_width = 0.55

ax.bar(
    x,
    plot_avg["sparsification_time_sec"],
    width=bar_width,
    label="Sparsification",
)

ax.bar(
    x,
    plot_avg["processing_time_sec"],
    width=bar_width,
    bottom=plot_avg["sparsification_time_sec"],
    label="Processing",
)

ax.bar(
    x,
    plot_avg["prediction_time_sec"],
    width=bar_width,
    bottom=plot_avg["sparsification_time_sec"] + plot_avg["processing_time_sec"],
    label="Predictor",
)

ax.set_xticks(x)
ax.set_xticklabels(plot_avg["method"])
ax.set_ylabel("Average runtime (seconds)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.grid(axis="y", alpha=0.18)
ax.grid(axis="x", visible=False)

stacked_out = OUT_DIR / "runtime_average_breakdown_stacked.png"
fig.savefig(stacked_out, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[SAVE] Stacked runtime plot -> {stacked_out}")


# ============================================================
# PLOT 2: TOTAL AVERAGE RUNTIME INCLUDING FULL GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.2))

plot_total = avg_df.copy()
x = np.arange(len(plot_total))

ax.bar(
    x,
    plot_total["total_time_sec"],
    width=0.55,
)

ax.set_xticks(x)
ax.set_xticklabels(plot_total["method"])
ax.set_ylabel("Average total runtime (seconds)")
ax.grid(axis="y", alpha=0.18)
ax.grid(axis="x", visible=False)

total_out = OUT_DIR / "runtime_average_total_comparison.png"
fig.savefig(total_out, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[SAVE] Total runtime plot -> {total_out}")


# ============================================================
# PLOT 3: SAVED PERCENTAGE RELATIVE TO FULL GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.2))

plot_saved = avg_df[avg_df["method"] != "Full graph"].copy()
x = np.arange(len(plot_saved))

ax.bar(
    x,
    plot_saved["saved_pct_vs_full"],
    width=0.55,
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(plot_saved["method"])
ax.set_ylabel("Average runtime saving vs. full graph (%)")
ax.grid(axis="y", alpha=0.18)
ax.grid(axis="x", visible=False)

saved_out = OUT_DIR / "runtime_average_saved_percentage.png"
fig.savefig(saved_out, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[SAVE] Runtime saving plot -> {saved_out}")