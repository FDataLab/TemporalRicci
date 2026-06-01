import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\azadp\PycharmProjects\TemporalRicci")

DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_ROOT = PROJECT_ROOT / "RicciResults" / "ricci_values_windowed_tau_sensitivity"
PLOT_DIR = PROJECT_ROOT / "results" / "tau_sensitivity" / "curvature_distributions"

DATASETS = [
    "ADX",
    "BAG",
    "BEPRO",
    "DFRC",
    "DINO",
    "ETH2X-FLI",
    "EVERMOON",
    "GLM",
    "HOICHI",
    "tgbl-coin",
    "tgbl-review",
]

TAU_VALUES = [0.25, 0.5, 1, 2, 4, 7, 14, 30]

WINDOW_DAYS = 7
WINDOW_STEP_DAYS = 7

N_BINS = 5
CURV_COL = "Temporal Forman-Ricci value"

EPS = 1e-12
SECONDS_PER_DAY = 86400.0

MIN_POSITIVE_WEIGHT = 1e-12
VALUE_SCALE_QUANTILE = 0.99
MIN_NORMALIZED_EDGE_WEIGHT = 0.05
MAX_NORMALIZED_EDGE_WEIGHT = 1.0

TIMESTAMP_TIE_BREAK = 1e-9

HIST_BINS = 60
PLOT_PERCENTILE_LOW = 1
PLOT_PERCENTILE_HIGH = 99


# ============================================================
# HELPERS
# ============================================================

def tau_tag(tau_days: float) -> str:
    return f"tau{tau_days:.2f}".replace(".", "p")


def find_input_file(dataset: str) -> Path | None:
    candidates = [
        DATA_ROOT / f"{dataset}.csv",
        DATA_ROOT / f"{dataset.upper()}.csv",
        DATA_ROOT / f"{dataset.lower()}.csv",
        DATA_ROOT / f"{dataset.replace('-', '_')}.csv",
        DATA_ROOT / f"{dataset.replace('_', '-')}.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def equal_bins(values: np.ndarray, n_bins: int = 5) -> pd.Series:
    s = pd.Series(values, dtype="float64")
    labels = list(range(1, n_bins + 1))

    try:
        bins = pd.qcut(
            s,
            q=n_bins,
            labels=labels,
            duplicates="drop"
        )

        if pd.Series(bins).nunique() < n_bins:
            raise ValueError("Collapsed bins")

    except Exception:
        ranks = s.rank(method="first", pct=True)

        bins = pd.cut(
            ranks,
            bins=n_bins,
            labels=labels,
            include_lowest=True
        )

    return pd.Series(bins, dtype=int)


# ============================================================
# TFRC COMPUTATION FOR ONE WINDOW
# ============================================================

def compute_tfrc_window(win: pd.DataFrame, tau_days: float) -> np.ndarray:
    n_edges = len(win)

    if n_edges == 0:
        return np.array([], dtype=np.float64)

    tau_days = max(float(tau_days), EPS)

    frm_raw = win["frm"].to_numpy()
    to_raw = win["to"].to_numpy()

    all_nodes, encoded = np.unique(
        np.concatenate([frm_raw, to_raw]),
        return_inverse=True
    )

    frm_arr = encoded[:n_edges].astype(np.int32)
    to_arr = encoded[n_edges:].astype(np.int32)
    n_nodes = len(all_nodes)

    val_arr_raw = win["value"].to_numpy(dtype=np.float64)
    val_arr_raw = np.where(val_arr_raw > 0.0, val_arr_raw, MIN_POSITIVE_WEIGHT)

    log_vals = np.log1p(val_arr_raw)

    if n_edges > 1:
        scale = float(np.quantile(log_vals, VALUE_SCALE_QUANTILE))
    else:
        scale = float(log_vals.max())

    scale = max(scale, EPS)

    val_arr = log_vals / scale
    val_arr = np.clip(
        val_arr,
        MIN_NORMALIZED_EDGE_WEIGHT,
        MAX_NORMALIZED_EDGE_WEIGHT
    )

    ts_arr_raw = win["timestamp"].to_numpy(dtype=np.float64)
    ts_arr = (ts_arr_raw - ts_arr_raw.min()) / SECONDS_PER_DAY

    order = np.arange(n_edges, dtype=np.float64)
    ts_eff = ts_arr + order * TIMESTAMP_TIE_BREAK

    sort_idx = np.argsort(ts_eff, kind="stable")

    ts_sorted = ts_eff[sort_idx]
    frm_sorted = frm_arr[sort_idx]
    to_sorted = to_arr[sort_idx]
    val_sorted = val_arr[sort_idx]

    sqrt_val = np.sqrt(np.maximum(val_arr, EPS))
    sqrt_val_sorted = np.sqrt(np.maximum(val_sorted, EPS))

    strength = (
        np.bincount(frm_arr, weights=val_arr, minlength=n_nodes) +
        np.bincount(to_arr, weights=val_arr, minlength=n_nodes)
    )

    node_weight = np.log1p(strength)
    node_weight = np.maximum(node_weight, EPS)

    S_arr = val_arr * (
        1.0 / node_weight[frm_arr] +
        1.0 / node_weight[to_arr]
    )

    out_sorted_pos: list[list[int]] = [[] for _ in range(n_nodes)]

    for pos in range(n_edges):
        out_sorted_pos[frm_sorted[pos]].append(pos)

    D_u = np.zeros(n_edges, dtype=np.float64)
    D_v = np.zeros(n_edges, dtype=np.float64)

    for i in range(n_edges):
        u = frm_arr[i]
        v = to_arr[i]

        w_e = val_arr[i]
        t_i = ts_eff[i]
        sv_e = sqrt_val[i]

        # Penalty from outgoing neighbors of u, excluding u -> v
        pos_list = out_sorted_pos[u]

        if pos_list:
            pos_arr = np.array(pos_list, dtype=np.int64)
            pos_arr = pos_arr[to_sorted[pos_arr] != v]

            if len(pos_arr) > 0:
                dt_days = np.maximum(np.abs(ts_sorted[pos_arr] - t_i), EPS)
                K = np.exp(-dt_days / tau_days)

                denom = np.maximum(sv_e * sqrt_val_sorted[pos_arr], EPS)
                inner = K / denom

                D_u[i] = w_e * float(np.sum(inner)) / float(len(pos_arr))

        # Penalty from outgoing neighbors of v, excluding v -> u
        pos_list = out_sorted_pos[v]

        if pos_list:
            pos_arr = np.array(pos_list, dtype=np.int64)
            pos_arr = pos_arr[to_sorted[pos_arr] != u]

            if len(pos_arr) > 0:
                dt_days = np.maximum(np.abs(ts_sorted[pos_arr] - t_i), EPS)
                K = np.exp(-dt_days / tau_days)

                denom = np.maximum(sv_e * sqrt_val_sorted[pos_arr], EPS)
                inner = K / denom

                D_v[i] = w_e * float(np.sum(inner)) / float(len(pos_arr))

    D_arr = 0.5 * (D_u + D_v)

    TFRC = S_arr - D_arr

    return TFRC


# ============================================================
# PROCESS ONE DATASET AND ONE TAU
# ============================================================

def process_dataset_tau(dataset: str, input_csv: Path, tau_days: float) -> None:
    tag = tau_tag(tau_days)

    out_dir = OUTPUT_ROOT / dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Dataset : {dataset}")
    print(f"Tau     : {tau_days} days")
    print(f"Input   : {input_csv}")
    print(f"Output  : {out_dir}")
    print("=" * 80)

    df_raw = pd.read_csv(input_csv)

    df_raw = df_raw.rename(columns={
        "source": "from",
        "target": "to",
        "from_address": "from",
        "to_address": "to",
        "block_timestamp": "timestamp",
        "amount": "value",
        "weight": "value",
    })

    required = {"from", "to", "timestamp", "value"}
    missing = required - set(df_raw.columns)

    if missing:
        raise ValueError(
            f"{dataset} is missing required columns: {missing}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df_raw["timestamp"] = pd.to_numeric(df_raw["timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["timestamp"])
    df_raw["timestamp"] = df_raw["timestamp"].astype(np.int64)

    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce").fillna(0.0)

    df_raw = df_raw.rename(columns={"from": "frm"})

    df_raw["_date"] = pd.to_datetime(
        df_raw["timestamp"],
        unit="s",
        errors="coerce"
    ).dt.floor("D")

    df_raw = df_raw.dropna(subset=["_date"])

    date_min = df_raw["_date"].min()
    date_max = df_raw["_date"].max()

    print(f"Edges      : {len(df_raw):,}")
    print(f"Date range : {date_min.date()} -> {date_max.date()}")

    window_td = pd.Timedelta(days=WINDOW_DAYS)
    step_td = pd.Timedelta(days=WINDOW_STEP_DAYS)

    window_starts = pd.date_range(date_min, date_max, freq=step_td)

    all_chunks: list[pd.DataFrame] = []

    for ws in tqdm(window_starts, desc=f"{dataset} | {tag}"):
        we = ws + window_td

        mask = (df_raw["_date"] >= ws) & (df_raw["_date"] < we)
        win_df = df_raw.loc[mask].reset_index(drop=True)

        if len(win_df) == 0:
            continue

        tfrc = compute_tfrc_window(
            win=win_df,
            tau_days=tau_days
        )

        chunk = win_df[["frm", "to", "timestamp", "value"]].copy()
        chunk[CURV_COL] = tfrc
        chunk["window_start"] = ws.strftime("%Y-%m-%d")

        all_chunks.append(chunk)

    if not all_chunks:
        print(f"[WARN] No windows produced output for {dataset}, tau={tau_days}")
        return

    full = pd.concat(all_chunks, ignore_index=True)
    full = full.rename(columns={"frm": "from"})

    base_stem = dataset

    full_out = out_dir / f"{base_stem}.csv"
    full.drop(columns=["window_start"]).to_csv(full_out, index=False)

    print(f"[SAVE] Full TFRC file -> {full_out}")

    curv = pd.to_numeric(full[CURV_COL], errors="coerce").fillna(0.0)
    full["bin"] = 1

    for _, grp_idx in full.groupby("window_start").groups.items():
        grp_curv = curv.loc[grp_idx].to_numpy()

        if len(grp_curv) >= N_BINS:
            b_series = equal_bins(grp_curv, N_BINS)
            b_series.index = grp_idx
            full.loc[grp_idx, "bin"] = b_series

    full["bin"] = full["bin"].fillna(1).astype(int)

    bin_medians = full.groupby("bin")[CURV_COL].median().sort_index()

    print("Median curvature per bin:")
    print(bin_medians.to_string())

    if len(bin_medians) >= 2 and not bin_medians.is_monotonic_increasing:
        print("[WARN] Bin medians are not monotonically increasing.")

    stats_rows = []
    total_edges = len(full)

    for bin_id in range(1, N_BINS + 1):
        bin_df = full[full["bin"] == bin_id].drop(
            columns=["window_start", "bin"],
            errors="ignore"
        )

        bin_out = out_dir / f"{base_stem}_bin{bin_id}.csv"
        bin_df.to_csv(bin_out, index=False)

        n = len(bin_df)
        pct = 100.0 * n / max(total_edges, 1)

        stats_rows.append({
            "dataset": dataset,
            "tau_days": tau_days,
            "bin": bin_id,
            "num_edges": n,
            "percent_of_total": round(pct, 2),
            "min_curvature": float(bin_df[CURV_COL].min()) if n else np.nan,
            "median_curvature": float(bin_df[CURV_COL].median()) if n else np.nan,
            "max_curvature": float(bin_df[CURV_COL].max()) if n else np.nan,
        })

        print(f"[SAVE] Bin {bin_id} -> {bin_out}")

    stats_df = pd.DataFrame(stats_rows)

    stats_out = out_dir / f"{base_stem}_bin_counts.csv"
    stats_df.to_csv(stats_out, index=False)

    print(f"[SAVE] Bin stats -> {stats_out}")

    plot_distribution(full, dataset, tau_days)

    print(f"[DONE] {dataset} | tau={tau_days}")
    print()


# ============================================================
# PLOT CURVATURE DISTRIBUTION
# ============================================================

def plot_distribution(full: pd.DataFrame, dataset: str, tau_days: float) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[INFO] matplotlib not found. Skipping distribution plot.")
        return

    vals = pd.to_numeric(full[CURV_COL], errors="coerce").dropna().to_numpy()

    if len(vals) == 0:
        return

    tag = tau_tag(tau_days)

    p_low = np.percentile(vals, PLOT_PERCENTILE_LOW)
    p_high = np.percentile(vals, PLOT_PERCENTILE_HIGH)

    true_min = float(np.min(vals))
    true_max = float(np.max(vals))
    mean_v = float(np.mean(vals))
    median_v = float(np.median(vals))
    std_v = float(np.std(vals))

    fig, ax = plt.subplots(figsize=(9, 5))

    _, bins_edges, patches = ax.hist(
        vals,
        bins=HIST_BINS,
        range=(p_low, p_high),
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )

    norm_bins = (
        (bins_edges[:-1] - bins_edges[:-1].min()) /
        (bins_edges[:-1].max() - bins_edges[:-1].min() + EPS)
    )

    cmap = plt.cm.coolwarm

    for patch, nb in zip(patches, norm_bins):
        patch.set_facecolor(cmap(nb))
        patch.set_alpha(0.85)

    ax.axvline(
        mean_v,
        linewidth=1.8,
        linestyle="--",
        label=f"Mean {mean_v:.4f}"
    )

    ax.axvline(
        median_v,
        linewidth=1.8,
        linestyle="-",
        label=f"Median {median_v:.4f}"
    )

    ax.set_xlabel("Temporal Forman-Ricci Curvature", fontsize=12)
    ax.set_ylabel("Number of Edges", fontsize=12)

    ax.set_title(
        f"TFRC Distribution — {dataset} | tau={tau_days} days\n"
        f"n={len(vals):,} | std={std_v:.4f} | "
        f"p{PLOT_PERCENTILE_LOW}-p{PLOT_PERCENTILE_HIGH} "
        f"[{p_low:.4f}, {p_high:.4f}] | "
        f"true [{true_min:.4f}, {true_max:.4f}]",
        fontsize=11,
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.grid(axis="x", alpha=0.2, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.6)

    plt.tight_layout()

    plot_path = PLOT_DIR / f"TFRC_{dataset}_distribution_{tag}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[SAVE] Distribution plot -> {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TFRC Tau Sensitivity Generation")
    print("=" * 80)
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Tau values  : {TAU_VALUES}")
    print("=" * 80)
    print()

    for dataset in DATASETS:
        input_csv = find_input_file(dataset)

        if input_csv is None:
            print(f"[SKIP] Could not find input CSV for dataset: {dataset}")
            continue

        for tau_days in TAU_VALUES:
            try:
                process_dataset_tau(
                    dataset=dataset,
                    input_csv=input_csv,
                    tau_days=tau_days
                )

            except Exception as e:
                print("=" * 80)
                print(f"[ERROR] Dataset={dataset} | tau={tau_days}")
                print(str(e))
                print("=" * 80)
                print()

    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()