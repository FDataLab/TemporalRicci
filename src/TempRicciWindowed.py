"""
temporal_forman_ricci_windowed.py
=================================

Windowed Temporal Forman-Ricci Curvature (TFRC) for weighted temporal graphs.

This script:
  1. Computes TFRC independently inside each non-overlapping time window
     (no data leakage across windows).
  2. Fits equal-quantile curvature bins separately inside each window.
  3. Saves:
       - full TFRC edge list
       - one CSV per bin
       - bin statistics
       - curvature distribution plot
       - runtime log for TFRC + sparsification

IMPORTANT BIN ORDER
-------------------
bin 1 = LOWEST curvature values
...
bin 5 = HIGHEST curvature values
"""

# ============================================================
# Imports
# ============================================================

import time
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

DATASETS = [
    "ADX",
    "BAG",
    "BEPRO",
    "DERC",
    "DFRC",
    "DINO",
    "ETH2X-FLI",
    "EVERMOON",
    "GLM",
    "HOICHI",
    "TGBL-COIN",
    "TGBL-REVIEW",
]

INPUT_DIR = PROJECT_ROOT / "data"

OUTPUT_ROOT = PROJECT_ROOT / "RicciResults" / "ricci_values_windowed"
PLOT_DIR = PROJECT_ROOT / "results"

RUNTIME_DIR = PROJECT_ROOT / "RicciResults" / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

RUNTIME_FILE = RUNTIME_DIR / "tfrc_sparsification_runtime.csv"

TAU_DAYS = 1.0

WINDOW_DAYS = 7
WINDOW_STEP_DAYS = 7

N_BINS = 5
CURV_COL = "Temporal Forman-Ricci value"


# ============================================================
# NUMERICAL CONSTANTS
# ============================================================

EPS = 1e-12
SECONDS_PER_DAY = 86400.0

# Weight handling
MIN_POSITIVE_WEIGHT = 1e-12
VALUE_SCALE_QUANTILE = 0.99
MIN_NORMALIZED_EDGE_WEIGHT = 0.05
MAX_NORMALIZED_EDGE_WEIGHT = 1.0

# Timestamp tie-breaking for stable ordering only
TIMESTAMP_TIE_BREAK = 1e-9

# Plot constants
HIST_BINS = 60
PLOT_PERCENTILE_LOW = 1
PLOT_PERCENTILE_HIGH = 99


# ============================================================
# RUNTIME HELPERS
# ============================================================

def ensure_runtime_header():
    """
    Create runtime CSV header if the file does not exist.
    Runtime rows are appended, so each dataset/run is preserved.
    """
    if (not RUNTIME_FILE.exists()) or RUNTIME_FILE.stat().st_size == 0:
        with open(RUNTIME_FILE, "w", encoding="utf-8") as f:
            f.write(
                "dataset,"
                "tau_days,"
                "window_days,"
                "window_step_days,"
                "n_bins,"
                "num_raw_edges,"
                "num_edge_window_records,"
                "num_windows,"
                "num_nonempty_windows,"
                "start_datetime,"
                "end_datetime,"
                "total_seconds\n"
            )


def seconds_to_min_sec(seconds: float) -> str:
    seconds = float(seconds)
    minutes = int(seconds // 60)
    sec = seconds % 60

    if minutes == 0:
        return f"{sec:.2f}s"

    return f"{minutes}m {sec:.2f}s"


def find_input_csv(dataset: str) -> Path:
    """
    Find the input CSV for a dataset.

    This checks a few common filename variants because some files may use
    lowercase names or underscores instead of hyphens.
    """
    candidates = [
        INPUT_DIR / f"{dataset}.csv",
        INPUT_DIR / f"{dataset.upper()}.csv",
        INPUT_DIR / f"{dataset.lower()}.csv",
        INPUT_DIR / f"{dataset.replace('-', '_')}.csv",
        INPUT_DIR / f"{dataset.replace('-', '_').upper()}.csv",
        INPUT_DIR / f"{dataset.replace('-', '_').lower()}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Input CSV not found for dataset "
        f"{dataset}. Tried:\n" +
        "\n".join(str(p) for p in candidates)
    )


# ============================================================
# SECTION 1 — TFRC FOR ONE WINDOW
# ============================================================

def compute_tfrc_window(
    win: pd.DataFrame,
    tau_days: float,
) -> np.ndarray:
    """
    Compute Temporal Forman-Ricci Curvature (TFRC) for each edge in one window.

    For edge e=(u,v):

        TFRC(e) = S_e - D_e

    Node-support term:
        S_e = w_e * ( 1/s(u) + 1/s(v) )

    where:
        strength(x) = sum of weights of edges incident to x
        s(x) = log(1 + strength(x))

    Temporal parallel-edge penalty:
        D_e = ( D_u(e) + D_v(e) ) / 2

        D_u(e) = (w_e / |N_u(e)|) * sum_{e' in N_u(e)} K(e,e') / sqrt(w_e * w_e')
        D_v(e) = (w_e / |N_v(e)|) * sum_{e' in N_v(e)} K(e,e') / sqrt(w_e * w_e')

    with:
        K(e,e') = exp( - |t_e - t_e'| / tau_days )

    Notes
    -----
    - Outgoing edges only are used for N_u(e) and N_v(e).
    - If a neighborhood is empty, its contribution is set to 0.
    - Edge weights are robustly normalized per window for stability.
    """
    n_edges = len(win)

    if n_edges == 0:
        return np.array([], dtype=np.float64)

    tau_days = max(float(tau_days), EPS)

    # ----------------------------------------------------------
    # 1. Integer-encode node IDs
    # ----------------------------------------------------------
    frm_raw = win["frm"].to_numpy()
    to_raw = win["to"].to_numpy()

    all_nodes, encoded = np.unique(
        np.concatenate([frm_raw, to_raw]),
        return_inverse=True
    )

    frm_arr = encoded[:n_edges].astype(np.int32)
    to_arr = encoded[n_edges:].astype(np.int32)
    n_nodes = len(all_nodes)

    # ----------------------------------------------------------
    # 2. Robust per-window edge-weight normalization
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 3. Relative timestamps in days within the window
    # ----------------------------------------------------------
    ts_arr_raw = win["timestamp"].to_numpy(dtype=np.float64)
    ts_arr = (ts_arr_raw - ts_arr_raw.min()) / SECONDS_PER_DAY

    order = np.arange(n_edges, dtype=np.float64)
    ts_eff = ts_arr + order * TIMESTAMP_TIE_BREAK

    # ----------------------------------------------------------
    # 4. Sort by effective time
    # ----------------------------------------------------------
    sort_idx = np.argsort(ts_eff, kind="stable")

    ts_sorted = ts_eff[sort_idx]
    frm_sorted = frm_arr[sort_idx]
    to_sorted = to_arr[sort_idx]
    val_sorted = val_arr[sort_idx]

    sqrt_val = np.sqrt(np.maximum(val_arr, EPS))
    sqrt_val_sorted = np.sqrt(np.maximum(val_sorted, EPS))

    # ----------------------------------------------------------
    # 5. Node strengths and smoothed node weights
    # ----------------------------------------------------------
    strength = (
        np.bincount(frm_arr, weights=val_arr, minlength=n_nodes) +
        np.bincount(to_arr, weights=val_arr, minlength=n_nodes)
    )

    node_weight = np.log1p(strength)
    node_weight = np.maximum(node_weight, EPS)

    # ----------------------------------------------------------
    # 6. Forman-like node-support term
    # ----------------------------------------------------------
    S_arr = val_arr * (
        1.0 / node_weight[frm_arr] +
        1.0 / node_weight[to_arr]
    )

    # ----------------------------------------------------------
    # 7. Build outgoing adjacency in sorted-position space
    # ----------------------------------------------------------
    out_sorted_pos: list[list[int]] = [[] for _ in range(n_nodes)]

    for pos in range(n_edges):
        out_sorted_pos[frm_sorted[pos]].append(pos)

    # ----------------------------------------------------------
    # 8. Main loop — temporal parallel-edge penalties
    # ----------------------------------------------------------
    D_u = np.zeros(n_edges, dtype=np.float64)
    D_v = np.zeros(n_edges, dtype=np.float64)

    for i in range(n_edges):
        u = frm_arr[i]
        v = to_arr[i]
        w_e = val_arr[i]
        t_i = ts_eff[i]
        sv_e = sqrt_val[i]

        # -------------------------
        # D_u(e): outgoing neighbors of u excluding target v
        # -------------------------
        pos_list = out_sorted_pos[u]

        if pos_list:
            pos_arr = np.array(pos_list, dtype=np.int64)

            # Exclude edges pointing to v
            pos_arr = pos_arr[to_sorted[pos_arr] != v]

            if len(pos_arr) > 0:
                dt_days = np.maximum(np.abs(ts_sorted[pos_arr] - t_i), EPS)
                K = np.exp(-dt_days / tau_days)

                denom = np.maximum(sv_e * sqrt_val_sorted[pos_arr], EPS)
                inner = K / denom

                D_u[i] = w_e * float(np.sum(inner)) / float(len(pos_arr))

        # -------------------------
        # D_v(e): outgoing neighbors of v excluding target u
        # -------------------------
        pos_list = out_sorted_pos[v]

        if pos_list:
            pos_arr = np.array(pos_list, dtype=np.int64)

            # Exclude edges pointing to u
            pos_arr = pos_arr[to_sorted[pos_arr] != u]

            if len(pos_arr) > 0:
                dt_days = np.maximum(np.abs(ts_sorted[pos_arr] - t_i), EPS)
                K = np.exp(-dt_days / tau_days)

                denom = np.maximum(sv_e * sqrt_val_sorted[pos_arr], EPS)
                inner = K / denom

                D_v[i] = w_e * float(np.sum(inner)) / float(len(pos_arr))

    # ----------------------------------------------------------
    # 9. Overall temporal parallel-edge penalty
    # ----------------------------------------------------------
    D_arr = 0.5 * (D_u + D_v)

    # ----------------------------------------------------------
    # 10. Final TFRC
    # ----------------------------------------------------------
    TFRC = S_arr - D_arr

    return TFRC


# ============================================================
# SECTION 2 — EQUAL QUANTILE BINNING
# ============================================================

def equal_bins(values: np.ndarray, n_bins: int = 5) -> pd.Series:
    """
    Assign equal-quantile bins in ASCENDING curvature order.

      bin 1 = lowest curvature values
      ...
      bin n = highest curvature values
    """
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
# SECTION 3 — MAIN PIPELINE FOR ONE DATASET
# ============================================================

def main(dataset: str):
    ensure_runtime_header()

    start_unix = time.time()
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_unix))

    tau_days = TAU_DAYS
    window_days = WINDOW_DAYS
    step_days = WINDOW_STEP_DAYS
    n_bins = N_BINS

    input_csv = find_input_csv(dataset)

    param_tag = f"tau{tau_days:.4f}".replace(".", "p")

    out_dir = OUTPUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Windowed Temporal Forman-Ricci Curvature (TFRC)")
    print("=" * 64)
    print(f"  Dataset            : {dataset}")
    print(f"  Input CSV          : {input_csv}")
    print(f"  Window             : {window_days} days (step {step_days} days)")
    print(f"  Tau (days)         : {tau_days}")
    print(f"  Output dir         : {out_dir}")
    print(f"  Runtime file       : {RUNTIME_FILE}")
    print()

    # ----------------------------------------------------------
    # Load raw CSV
    # ----------------------------------------------------------
    print("[1/4] Loading raw edge CSV ...")

    df_raw = pd.read_csv(input_csv)

    df_raw = df_raw.rename(columns={"source": "from", "target": "to"})

    required = {"from", "to", "timestamp", "value"}
    missing = required - set(df_raw.columns)

    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    df_raw["timestamp"] = pd.to_numeric(df_raw["timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["timestamp"])
    df_raw["timestamp"] = df_raw["timestamp"].astype(np.int64)

    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce").fillna(0.0)

    # Internal rename
    df_raw = df_raw.rename(columns={"from": "frm"})
    df_raw["_date"] = pd.to_datetime(df_raw["timestamp"], unit="s").dt.floor("D")

    date_min = df_raw["_date"].min()
    date_max = df_raw["_date"].max()

    if pd.isna(date_min) or pd.isna(date_max):
        raise ValueError("No valid timestamps found after cleaning.")

    print(f"       Edges         : {len(df_raw):,}")
    print(f"       Date range    : {date_min.date()} -> {date_max.date()}")
    print(f"       Unique sources: {df_raw['frm'].nunique():,}")
    print(f"       Unique targets: {df_raw['to'].nunique():,}")

    # ----------------------------------------------------------
    # Compute per-window TFRC
    # ----------------------------------------------------------
    print("\n[2/4] Computing per-window TFRC ...")

    window_td = pd.Timedelta(days=window_days)
    step_td = pd.Timedelta(days=step_days)
    window_starts = pd.date_range(date_min, date_max, freq=step_td)

    all_chunks: list[pd.DataFrame] = []
    window_sizes = []
    nonempty_windows = 0

    for ws in tqdm(window_starts, desc=f"Windows ({dataset})"):
        we = ws + window_td

        mask = (df_raw["_date"] >= ws) & (df_raw["_date"] < we)
        win_df = df_raw.loc[mask].reset_index(drop=True)

        window_sizes.append(len(win_df))

        if len(win_df) == 0:
            continue

        nonempty_windows += 1

        tfrc = compute_tfrc_window(
            win=win_df,
            tau_days=tau_days,
        )

        chunk = win_df[["frm", "to", "timestamp", "value"]].copy()
        chunk[CURV_COL] = tfrc
        chunk["window_start"] = ws.strftime("%Y-%m-%d")

        all_chunks.append(chunk)

    window_sizes_arr = np.array(window_sizes, dtype=np.int64)

    print("\n[DEBUG] Window size summary")
    print(f"       Total windows              : {len(window_sizes_arr):,}")
    print(f"       Non-empty windows          : {(window_sizes_arr > 0).sum():,}")

    if len(window_sizes_arr) > 0:
        print(f"       Min edges/window           : {window_sizes_arr.min():,}")
        print(f"       Median edges/window        : {int(np.median(window_sizes_arr)):,}")
        print(f"       Mean edges/window          : {window_sizes_arr.mean():.1f}")
        print(f"       Max edges/window           : {window_sizes_arr.max():,}")
        print(f"       Total edge-window records  : {window_sizes_arr.sum():,}")

    if not all_chunks:
        print("[WARN] No windows produced output. Exiting.")

        end_unix = time.time()
        end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_unix))
        total_seconds = end_unix - start_unix

        with open(RUNTIME_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{dataset},"
                f"{tau_days},"
                f"{window_days},"
                f"{step_days},"
                f"{n_bins},"
                f"{len(df_raw)},"
                f"0,"
                f"{len(window_starts)},"
                f"0,"
                f"{start_datetime},"
                f"{end_datetime},"
                f"{total_seconds:.2f}\n"
            )

        print(f"Runtime saved to: {RUNTIME_FILE}")
        print(f"Total TFRC + sparsification runtime: {seconds_to_min_sec(total_seconds)}")
        return

    full = pd.concat(all_chunks, ignore_index=True)
    full = full.rename(columns={"frm": "from"})

    print(f"       Total edge-window records: {len(full):,}")
    print(f"       Non-empty windows        : {nonempty_windows:,}")

    # ----------------------------------------------------------
    # Save full CSV
    # ----------------------------------------------------------
    print("\n[3/4] Saving full TFRC CSV ...")

    base_stem = f"{dataset}"
    full_out = out_dir / f"{base_stem}.csv"

    full.drop(columns=["window_start"]).to_csv(full_out, index=False)

    print(f"       -> {full_out}")

    # ----------------------------------------------------------
    # Per-window equal-quantile binning
    # ----------------------------------------------------------
    print(f"\n[4/4] Equal {n_bins}-quantile binning (per window) ...")

    curv = pd.to_numeric(full[CURV_COL], errors="coerce").fillna(0.0)

    full["bin"] = 1

    for _, grp_idx in full.groupby("window_start").groups.items():
        grp_curv = curv.loc[grp_idx].to_numpy()

        if len(grp_curv) >= n_bins:
            b_series = equal_bins(grp_curv, n_bins)
            b_series.index = grp_idx
            full.loc[grp_idx, "bin"] = b_series

    full["bin"] = full["bin"].fillna(1).astype(int)

    # Sanity check
    print("\nMedian curvature per bin (should increase from bin 1 to bin 5):")

    bin_medians_check = full.groupby("bin")[CURV_COL].median().sort_index()

    print(bin_medians_check.to_string())

    if len(bin_medians_check) >= 2 and not bin_medians_check.is_monotonic_increasing:
        print("[WARN] Bin medians are not monotonically increasing. Check repeated values / collapsed quantiles.")

    # ----------------------------------------------------------
    # Save per-bin CSVs and stats
    # ----------------------------------------------------------
    drop_cols = ["window_start", "bin"]
    stats_rows = []
    total_edges = len(full)

    for bin_id in range(1, n_bins + 1):
        bin_df = full[full["bin"] == bin_id].drop(columns=drop_cols, errors="ignore")

        bin_out = out_dir / f"{base_stem}_bin{bin_id}.csv"
        bin_df.to_csv(bin_out, index=False)

        n = len(bin_df)
        pct = 100.0 * n / max(total_edges, 1)

        stats_rows.append({
            "bin": bin_id,
            "num_edges": n,
            "percent_of_total": round(pct, 2),
            "min_curvature": float(bin_df[CURV_COL].min()) if n else float("nan"),
            "median_curvature": float(bin_df[CURV_COL].median()) if n else float("nan"),
            "max_curvature": float(bin_df[CURV_COL].max()) if n else float("nan"),
        })

    stats_df = pd.DataFrame(stats_rows)

    stats_out = out_dir / f"{base_stem}_bin_counts.csv"
    stats_df.to_csv(stats_out, index=False)

    # ----------------------------------------------------------
    # Plot distribution
    # ----------------------------------------------------------
    plot_path = None

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        vals = pd.to_numeric(full[CURV_COL], errors="coerce").dropna().to_numpy()

        if len(vals) > 0:
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
                color="#4C9BE8",
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
                color="#E84C4C",
                linewidth=1.8,
                linestyle="--",
                label=f"Mean   {mean_v:.4f}"
            )

            ax.axvline(
                median_v,
                color="#2ECC71",
                linewidth=1.8,
                linestyle="-",
                label=f"Median {median_v:.4f}"
            )

            ax.set_xlabel("Temporal Forman-Ricci Curvature", fontsize=12)
            ax.set_ylabel("Number of Edges", fontsize=12)

            ax.set_title(
                f"TFRC Distribution — {dataset} (tau={tau_days} days)\n"
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

            plot_path = PLOT_DIR / f"TFRC_{dataset}_distribution_{param_tag}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"\n  Distribution plot -> {plot_path}")

    except ImportError:
        print("\n  [INFO] matplotlib not found — skipping plot.")

    # ----------------------------------------------------------
    # Save runtime
    # ----------------------------------------------------------
    end_unix = time.time()
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_unix))
    total_seconds = end_unix - start_unix

    with open(RUNTIME_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{dataset},"
            f"{tau_days},"
            f"{window_days},"
            f"{step_days},"
            f"{n_bins},"
            f"{len(df_raw)},"
            f"{len(full)},"
            f"{len(window_starts)},"
            f"{nonempty_windows},"
            f"{start_datetime},"
            f"{end_datetime},"
            f"{total_seconds:.2f}\n"
        )

    print()
    print(f"Runtime saved to: {RUNTIME_FILE}")
    print(f"Total TFRC + sparsification runtime: {seconds_to_min_sec(total_seconds)}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print()
    print(stats_df[["bin", "num_edges", "percent_of_total"]].to_string(index=False))
    print()
    print("=" * 64)
    print("DONE")
    print(f"  Full CSV  : {full_out}")
    print(f"  Bin CSVs  : {out_dir}/{base_stem}_bin1..{n_bins}.csv")
    print(f"  Bin stats : {stats_out}")

    if plot_path is not None:
        print(f"  Plot      : {plot_path}")

    print(f"  Runtime   : {RUNTIME_FILE}")
    print("=" * 64)


# ============================================================
# RUN ALL DATASETS
# ============================================================

if __name__ == "__main__":
    ensure_runtime_header()

    for dataset_name in DATASETS:
        print("\n\n")
        print("#" * 80)
        print(f"Running TFRC for dataset: {dataset_name}")
        print("#" * 80)

        try:
            main(dataset_name)
        except Exception as e:
            print(f"[ERROR] Failed for dataset {dataset_name}: {e}")