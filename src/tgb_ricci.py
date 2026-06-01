#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "tgbl-comment"

TAU_DAYS = 1.0

WINDOW_DAYS = 7
WINDOW_STEP_DAYS = 7

N_BINS = 5

OUTPUT_ROOT = os.path.join("results", "Ricci Values")

EPS = 1e-12
SECONDS_PER_DAY = 86400.0
VALUE_SCALE_QUANTILE = 0.99
MIN_POSITIVE_WEIGHT = 1e-12
MIN_NORMALIZED_EDGE_WEIGHT = 0.05
MAX_NORMALIZED_EDGE_WEIGHT = 1.0
TIMESTAMP_TIE_BREAK = 1e-9

HIST_BINS = 60
PLOT_PERCENTILE_LOW = 1
PLOT_PERCENTILE_HIGH = 99

CURV_COL = "Temporal Forman-Ricci value"


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_ricci_distribution(ricci_values, output_png):
    vals = pd.to_numeric(pd.Series(ricci_values), errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return

    p_low = np.percentile(vals, PLOT_PERCENTILE_LOW)
    p_high = np.percentile(vals, PLOT_PERCENTILE_HIGH)

    plt.figure(figsize=(7, 4.5))
    plt.hist(
        vals,
        bins=HIST_BINS,
        color="lightblue",
        edgecolor="black",
        range=(p_low, p_high)
    )
    plt.xlabel("Temporal Forman-Ricci Curvature")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


# ============================================================
# DATASET HELPERS
# ============================================================

def to_numpy_safe(x):
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def print_dataset_fields(dataset):
    print("\nAvailable attributes in dataset:")
    attrs = [attr for attr in dir(dataset) if not attr.startswith("_")]
    print(attrs)

    print("\nUseful dataset fields and shapes:")
    candidate_fields = [
        "src",
        "dst",
        "ts",
        "edge_feat",
        "num_edges",
        "num_nodes",
    ]

    for field in candidate_fields:
        if hasattr(dataset, field):
            value = getattr(dataset, field)
            try:
                arr = to_numpy_safe(value)
                print(f"{field}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception:
                print(f"{field}: {type(value)}")


def extract_edge_weight_from_pyg(dataset, n_edges):
    if not hasattr(dataset, "edge_feat"):
        print("[INFO] No edge_feat found. Using all-ones weights.")
        return np.ones(n_edges, dtype=float)

    try:
        edge_feat = to_numpy_safe(dataset.edge_feat)
    except Exception:
        print("[INFO] Could not read edge_feat. Using all-ones weights.")
        return np.ones(n_edges, dtype=float)

    if edge_feat is None:
        print("[INFO] edge_feat is None. Using all-ones weights.")
        return np.ones(n_edges, dtype=float)

    edge_feat = np.asarray(edge_feat)

    if edge_feat.ndim == 1:
        return pd.to_numeric(edge_feat, errors="coerce").astype(float)

    if edge_feat.ndim == 2:
        if edge_feat.shape[1] == 0:
            return np.ones(n_edges, dtype=float)
        return pd.to_numeric(edge_feat[:, 0], errors="coerce").astype(float)

    print(f"[WARN] Unexpected edge_feat shape: {edge_feat.shape}. Using all-ones weights.")
    return np.ones(n_edges, dtype=float)


def build_dataframe_from_tgbl_coin(dataset):
    src = to_numpy_safe(dataset.src)
    dst = to_numpy_safe(dataset.dst)
    ts = to_numpy_safe(dataset.ts)

    src = np.asarray(src)
    dst = np.asarray(dst)
    ts = np.asarray(ts)

    n_edges = len(src)
    if len(dst) != n_edges or len(ts) != n_edges:
        raise ValueError(
            f"Mismatched lengths: src={len(src)}, dst={len(dst)}, ts={len(ts)}"
        )

    values = extract_edge_weight_from_pyg(dataset, n_edges)

    if len(values) != n_edges:
        raise ValueError(
            f"Weight length mismatch: weights={len(values)}, edges={n_edges}"
        )

    df = pd.DataFrame({
        "from": src.astype(str),
        "to": dst.astype(str),
        "timestamp": pd.to_numeric(ts, errors="coerce"),
        "value": pd.to_numeric(values, errors="coerce"),
    })

    df = df.dropna(subset=["from", "to", "timestamp", "value"]).copy()
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["value"] = df["value"].astype(float)

    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)
    df = df.dropna(subset=["datetime_utc"]).copy()
    df["_date"] = df["datetime_utc"].dt.floor("D")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inspect_timestamps(df):
    print("\nDataFrame columns:")
    print(df.columns.tolist())

    print("\nDataFrame dtypes:")
    print(df.dtypes)

    print(f"\nTotal cleaned edges: {len(df):,}")

    print("\nRaw timestamp statistics:")
    print("Min timestamp:", df["timestamp"].min())
    print("Max timestamp:", df["timestamp"].max())
    print("Number of unique timestamps:", df["timestamp"].nunique())

    print("\nConverted datetime preview:")
    print(df[["timestamp", "datetime_utc", "_date"]].head(20))

    print("\nConverted datetime tail:")
    print(df[["timestamp", "datetime_utc", "_date"]].tail(20))

    print("\nDate range after conversion:")
    print("First datetime:", df["datetime_utc"].min())
    print("Last datetime :", df["datetime_utc"].max())

    daily_counts = (
        df["_date"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    daily_counts.columns = ["date", "num_edges"]

    print("\nDaily edge counts:")
    print(daily_counts.head(20))
    print("...")
    print(daily_counts.tail(20))

    return daily_counts


# ============================================================
# TFRC HELPERS
# ============================================================

def equal_bins(values: np.ndarray, n_bins: int = 5) -> pd.Series:
    """
    Ascending quantile bins:
      bin 1 = lowest curvature
      ...
      bin n = highest curvature
    """
    s = pd.Series(values, dtype="float64")
    labels = list(range(1, n_bins + 1))

    try:
        bins = pd.qcut(s, q=n_bins, labels=labels, duplicates="drop")
        if pd.Series(bins).nunique() < n_bins:
            raise ValueError("Collapsed bins")
    except Exception:
        ranks = s.rank(method="first", pct=True)
        bins = pd.cut(ranks, bins=n_bins, labels=labels, include_lowest=True)

    return pd.Series(bins, dtype=int)


def compute_tfrc_for_window(window_df: pd.DataFrame, tau_days: float) -> pd.DataFrame:
    """
    Compute TFRC for one window.

    Input columns required:
      from, to, timestamp, value

    Output columns:
      from, to, timestamp, value, Temporal Forman-Ricci value
    """
    if len(window_df) == 0:
        return pd.DataFrame(columns=[
            "from", "to", "timestamp", "value",
            CURV_COL
        ])

    win = window_df[["from", "to", "timestamp", "value"]].copy()
    n_edges = len(win)

    # ----------------------------------------------------------
    # 1. Encode nodes
    # ----------------------------------------------------------
    frm_raw = win["from"].to_numpy()
    to_raw = win["to"].to_numpy()

    all_nodes, encoded = np.unique(
        np.concatenate([frm_raw, to_raw]),
        return_inverse=True
    )
    frm_arr = encoded[:n_edges].astype(np.int32)
    to_arr = encoded[n_edges:].astype(np.int32)
    n_nodes = len(all_nodes)

    # ----------------------------------------------------------
    # 2. Weight normalization
    # ----------------------------------------------------------
    val_arr_raw = win["value"].to_numpy(dtype=np.float64)
    val_arr_raw = np.where(val_arr_raw > 0.0, val_arr_raw, MIN_POSITIVE_WEIGHT)

    log_vals = np.log1p(val_arr_raw)
    scale = float(np.quantile(log_vals, VALUE_SCALE_QUANTILE)) if n_edges > 1 else float(log_vals.max())
    scale = max(scale, EPS)

    val_arr = log_vals / scale
    val_arr = np.clip(val_arr, MIN_NORMALIZED_EDGE_WEIGHT, MAX_NORMALIZED_EDGE_WEIGHT)

    # ----------------------------------------------------------
    # 3. Relative timestamps in days
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
    # 6. Node-support term
    # ----------------------------------------------------------
    S_arr = val_arr * (
        1.0 / node_weight[frm_arr] +
        1.0 / node_weight[to_arr]
    )

    # ----------------------------------------------------------
    # 7. Outgoing adjacency in sorted-position space
    # ----------------------------------------------------------
    out_sorted_pos = [[] for _ in range(n_nodes)]
    for pos in range(n_edges):
        out_sorted_pos[frm_sorted[pos]].append(pos)

    # ----------------------------------------------------------
    # 8. Temporal penalties
    # ----------------------------------------------------------
    D_u = np.zeros(n_edges, dtype=np.float64)
    D_v = np.zeros(n_edges, dtype=np.float64)

    tau_days = max(float(tau_days), EPS)

    for i in range(n_edges):
        u = frm_arr[i]
        v = to_arr[i]
        w_e = val_arr[i]
        t_i = ts_eff[i]
        sv_e = sqrt_val[i]

        # D_u(e)
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

        # D_v(e)
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
    tfrc = S_arr - D_arr

    out_df = win.copy()
    out_df[CURV_COL] = tfrc
    return out_df


# ============================================================
# WINDOWED PROCESSING
# ============================================================

def compute_windowed_tfrc(
    dataset_name,
    graph_df,
    output_root="results"
):
    date_min = graph_df["_date"].min()
    date_max = graph_df["_date"].max()

    print(f"\nWindowing from {date_min.date()} to {date_max.date()}")
    print(f"Window size = {WINDOW_DAYS} days, step = {WINDOW_STEP_DAYS} days")

    window_td = pd.Timedelta(days=WINDOW_DAYS)
    step_td = pd.Timedelta(days=WINDOW_STEP_DAYS)
    window_starts = pd.date_range(date_min, date_max, freq=step_td)

    all_chunks = []

    for ws in window_starts:
        we = ws + window_td
        mask = (graph_df["_date"] >= ws) & (graph_df["_date"] < we)
        win_df = graph_df.loc[mask, ["from", "to", "timestamp", "value"]].copy()

        if len(win_df) == 0:
            continue

        print(f"[WINDOW] {ws.date()} -> {we.date()} : {len(win_df):,} edges")

        curv_df = compute_tfrc_for_window(win_df, tau_days=TAU_DAYS)
        curv_df["window_start"] = ws.strftime("%Y-%m-%d")
        all_chunks.append(curv_df)

    if not all_chunks:
        print("[WARN] No windows produced output.")
        return pd.DataFrame()

    full_df = pd.concat(all_chunks, ignore_index=True)

    # ----------------------------------------------------------
    # Per-window binning
    # ----------------------------------------------------------
    full_df["bin"] = 1
    curv = pd.to_numeric(full_df[CURV_COL], errors="coerce").fillna(0.0)

    for _, grp_idx in full_df.groupby("window_start").groups.items():
        grp_curv = curv.loc[grp_idx].to_numpy()
        if len(grp_curv) >= N_BINS:
            b_series = equal_bins(grp_curv, n_bins=N_BINS)
            b_series.index = grp_idx
            full_df.loc[grp_idx, "bin"] = b_series

    full_df["bin"] = full_df["bin"].fillna(1).astype(int)

    # ----------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------
    out_dir = os.path.join(output_root, "Ricci Values", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    full_csv = os.path.join(out_dir, f"{dataset_name}_full.csv")
    full_df.drop(columns=["window_start"]).to_csv(full_csv, index=False)

    for bin_id in range(1, N_BINS + 1):
        bin_df = full_df[full_df["bin"] == bin_id].drop(columns=["window_start"], errors="ignore")
        bin_csv = os.path.join(out_dir, f"{dataset_name}_bin{bin_id}.csv")
        bin_df.to_csv(bin_csv, index=False)

    stats_rows = []
    total_edges = len(full_df)

    for bin_id in range(1, N_BINS + 1):
        bin_df = full_df[full_df["bin"] == bin_id]
        n = len(bin_df)
        pct = 100.0 * n / max(total_edges, 1)

        stats_rows.append({
            "bin": bin_id,
            "num_edges": n,
            "percent_of_total": round(pct, 2),
            "min_curvature": float(bin_df[CURV_COL].min()) if n else np.nan,
            "median_curvature": float(bin_df[CURV_COL].median()) if n else np.nan,
            "max_curvature": float(bin_df[CURV_COL].max()) if n else np.nan,
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(out_dir, f"{dataset_name}_bin_stats.csv")
    stats_df.to_csv(stats_csv, index=False)

    plot_png = os.path.join(out_dir, f"{dataset_name}_distribution.png")
    visualize_ricci_distribution(full_df[CURV_COL], plot_png)

    print(f"\nSaved full TFRC CSV: {full_csv}")
    print(f"Saved bin stats CSV: {stats_csv}")
    print(f"Saved distribution plot: {plot_png}")

    return full_df


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading dataset: {DATASET_NAME}")

    dataset = PyGLinkPropPredDataset(
        name=DATASET_NAME,
        root="datasets"
    )

    print_dataset_fields(dataset)

    df = build_dataframe_from_tgbl_coin(dataset)

    print("\nPrepared dataframe preview:")
    print(df.head())

    daily_counts = inspect_timestamps(df)

    inspect_dir = os.path.join(OUTPUT_ROOT, DATASET_NAME)
    os.makedirs(inspect_dir, exist_ok=True)

    cleaned_csv = os.path.join(inspect_dir, f"{DATASET_NAME}_cleaned.csv")
    preview_csv = os.path.join(inspect_dir, f"{DATASET_NAME}_timestamp_preview.csv")
    daily_counts_csv = os.path.join(inspect_dir, f"{DATASET_NAME}_daily_counts.csv")

    df.to_csv(cleaned_csv, index=False)
    df[["from", "to", "timestamp", "datetime_utc", "_date", "value"]].head(5000).to_csv(
        preview_csv, index=False
    )
    daily_counts.to_csv(daily_counts_csv, index=False)

    print(f"\nSaved cleaned dataset: {cleaned_csv}")
    print(f"Saved preview CSV: {preview_csv}")
    print(f"Saved daily counts CSV: {daily_counts_csv}")

    full_tfrc_df = compute_windowed_tfrc(
        dataset_name=DATASET_NAME,
        graph_df=df,
        output_root="results"
    )

    if not full_tfrc_df.empty:
        print("\nCombined windowed TFRC output preview:")
        print(full_tfrc_df.head())

    print("\nDone.")


if __name__ == "__main__":
    main()