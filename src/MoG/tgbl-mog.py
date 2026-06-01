#!/usr/bin/env python3

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "tgbl-coin"

# Windowing
WINDOW_DAYS = 7
WINDOW_STEP_DAYS = 7

# Sparsification
KEEP_RATIO = 0.20
SCORE_COL = "er_score"

# Output
OUTPUT_ROOT = os.path.join("results", "MoG_20Percent")
DATASET_ROOT = "datasets"


ER_EXACT_NODE_LIMIT = 2000

SECONDS_PER_DAY = 86400.0


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


def build_dataframe_from_tgbl_review(dataset):
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

    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
    df["value"] = df["value"].fillna(1.0)

    df["datetime_utc"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
        utc=True
    )

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
# EFFECTIVE RESISTANCE SCORING
# ============================================================

def score_effective_resistance(G: nx.DiGraph) -> dict:
    """
    Actual MoG-style Effective Resistance scoring.

    ER(u,v) = (e_u - e_v)^T L^+ (e_u - e_v)

    High ER:
        bottleneck / structurally important edge

    Low ER:
        redundant edge with many alternative paths

    For large windows, exact pseudo-inverse is expensive.
    Therefore, for windows with more than ER_EXACT_NODE_LIMIT nodes,
    the script uses the common 1 / weight fallback approximation.
    """

    H = G.to_undirected()
    H.remove_edges_from(nx.selfloop_edges(H))

    nodes = list(H.nodes())
    n_nodes = len(nodes)

    scores = {}

    if n_nodes < 3 or H.number_of_edges() == 0:
        for u, v in G.edges():
            scores[(u, v)] = 1.0
        return scores

    # Large graph fallback
    if n_nodes > ER_EXACT_NODE_LIMIT:
        for u, v, data in G.edges(data=True):
            w = data.get("value", 1.0)
            scores[(u, v)] = 1.0 / max(float(w), 1e-12)
        return scores

    # Exact Effective Resistance using Laplacian pseudo-inverse
    try:
        node_idx = {node: i for i, node in enumerate(nodes)}

        L = nx.laplacian_matrix(H, nodelist=nodes).toarray().astype(np.float64)
        L_pinv = np.linalg.pinv(L)

        for u, v in G.edges():
            if u not in node_idx or v not in node_idx:
                scores[(u, v)] = 0.0
                continue

            i = node_idx[u]
            j = node_idx[v]

            er = L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j]
            scores[(u, v)] = max(float(er), 0.0)

    except Exception as e:
        print(f"[WARN] Exact ER failed. Using fallback score=1.0. Reason: {e}")
        for u, v in G.edges():
            scores[(u, v)] = 1.0

    return scores


# ============================================================
# SPARSIFY ONE WINDOW
# ============================================================

def sparsify_window_effective_resistance(
    win_df: pd.DataFrame,
    keep_ratio: float
) -> pd.DataFrame:
    """
    Keep top keep_ratio edges by Effective Resistance score
    inside one temporal window.
    """

    if len(win_df) == 0:
        return win_df.copy()

    win_df = win_df[["from", "to", "timestamp", "value"]].copy()


    edge_agg = (
        win_df
        .groupby(["from", "to"], as_index=False)
        .agg({"value": "sum"})
    )

    G = nx.from_pandas_edgelist(
        edge_agg,
        source="from",
        target="to",
        edge_attr=["value"],
        create_using=nx.DiGraph()
    )

    scores = score_effective_resistance(G)

    scored_df = win_df.copy()
    scored_df[SCORE_COL] = scored_df.apply(
        lambda row: scores.get((row["from"], row["to"]), 0.0),
        axis=1
    )

    n_keep = max(1, int(np.ceil(len(scored_df) * keep_ratio)))

    kept_df = (
        scored_df
        .sort_values(SCORE_COL, ascending=False)
        .head(n_keep)
        .drop(columns=[SCORE_COL])
        .copy()
    )

    return kept_df


# ============================================================
# WINDOWED MoG PROCESSING
# ============================================================

def compute_windowed_mog_sparsification(
    dataset_name: str,
    graph_df: pd.DataFrame,
    output_root: str
) -> pd.DataFrame:

    date_min = graph_df["_date"].min()
    date_max = graph_df["_date"].max()

    print(f"\nWindowing from {date_min.date()} to {date_max.date()}")
    print(f"Window size = {WINDOW_DAYS} days")
    print(f"Window step = {WINDOW_STEP_DAYS} days")
    print(f"Keep ratio  = {KEEP_RATIO:.0%}")

    window_td = pd.Timedelta(days=WINDOW_DAYS)
    step_td = pd.Timedelta(days=WINDOW_STEP_DAYS)
    window_starts = pd.date_range(date_min, date_max, freq=step_td)

    kept_chunks = []
    stats_rows = []

    for ws in tqdm(window_starts, desc="MoG ER windows"):
        we = ws + window_td

        mask = (graph_df["_date"] >= ws) & (graph_df["_date"] < we)

        win_df = (
            graph_df
            .loc[mask, ["from", "to", "timestamp", "value"]]
            .copy()
            .reset_index(drop=True)
        )

        if len(win_df) == 0:
            continue

        kept_df = sparsify_window_effective_resistance(
            win_df=win_df,
            keep_ratio=KEEP_RATIO
        )

        kept_df["window_start"] = ws.strftime("%Y-%m-%d")
        kept_df["window_end"] = we.strftime("%Y-%m-%d")

        kept_chunks.append(kept_df)

        stats_rows.append({
            "dataset": dataset_name,
            "window_start": ws.strftime("%Y-%m-%d"),
            "window_end": we.strftime("%Y-%m-%d"),
            "edges_original_window": len(win_df),
            "edges_kept_window": len(kept_df),
            "percent_kept_window": round(
                100.0 * len(kept_df) / max(len(win_df), 1),
                2
            ),
            "keep_ratio_target": KEEP_RATIO,
            "criterion": "effective_resistance",
        })

    if not kept_chunks:
        print("[WARN] No windows produced output.")
        return pd.DataFrame()

    sparse_df = pd.concat(kept_chunks, ignore_index=True)
    stats_df = pd.DataFrame(stats_rows)

    sparse_graph_df = sparse_df.drop(
        columns=["window_start", "window_end"],
        errors="ignore"
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    sparse_csv = os.path.join(
        out_dir,
        f"{dataset_name}_mog_er_top20.csv"
    )

    stats_csv = os.path.join(
        out_dir,
        f"{dataset_name}_mog_er_top20_stats.csv"
    )

    sparse_with_windows_csv = os.path.join(
        out_dir,
        f"{dataset_name}_mog_er_top20_with_windows.csv"
    )

    sparse_graph_df.to_csv(sparse_csv, index=False)
    sparse_df.to_csv(sparse_with_windows_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    n_original = len(graph_df)
    n_kept = len(sparse_graph_df)
    pct_kept = 100.0 * n_kept / max(n_original, 1)

    print("\nSaved MoG 20% sparsified graph:")
    print(f"  {sparse_csv}")

    print("\nSaved MoG 20% sparsified graph with window labels:")
    print(f"  {sparse_with_windows_csv}")

    print("\nSaved stats:")
    print(f"  {stats_csv}")

    print("\nOverall sparsification summary:")
    print(f"  Original edges : {n_original:,}")
    print(f"  Kept edges     : {n_kept:,}")
    print(f"  Percent kept   : {pct_kept:.2f}%")
    print(f"  Target ratio   : {KEEP_RATIO:.0%}")

    return sparse_graph_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MoG Effective Resistance 20% Sparsification")
    print("=" * 70)

    print(f"\nLoading dataset: {DATASET_NAME}")

    dataset = PyGLinkPropPredDataset(
        name=DATASET_NAME,
        root=DATASET_ROOT
    )

    print_dataset_fields(dataset)

    df = build_dataframe_from_tgbl_review(dataset)

    print("\nPrepared dataframe preview:")
    print(df.head())

    daily_counts = inspect_timestamps(df)

    inspect_dir = os.path.join(OUTPUT_ROOT, DATASET_NAME)
    os.makedirs(inspect_dir, exist_ok=True)

    cleaned_csv = os.path.join(
        inspect_dir,
        f"{DATASET_NAME}_cleaned.csv"
    )

    preview_csv = os.path.join(
        inspect_dir,
        f"{DATASET_NAME}_timestamp_preview.csv"
    )

    daily_counts_csv = os.path.join(
        inspect_dir,
        f"{DATASET_NAME}_daily_counts.csv"
    )

    df.to_csv(cleaned_csv, index=False)

    df[
        ["from", "to", "timestamp", "datetime_utc", "_date", "value"]
    ].head(5000).to_csv(preview_csv, index=False)

    daily_counts.to_csv(daily_counts_csv, index=False)

    print(f"\nSaved cleaned dataset:")
    print(f"  {cleaned_csv}")

    print(f"\nSaved timestamp preview:")
    print(f"  {preview_csv}")

    print(f"\nSaved daily counts:")
    print(f"  {daily_counts_csv}")

    sparse_df = compute_windowed_mog_sparsification(
        dataset_name=DATASET_NAME,
        graph_df=df,
        output_root=OUTPUT_ROOT
    )

    if not sparse_df.empty:
        print("\nFinal 20% sparsified graph preview:")
        print(sparse_df.head())

    print("\nDone.")


if __name__ == "__main__":
    main()