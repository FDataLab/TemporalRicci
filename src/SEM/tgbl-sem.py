#!/usr/bin/env python3

"""
SEM_sparsifier_tgbl_review.py

Loads tgbl-review directly from TGB, applies SEM Balanced Forman Curvature
per 7-day non-overlapping window, and saves the final 20% sparsified graph.

Output:
C:/Users/azadp/PycharmProjects/TemporalRicci/sem_sparsified/tgbl-review_sem_bfc.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "tgbl-wiki"

DATASET_ROOT = "datasets"

OUTPUT_DIR = Path(r"C:\Users\azadp\PycharmProjects\TemporalRicci\sem_sparsified")

WINDOW_DAYS = 7
WINDOW_STEP = 7
KEEP_RATIO = 0.20
SCORE_COL = "bfc_score"

BFC_EXACT_NODE_LIMIT = 3000


# ============================================================
# TGB HELPERS
# ============================================================

def to_numpy_safe(x):
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


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
    src = np.asarray(to_numpy_safe(dataset.src))
    dst = np.asarray(to_numpy_safe(dataset.dst))
    ts = np.asarray(to_numpy_safe(dataset.ts))

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
    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["_date"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
        utc=True
    ).dt.floor("D")

    df = df.dropna(subset=["_date"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# BALANCED FORMAN CURVATURE SCORING
# ============================================================

def _count_triangles_and_squares(H: nx.Graph):
    tri = {}
    sq_u = {}
    sq_v = {}
    gamma = {}

    nbrs = {node: set(H.neighbors(node)) for node in H.nodes()}

    for u, v in H.edges():
        key = (u, v)

        Nu = nbrs[u] - {v}
        Nv = nbrs[v] - {u}

        common = Nu & Nv
        t = len(common)
        tri[key] = t

        count_u = 0
        path_nodes_u = []

        for w in Nu:
            Nw = nbrs[w] - {u}
            qs = (Nw & Nv) - Nu
            count_u += len(qs)
            path_nodes_u.extend(qs)

        sq_u[key] = count_u

        count_v = 0
        path_nodes_v = []

        for w in Nv:
            Nw = nbrs[w] - {v}
            qs = (Nw & Nu) - Nv
            count_v += len(qs)
            path_nodes_v.extend(qs)

        sq_v[key] = count_v

        all_path_nodes = path_nodes_u + path_nodes_v

        if all_path_nodes:
            from collections import Counter
            freq = Counter(all_path_nodes)
            gamma_max = max(freq.values())
        else:
            gamma_max = 1

        gamma[key] = 1.0 / max(gamma_max, 1)

    return tri, sq_u, sq_v, gamma


def score_balanced_forman_curvature(G: nx.DiGraph) -> dict:
    H = G.to_undirected()
    H.remove_edges_from(nx.selfloop_edges(H))

    nodes = list(H.nodes())
    n = len(nodes)
    scores = {}

    if n < 3 or H.number_of_edges() == 0:
        for u, v in G.edges():
            scores[(u, v)] = 0.0
        return scores

    if n > BFC_EXACT_NODE_LIMIT:
        deg = dict(H.degree())

        for u, v in G.edges():
            du = deg.get(u, 1)
            dv = deg.get(v, 1)
            scores[(u, v)] = (1.0 / max(du, 1)) + (1.0 / max(dv, 1)) - 2.0

        return scores

    try:
        deg = dict(H.degree())
        tri, sq_u_map, sq_v_map, gamma_map = _count_triangles_and_squares(H)

        for u, v in G.edges():
            key = (u, v) if (u, v) in tri else (v, u)

            du = max(deg.get(u, 1), 1)
            dv = max(deg.get(v, 1), 1)

            d_max = max(du, dv)
            d_min = min(du, dv)

            t = tri.get(key, 0)
            sq_u_val = sq_u_map.get(key, 0)
            sq_v_val = sq_v_map.get(key, 0)
            gam = gamma_map.get(key, 1.0)

            ric = (
                1.0 / du + 1.0 / dv
                - 2.0
                + 2.0 * t / d_max
                + t / d_min
                + gam / d_max * (sq_u_val + sq_v_val)
            )

            scores[(u, v)] = ric

    except Exception:
        deg = dict(H.degree())

        for u, v in G.edges():
            du = deg.get(u, 1)
            dv = deg.get(v, 1)
            scores[(u, v)] = (1.0 / max(du, 1)) + (1.0 / max(dv, 1)) - 2.0

    return scores


# ============================================================
# SPARSIFY ONE WINDOW
# ============================================================

def sparsify_window(win_df: pd.DataFrame, keep_ratio: float) -> pd.DataFrame:
    if len(win_df) == 0:
        return win_df.copy()

    G = nx.from_pandas_edgelist(
        win_df,
        source="from",
        target="to",
        edge_attr=["value"],
        create_using=nx.DiGraph(),
    )

    scores = score_balanced_forman_curvature(G)

    win_scored = win_df.copy()
    win_scored[SCORE_COL] = win_df.apply(
        lambda row: scores.get((row["from"], row["to"]), 0.0),
        axis=1
    )

    n_keep = max(1, int(np.ceil(len(win_scored) * keep_ratio)))

    kept = (
        win_scored
        .nlargest(n_keep, SCORE_COL)
        .drop(columns=[SCORE_COL])
    )

    return kept


# ============================================================
# PROCESS TGBL-REVIEW
# ============================================================

def process_tgbl_review():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("SEM Baseline -- Balanced Forman Curvature Sparsifier")
    print("=" * 64)
    print(f"Dataset    : {DATASET_NAME}")
    print(f"Keep ratio : {KEEP_RATIO:.0%}")
    print(f"Window     : {WINDOW_DAYS} days non-overlapping")
    print(f"Output dir : {OUTPUT_DIR}")
    print()

    print("[1/3] Loading tgbl-review from TGB ...")

    dataset = PyGLinkPropPredDataset(
        name=DATASET_NAME,
        root=str(DATASET_ROOT)
    )

    df = build_dataframe_from_tgbl_review(dataset)

    date_min = df["_date"].min()
    date_max = df["_date"].max()

    print(f"Edges      : {len(df):,}")
    print(f"Date range : {date_min.date()} -> {date_max.date()}")

    print("\n[2/3] Sparsifying per window ...")

    window_td = pd.Timedelta(days=WINDOW_DAYS)
    step_td = pd.Timedelta(days=WINDOW_STEP)
    window_starts = pd.date_range(date_min, date_max, freq=step_td)

    kept_chunks = []

    for ws in tqdm(window_starts, desc="SEM windows"):
        we = ws + window_td

        mask = (df["_date"] >= ws) & (df["_date"] < we)

        win = (
            df
            .loc[mask, ["from", "to", "timestamp", "value"]]
            .reset_index(drop=True)
        )

        if len(win) == 0:
            continue

        kept = sparsify_window(win, KEEP_RATIO)
        kept_chunks.append(kept)

    print("\n[3/3] Saving output CSV ...")

    if not kept_chunks:
        print("[WARN] No edges were kept.")
        return

    out_df = pd.concat(kept_chunks, ignore_index=True)

    out_path = OUTPUT_DIR / f"{DATASET_NAME}_sem_bfc.csv"
    stats_path = OUTPUT_DIR / f"{DATASET_NAME}_sem_stats.csv"

    out_df.to_csv(out_path, index=False)

    n_total = len(df)
    n_kept = len(out_df)
    pct_kept = 100.0 * n_kept / max(n_total, 1)

    stats = pd.DataFrame([{
        "dataset": DATASET_NAME,
        "criterion": "balanced_forman_curvature",
        "edges_original": n_total,
        "edges_kept": n_kept,
        "percent_kept": round(pct_kept, 2),
        "keep_ratio_target": KEEP_RATIO,
        "windows": len(kept_chunks),
    }])

    stats.to_csv(stats_path, index=False)

    print(f"Edges kept : {n_kept:,} / {n_total:,} ({pct_kept:.2f}%)")
    print(f"Output     : {out_path}")
    print(f"Stats      : {stats_path}")
    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    process_tgbl_review()