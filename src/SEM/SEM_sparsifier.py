"""
SEM_sparsifier.py
=================

Temporal snapshot sparsifier using Balanced Forman Curvature (BFC).

Processes only:

    ADX, BAG, BEPRO, DERC, DINO, ETH2x-FLI, EVERMOON, GLM, HOICHI

Saves:
    sem_sparsified/<DATASET>_sem_bfc.csv
    sem_sparsified/<DATASET>_sem_stats.csv
    sem_sparsified/sem_sparsification_runtime.csv
"""

import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(r"../../data")
OUTPUT_DIR = Path(r"../../sem_sparsified")

WINDOW_DAYS = 7
WINDOW_STEP = 7
KEEP_RATIO = 0.20
SCORE_COL = "bfc_score"

BFC_EXACT_NODE_LIMIT = 3000

DATASETS = [
    "ADX",
    "BAG",
    "BEPRO",
    "DERC",
    "DINO",
    "ETH2x-FLI",
    "EVERMOON",
    "GLM",
    "HOICHI",
]


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
        tri[key] = len(common)

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
                1.0 / du
                + 1.0 / dv
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
# PROCESS ONE DATASET
# ============================================================

def process_dataset(input_csv: Path, keep_ratio: float) -> None:
    start_time = time.perf_counter()

    dataset = input_csv.stem

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime_path = OUTPUT_DIR / "sem_sparsification_runtime.csv"

    if not input_csv.exists():
        print(f"[WARN] Input CSV not found: {input_csv}")
        return

    print("=" * 64)
    print("SEM Baseline -- Balanced Forman Curvature Sparsifier")
    print("=" * 64)
    print(f"  Dataset    : {dataset}")
    print(f"  Input CSV  : {input_csv}")
    print(f"  Keep ratio : {keep_ratio:.0%}")
    print(f"  Window     : {WINDOW_DAYS} days non-overlapping")
    print()

    # -----------------------------
    # Load
    # -----------------------------
    print("[1/3] Loading CSV ...")
    load_start = time.perf_counter()

    df = pd.read_csv(input_csv)

    df = df.rename(columns={
        "source": "from",
        "target": "to",
        "from_address": "from",
        "to_address": "to",
        "block_timestamp": "timestamp",
        "amount": "value",
        "weight": "value",
    })

    required = {"from", "to", "timestamp", "value"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64)

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    df["_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("D")

    date_min = df["_date"].min()
    date_max = df["_date"].max()

    load_seconds = time.perf_counter() - load_start

    print(f"       Edges      : {len(df):,}")
    print(f"       Date range : {date_min.date()} -> {date_max.date()}")

    # -----------------------------
    # Sparsify
    # -----------------------------
    print("\n[2/3] Sparsifying per window ...")
    sparsify_start = time.perf_counter()

    window_td = pd.Timedelta(days=WINDOW_DAYS)
    step_td = pd.Timedelta(days=WINDOW_STEP)

    window_starts = pd.date_range(date_min, date_max, freq=step_td)

    kept_chunks = []

    for ws in tqdm(window_starts, desc=f"{dataset} windows"):
        we = ws + window_td

        mask = (df["_date"] >= ws) & (df["_date"] < we)

        win = df.loc[
            mask,
            ["from", "to", "timestamp", "value"]
        ].reset_index(drop=True)

        if len(win) == 0:
            continue

        kept = sparsify_window(win, keep_ratio)
        kept_chunks.append(kept)

    sparsify_seconds = time.perf_counter() - sparsify_start

    # -----------------------------
    # Save
    # -----------------------------
    print("\n[3/3] Saving output CSV ...")
    save_start = time.perf_counter()

    if not kept_chunks:
        print("[WARN] No edges were kept. Check your input data.")
        return

    out_df = pd.concat(kept_chunks, ignore_index=True)

    out_path = OUTPUT_DIR / f"{dataset}_sem_bfc.csv"
    out_df.to_csv(out_path, index=False)

    n_kept = len(out_df)
    n_total = len(df)
    pct_kept = 100.0 * n_kept / max(n_total, 1)

    stats = pd.DataFrame([{
        "dataset": dataset,
        "criterion": "balanced_forman_curvature",
        "edges_original": n_total,
        "edges_kept": n_kept,
        "percent_kept": round(pct_kept, 2),
        "keep_ratio_target": keep_ratio,
        "windows": len(kept_chunks),
    }])

    stats_path = OUTPUT_DIR / f"{dataset}_sem_stats.csv"
    stats.to_csv(stats_path, index=False)

    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - start_time

    runtime_row = pd.DataFrame([{
        "dataset": dataset,
        "method": "SEM",
        "load_seconds": round(load_seconds, 6),
        "sparsify_seconds": round(sparsify_seconds, 6),
        "save_seconds": round(save_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "edges_original": n_total,
        "edges_kept": n_kept,
        "percent_kept": round(pct_kept, 2),
        "windows": len(kept_chunks),
    }])

    if runtime_path.exists():
        runtime_row.to_csv(runtime_path, mode="a", header=False, index=False)
    else:
        runtime_row.to_csv(runtime_path, index=False)

    print(f"  Edges kept : {n_kept:,} / {n_total:,} ({pct_kept:.1f}%)")
    print(f"  Output     : {out_path}")
    print(f"  Stats      : {stats_path}")
    print(f"  Runtime    : {runtime_path}")
    print(f"  Total time : {total_seconds:.2f}s")
    print()
    print("=" * 64)
    print(f"DONE -- {dataset}")
    print("=" * 64)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    selected_dataset_norms = {
        d.lower().replace("_", "-")
        for d in DATASETS
    }

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"[FATAL] No CSV files found in {INPUT_DIR}")
        raise SystemExit(1)

    selected_files = [
        p for p in csv_files
        if p.stem.lower().replace("_", "-") in selected_dataset_norms
    ]

    if not selected_files:
        print("[FATAL] None of the selected DATASETS were found in INPUT_DIR.")
        print(f"INPUT_DIR: {INPUT_DIR}")
        print(f"DATASETS : {DATASETS}")
        raise SystemExit(1)

    print("=" * 64)
    print("SEM sparsification batch run")
    print("=" * 64)
    print(f"Input dir      : {INPUT_DIR}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Selected       : {DATASETS}")
    print(f"Found selected : {len(selected_files)} dataset(s)")
    print("=" * 64)
    print()

    for csv_path in selected_files:
        process_dataset(csv_path, KEEP_RATIO)

    print("\nAll selected datasets completed.")