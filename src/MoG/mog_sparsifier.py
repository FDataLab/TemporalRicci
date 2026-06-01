"""
mog_sparsifier.py
=================

MoG baseline sparsifier using Effective Resistance (ER).

Processes all CSV datasets in INPUT_DIR except:
    - tgbl-comment.csv
    - tgbl-coin.csv

Also saves sparsification runtime to:
    mog_sparsified/mog_sparsification_runtime.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(r"../../data")
OUTPUT_DIR = Path(r"../../mog_sparsified")

WINDOW_DAYS = 7
WINDOW_STEP = 7
KEEP_RATIO = 0.20
SCORE_COL = "er_score"

ER_EXACT_NODE_LIMIT = 2000

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
# EFFECTIVE RESISTANCE SCORING
# ============================================================

def score_effective_resistance(G: nx.DiGraph) -> dict:
    H = G.to_undirected()
    H.remove_edges_from(nx.selfloop_edges(H))

    nodes = list(H.nodes())
    n = len(nodes)
    scores = {}

    if n < 3 or H.number_of_edges() == 0:
        for u, v in G.edges():
            scores[(u, v)] = 1.0
        return scores

    if n > ER_EXACT_NODE_LIMIT:
        for u, v, data in G.edges(data=True):
            w = data.get("value", 1.0)
            scores[(u, v)] = 1.0 / max(float(w), 1e-12)
        return scores

    try:
        node_idx = {node: i for i, node in enumerate(nodes)}
        L = nx.laplacian_matrix(H).toarray().astype(np.float64)
        L_pinv = np.linalg.pinv(L)

        for u, v in G.edges():
            if u not in node_idx or v not in node_idx:
                scores[(u, v)] = 0.0
                continue

            i, j = node_idx[u], node_idx[v]
            er = L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j]
            scores[(u, v)] = max(er, 0.0)

    except Exception:
        for u, v in G.edges():
            scores[(u, v)] = 1.0

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

    scores = score_effective_resistance(G)

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
    runtime_path = OUTPUT_DIR / "mog_sparsification_runtime.csv"

    if not input_csv.exists():
        print(f"[WARN] Input CSV not found: {input_csv}")
        return

    print("=" * 64)
    print("MoG Baseline -- Effective Resistance Sparsifier")
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

    out_path = OUTPUT_DIR / f"{dataset}_mog_er.csv"
    out_df.to_csv(out_path, index=False)

    n_kept = len(out_df)
    n_total = len(df)
    pct_kept = 100.0 * n_kept / max(n_total, 1)

    stats = pd.DataFrame([{
        "dataset": dataset,
        "criterion": "effective_resistance",
        "edges_original": n_total,
        "edges_kept": n_kept,
        "percent_kept": round(pct_kept, 2),
        "keep_ratio_target": keep_ratio,
        "windows": len(kept_chunks),
    }])

    stats_path = OUTPUT_DIR / f"{dataset}_mog_stats.csv"
    stats.to_csv(stats_path, index=False)

    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - start_time

    runtime_row = pd.DataFrame([{
        "dataset": dataset,
        "method": "MoG",
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
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"[FATAL] No CSV files found in {INPUT_DIR}")
        raise SystemExit(1)

    selected_dataset_norms = {
        d.lower().replace("_", "-")
        for d in DATASETS
    }

    selected_files = [
        p for p in csv_files
        if p.stem.lower().replace("_", "-") in selected_dataset_norms
    ]

    print("=" * 64)
    print("MoG sparsification batch run")
    print("=" * 64)
    print(f"Input dir      : {INPUT_DIR}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Found CSVs     : {len(csv_files)}")
    print(f"Selected       : {DATASETS}")
    print(f"Processing     : {len(selected_files)} dataset(s)")
    print(f"Processing     : {len(selected_files)} dataset(s)")
    print("=" * 64)
    print()

    for csv_path in selected_files:
        process_dataset(csv_path, KEEP_RATIO)

    print("\nAll selected datasets completed.")