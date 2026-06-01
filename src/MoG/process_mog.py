#!/usr/bin/env python3
"""
process_mog.py
==============

Runs the exact same TDA + RNN pipeline as process_dataset.py,
but reads sparsified graphs from the flat mog_sparcified/ folder
instead of RicciResults/ricci_values_windowed/.

Labels are always taken from the full graph (same as process_dataset.py)
so results are directly comparable.

Input layout expected:
    mog_sparcified/
        ADX_mog_er.csv
        BAG_mog_er.csv
        HOICHI_mog_er.csv
        ...

Output layout (mirrors process_dataset.py output):
    GraphPulseResultsWindowed/
        Sequence_task1/
            ADX_mog_er/
                seq_tda.txt
                seq_raw.txt
        Sequence_task2/ ...
        Sequence_task3/ ...
"""

import json
import os
import time
import csv
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
import gc
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler
# ============================================================
# CONFIG
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent          # src/MoG/
_BASE_DIR = _THIS_DIR.parent.parent                  # TemporalRicci/


DATASET = "tgbl-comment"

MOG_DIR  = _BASE_DIR / "mog_sparsified"

base_dir = str(_BASE_DIR)

# ============================================================
# DERIVED PATHS
# ============================================================

DATA_DIR = os.path.join(base_dir, "data")

GRAPHPULSE_RESULTS_DIR = os.path.join(base_dir, "GraphPulseResultsMoG")
os.makedirs(GRAPHPULSE_RESULTS_DIR, exist_ok=True)

# Mapper settings
OVER_LAP = 0.2
N_CUBE   = 2
CLS      = 5

windowSize, gap, labelWindowSize = 7, 3, 7

# ============================================================
# SANITY CHECK
# ============================================================

if not MOG_DIR.exists():
    print(f"[FATAL] MoG output folder not found: {MOG_DIR}")
    raise SystemExit(1)

mog_csv = MOG_DIR / f"{DATASET}_mog_er.csv"
if not mog_csv.exists():
    print(f"[FATAL] File not found: {mog_csv}")
    print(f"        Run mog_sparsifier.py first for dataset: {DATASET}")
    raise SystemExit(1)

print(f"Dataset  : {DATASET}")
print(f"MoG CSV  : {mog_csv}")
print(f"Output   : {GRAPHPULSE_RESULTS_DIR}")
print()

# ============================================================
# HELPERS
# ============================================================

def label_task1(df_data, df_label):
    return int(len(df_label) > len(df_data))


def label_task2(G_current, G_next, k_ratio=0.10, threshold=0.30):
    if G_next is None or G_next.number_of_nodes() == 0:
        return 0
    deg_curr = dict(G_current.degree())
    deg_next = dict(G_next.degree())
    if len(deg_curr) == 0 or len(deg_next) == 0:
        return 0
    k = max(1, int(len(deg_curr) * k_ratio))
    top_k_curr = set(n for n, _ in sorted(deg_curr.items(), key=lambda x: x[1], reverse=True)[:k])
    top_k_next = set(n for n, _ in sorted(deg_next.items(), key=lambda x: x[1], reverse=True)[:k])
    return int(len(top_k_next - top_k_curr) / k > threshold)


def label_task3(df_data, df_label):
    if len(df_data) < 1 or len(df_label) < 1:
        return 0
    data_nodes  = set(df_data["from"]).union(set(df_data["to"]))
    label_nodes = set(df_label["from"]).union(set(df_label["to"]))
    return int(len(label_nodes) > len(data_nodes))


def merge_dicts(list_of_dicts):
    merged = {}
    for d in list_of_dicts:
        for k, v in d.items():
            merged.setdefault(k, []).append(v)
    return merged


def to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    return obj


# ============================================================
# TDA
# ============================================================

def precompute_daily_tda(csv_path: str, variant_name: str,
                          timings_writer, **kwargs) -> dict | None:
    """
    Compute daily KeplerMapper TDA features.
    Reads from csv_path directly (works for both flat and subfolder layouts).
    """
    if not os.path.exists(csv_path):
        print(f"[SKIP] File not found: {csv_path}")
        return None

    print(f"[LOAD] Reading {csv_path}")
    df = pd.read_csv(csv_path)

    if df.shape[0] < 100:
        print(f"[SKIP] {variant_name}: only {df.shape[0]} edges (<100).")
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.floor("D")
    daily_tda_cache = {}
    all_dates = pd.date_range(start=df["timestamp"].min(),
                               end=df["timestamp"].max(), freq="D")
    print(f"  Date range: {all_dates[0].date()} -> {all_dates[-1].date()}")

    blank_day_counter = 0
    start_unix = time.time()
    t0 = time.perf_counter()

    print(f"[TDA] Computing daily Mapper for {variant_name} ({len(all_dates)} days)")
    for current_date in tqdm(all_dates, desc=f"{variant_name} - TDA"):
        next_day = current_date + pd.Timedelta(days=1)
        sub_df = df[(df["timestamp"] >= current_date) & (df["timestamp"] < next_day)]
        num_edges = len(sub_df)
        num_nodes = len(set(sub_df["from"]).union(sub_df["to"]))

        if num_edges < 3 or num_nodes < 2:
            daily_tda_cache[pd.Timestamp(current_date)] = {
                "mapper": [num_nodes, num_edges, num_nodes, num_nodes]
            }
            blank_day_counter += 1
            continue

        outgoing_wsum = sub_df.groupby("from")["value"].sum()
        incoming_wsum = sub_df.groupby("to")["value"].sum()
        outgoing_cnt  = sub_df.groupby("from")["value"].count()
        incoming_cnt  = sub_df.groupby("to")["value"].count()

        records = [{
            "nodeID": n,
            "outgoing_edge_weight_sum": outgoing_wsum.get(n, 0),
            "incoming_edge_weight_sum": incoming_wsum.get(n, 0),
            "outgoing_edge_count":      outgoing_cnt.get(n, 0),
            "incoming_edge_count":      incoming_cnt.get(n, 0),
        } for n in set(sub_df["from"]).union(sub_df["to"])]

        X = pd.DataFrame(records).drop(columns=["nodeID"], errors="ignore")
        if X.shape[0] < 3:
            daily_tda_cache[pd.Timestamp(current_date)] = {
                "mapper": [num_nodes, num_edges, num_nodes, num_nodes]
            }
            blank_day_counter += 1
            continue

        mapper   = km.KeplerMapper()
        X_scaled = MinMaxScaler((0, 1)).fit_transform(X)
        perplexity = max(2, min(30, X_scaled.shape[0] // 3))
        try:
            lens = sklearn.manifold.TSNE(
                perplexity=perplexity, init="random",
                random_state=42, max_iter=500
            ).fit_transform(X_scaled)
        except Exception:
            lens = X_scaled

        mapper_graph = mapper.map(
            lens, X_scaled,
            clusterer=sklearn.cluster.KMeans(
                n_clusters=min(kwargs["cls"], max(1, X_scaled.shape[0] // 2)),
                random_state=42
            ),
            cover=km.Cover(n_cubes=kwargs["n_cube"], perc_overlap=kwargs["over_lap"])
        )

        nodes_in_map  = len(mapper_graph["nodes"])
        edges_in_map  = sum(len(v) for v in mapper_graph["links"].values())
        cluster_sizes = [len(v) for v in mapper_graph["nodes"].values()]
        avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        max_size = max(cluster_sizes) if cluster_sizes else 0

        daily_tda_cache[pd.Timestamp(current_date)] = {
            "mapper": [nodes_in_map, edges_in_map, max_size, avg_size]
        }

    duration = time.perf_counter() - t0
    timings_writer.writerow({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": variant_name,
        "variant": variant_name,
        "phase": "tda",
        "alpha": "", "beta": "",
        "start_unix": f"{start_unix:.6f}",
        "end_unix":   f"{time.time():.6f}",
        "seconds":    f"{duration:.6f}",
    })
    print(f"[DONE] TDA {variant_name}: {blank_day_counter} empty days, {duration:.2f}s\n")
    return daily_tda_cache


# ============================================================
# SEQUENCE GENERATION
# ============================================================

def create_rnn_sequence(
    mog_csv_path: str,       # path to the MoG sparsified CSV
    basis_csv_path: str,     # path to the full-graph CSV (for labels)
    variant_name: str,       # e.g. "ADX_mog_er"
    daily_tda_cache: dict,
    timings_writer,
) -> None:
    """
    Generate seq_tda.txt and seq_raw.txt for all three tasks.
    Labels come from basis_csv_path (full graph) — same as process_dataset.py.
    """
    print(f"[RNN] Processing {variant_name}")
    start_unix = time.time()
    t0 = time.perf_counter()

    df       = pd.read_csv(mog_csv_path)
    df_basis = pd.read_csv(basis_csv_path)

    df["timestamp"]       = pd.to_datetime(df["timestamp"],       unit="s").dt.floor("D")
    df_basis["timestamp"] = pd.to_datetime(df_basis["timestamp"], unit="s").dt.floor("D")

    df       = df.sort_values("timestamp")
    df_basis = df_basis.sort_values("timestamp")
    df["value"] = df["value"].astype(float)

    start_date   = df_basis["timestamp"].min()
    last_date    = df_basis["timestamp"].max()
    num_windows  = max(0, int((last_date - start_date).days
                              - (windowSize + gap + labelWindowSize)))

    for task_id, task_name, task_folder in [
        (1, "task1", "Sequence_task1"),
        (2, "task2", "Sequence_task2"),
        (3, "task3", "Sequence_task3"),
    ]:
        print(f"\n=== {task_name} for {variant_name} ===")
        seq_tda, seq_raw, seq_labels = [], [], []
        ws_date = start_date

        for _ in tqdm(range(num_windows), desc=f"{variant_name} - {task_name}", leave=False):
            w_end   = ws_date + pd.Timedelta(days=windowSize)
            l_start = ws_date + pd.Timedelta(days=windowSize + gap)
            l_end   = l_start + pd.Timedelta(days=labelWindowSize)

            df_win   = df[(df["timestamp"] >= ws_date) & (df["timestamp"] < w_end)]
            df_label = df[(df["timestamp"] >= l_start) & (df["timestamp"] < l_end)]

            G = nx.from_pandas_edgelist(
                df_win, "from", "to", ["value"], create_using=nx.MultiDiGraph()
            )

            # Labels
            if task_id == 1:
                label = label_task1(df_win, df_label)
            elif task_id == 2:
                G_next = (nx.from_pandas_edgelist(
                    df_label, "from", "to", ["value"], create_using=nx.MultiDiGraph()
                ) if len(df_label) > 0 else None)
                label = label_task2(G, G_next)
            else:
                label = label_task3(df_win, df_label)

            seq_labels.append(label)

            # Features
            daily_feats = []
            daily_raws  = []

            for i in range(windowSize):
                key_date = pd.Timestamp(ws_date + pd.Timedelta(days=i)).floor("D")

                if key_date in daily_tda_cache:
                    daily_feats.append(daily_tda_cache[key_date])
                else:
                    daily_feats.append({"mapper": [0, 0, 0, 0]})

                # Raw graph features for this day
                df_win_day = df[
                    (df["timestamp"] >= ws_date) &
                    (df["timestamp"] < ws_date + pd.Timedelta(days=i + 1))
                ]
                G_day = nx.from_pandas_edgelist(
                    df_win_day, "from", "to", ["value"], create_using=nx.MultiDiGraph()
                )
                n = G_day.number_of_nodes()
                e = G_day.number_of_edges()
                avg_deg = 0 if n == 0 else (
                    sum(dict(G_day.to_undirected().degree()).values()) / n
                )

                wdeg = {
                    node: sum(d.get("value", 0) for _, _, d in G_day.edges(node, data=True))
                    for node in G_day.nodes()
                }
                total_wdeg = sum(wdeg.values())
                if total_wdeg > 0 and len(wdeg) > 0:
                    k_top = max(1, int(len(wdeg) * 0.10))
                    whale_dominance = sum(sorted(wdeg.values(), reverse=True)[:k_top]) / total_wdeg
                else:
                    whale_dominance = 0

                H = G_day.to_undirected()
                gcc_size       = len(max(nx.connected_components(H), key=len)) if H.number_of_nodes() > 0 else 0
                num_components = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
                total_volume   = sum(d.get("value", 0) for _, _, d in G_day.edges(data=True))
                avg_transaction = total_volume / e if e > 0 else 0

                daily_raws.append({"raw": [
                    n, e, avg_deg, whale_dominance,
                    gcc_size, num_components,
                    total_volume, avg_transaction
                ]})

            seq_tda.append(merge_dicts(daily_feats))
            seq_raw.append(merge_dicts(daily_raws))
            ws_date += pd.Timedelta(days=1)

        # Save
        output_dir = os.path.join(GRAPHPULSE_RESULTS_DIR, task_folder, variant_name)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "seq_tda.txt"), "w", encoding="utf-8") as f:
            json.dump(
                {"TDA_SEQUENCE": to_builtin(merge_dicts(seq_tda)), "LABELS": seq_labels},
                f, indent=2
            )
        with open(os.path.join(output_dir, "seq_raw.txt"), "w", encoding="utf-8") as f:
            json.dump(
                {"RAW_SEQUENCE": to_builtin(merge_dicts(seq_raw)), "LABELS": seq_labels},
                f, indent=2
            )

        gc.collect()

    duration = time.perf_counter() - t0
    timings_writer.writerow({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": variant_name,
        "variant": variant_name,
        "phase": "rnn_labeling",
        "alpha": "", "beta": "",
        "start_unix": f"{start_unix:.6f}",
        "end_unix":   f"{time.time():.6f}",
        "seconds":    f"{duration:.6f}",
    })
    print(f"[DONE] RNN sequences saved for {variant_name} in {duration:.2f}s\n")


# ============================================================
# DRIVER
# ============================================================

overall_start  = time.perf_counter()
timings_path   = os.path.join(GRAPHPULSE_RESULTS_DIR, "run_times_mog.csv")
timings_exists = os.path.exists(timings_path)

timings_f      = open(timings_path, "a", newline="", encoding="utf-8")
timings_writer = csv.DictWriter(
    timings_f,
    fieldnames=["current_time", "dataset", "variant", "phase",
                "alpha", "beta", "start_unix", "end_unix", "seconds"],
)
if not timings_exists:
    timings_writer.writeheader()

variant_name = f"{DATASET}_mog_er"

basis_csv = os.path.join(DATA_DIR, f"{DATASET}.csv")
if not os.path.exists(basis_csv):
    print(f"[FATAL] Full graph not found: {basis_csv}")
    print(f"        This is needed for label generation.")
    raise SystemExit(1)

print(f"{'='*60}")
print(f"Processing : {variant_name}")
print(f"MoG CSV    : {mog_csv}")
print(f"Basis CSV  : {basis_csv}")
print(f"{'='*60}\n")

# Step 1: TDA on the MoG sparsified graph
daily_tda_cache = precompute_daily_tda(
    csv_path=str(mog_csv),
    variant_name=variant_name,
    timings_writer=timings_writer,
    over_lap=OVER_LAP,
    n_cube=N_CUBE,
    cls=CLS,
)

if daily_tda_cache is None:
    print(f"[FATAL] TDA failed for {variant_name}")
    raise SystemExit(1)

# Step 2: RNN sequences
create_rnn_sequence(
    mog_csv_path=str(mog_csv),
    basis_csv_path=basis_csv,
    variant_name=variant_name,
    daily_tda_cache=daily_tda_cache,
    timings_writer=timings_writer,
)

timings_f.close()

total = time.perf_counter() - overall_start
h = int(total // 3600)
m = int((total % 3600) // 60)
s = total % 60
print(f"\n{'='*60}")
print(f"All MoG datasets completed in {h}h {m}m {s:.2f}s")
print(f"{'='*60}\n")