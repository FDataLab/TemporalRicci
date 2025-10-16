#!/usr/bin/env python3
"""
GraphPulse multi-dataset driver.

This script orchestrates:
  1. Loading Ricci curvature CSV datasets.
  2. Running daily Topological Data Analysis (TDA) via KeplerMapper.
  3. Generating temporal RNN training sequences and labels.
  4. Logging runtimes for each phase in Unix seconds.

"""

import argparse
import os
import time
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
import gc
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# ARGUMENT PARSING
# ============================================================
parser_args = argparse.ArgumentParser(
    description="GraphPulse multi-dataset driver with Unix time logging."
)
parser_args.add_argument(
    "--datasets",
    nargs="+",
    required=True,
    help="List of dataset names under RicciResults/ricci_values/ (e.g. BEPRO LCC ETH)",
)
parser_args.add_argument("--alpha", type=float, default=3.0)
parser_args.add_argument("--beta", type=float, default=1.0)
parser_args.add_argument("--overlap", type=float, default=0.2, help="Mapper overlap fraction.")
parser_args.add_argument("--ncube", type=int, default=2, help="Number of cubes for Mapper cover.")
parser_args.add_argument("--cls", type=int, default=5, help="Cluster count (k-means).")
parser_args.add_argument("--bins", type=int, default=10, help="Number of bin variants per dataset.")
args = parser_args.parse_args()

# Static configuration derived from arguments.
ALPHA = args.alpha
BETA = args.beta
OVER_LAP = args.overlap
N_CUBE = args.ncube
CLS = args.cls
BINS = args.bins

# Base I/O paths.
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
GRAPHPULSE_RESULTS_DIR = os.path.join(base_dir, "GraphPulseResults")
RICCI_ROOT = os.path.join(base_dir, "RicciResults/ricci_values")
os.makedirs(GRAPHPULSE_RESULTS_DIR, exist_ok=True)


# ============================================================
# SANITY CHECKS
# ============================================================
# Fail fast if the Ricci curvature directory is missing or empty.
if not os.path.exists(RICCI_ROOT) or not os.listdir(RICCI_ROOT):
    print(
        f"[FATAL ❌] Ricci curvature values are missing.\n"
        f"Expected directory: {RICCI_ROOT}\n"
        f"Ensure RicciResults/ricci_values/ contains dataset subfolders like 'BEPRO/'."
    )
    exit(1)


# ============================================================
# LABELING HELPERS
# ============================================================
def label_task1(selectedNetworkInGraphDataWindow: pd.DataFrame,
                selectedNetworkInLabelingWindow: pd.DataFrame) -> int:
    """Return 1 if the labeling window has more edges than the data window."""
    return int(len(selectedNetworkInLabelingWindow) > len(selectedNetworkInGraphDataWindow))


def label_task2(G: nx.MultiDiGraph,
                selectedNetwork: pd.DataFrame,
                prevNetwork: pd.DataFrame | None = None) -> int:
    """
    Label based on the change in top 1% address volume.
    Returns 1 if current top-volume count > previous; else 0.
    """
    volume_by_addr = (
        selectedNetwork.groupby("from")["value"].sum()
        .add(selectedNetwork.groupby("to")["value"].sum(), fill_value=0)
    )
    top_n = max(1, int(len(volume_by_addr) * 0.01))
    current_count = len(volume_by_addr.nlargest(top_n))

    if prevNetwork is None:
        return 0

    prev_volume = (
        prevNetwork.groupby("from")["value"].sum()
        .add(prevNetwork.groupby("to")["value"].sum(), fill_value=0)
    )
    prev_top_n = max(1, int(len(prev_volume) * 0.01))
    prev_count = len(prev_volume.nlargest(prev_top_n))
    return int(current_count > prev_count)


def label_task3(G: nx.MultiDiGraph, prevG: nx.MultiDiGraph | None = None) -> int:
    """
    Label based on change in connected component count.
    Returns 1 if components increased since previous day.
    """
    H = G.to_undirected()
    current_count = nx.number_connected_components(H) if H.nodes else 0

    if prevG is None:
        return 0

    prevH = prevG.to_undirected()
    prev_count = nx.number_connected_components(prevH) if prevH.nodes else 0
    return int(current_count > prev_count)


def merge_dicts(list_of_dicts: list[dict]) -> dict:
    """Merge a list of daily feature dicts by concatenating values per key."""
    merged = {}
    for d in list_of_dicts:
        for k, v in d.items():
            merged.setdefault(k, []).append(v)
    return merged


# ============================================================
# DAILY MAPPER / TDA COMPUTATION
# ============================================================
def precompute_all_daily_tda(dataset_name: str,
                             variant_name: str,
                             alpha: float,
                             beta: float,
                             timings_writer: csv.DictWriter,
                             **kwargs) -> dict | None:
    """
    For each day in a dataset, compute a Mapper graph using KeplerMapper.

    Args:
        dataset_name: Base folder under RICCI_ROOT.
        variant_name: CSV stem (e.g. BEPRO_TFR_a3.00_b1.00_bin3).
        alpha, beta: Parameters recorded for tracking.
        timings_writer: CSV writer for runtime logging.

    Returns:
        Dictionary keyed by date (Timestamp) → Mapper feature summary.
        None if the dataset is missing or too small.
    """
    csv_path = os.path.join(RICCI_ROOT, dataset_name, f"{variant_name}.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {variant_name}: file not found.")
        return None

    print(f"[LOAD] Reading {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[0] < 100:
        print(f"[SKIP] {variant_name}: only {df.shape[0]} edges (<100).")
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.floor("D")
    daily_tda_cache: dict = {}
    all_dates = sorted(df["timestamp"].unique())

    start_unix = time.time()
    t0 = time.perf_counter()

    print(f"[TDA] Computing daily Mapper for {variant_name} ({len(all_dates)} days)")
    for current_date in tqdm(all_dates, desc=f"{variant_name} - TDA"):
        # Filter one-day subgraph.
        next_day = current_date + pd.Timedelta(days=1)
        sub_df = df[(df["timestamp"] >= current_date) & (df["timestamp"] < next_day)]
        num_edges, num_nodes = len(sub_df), len(set(sub_df["from"]).union(sub_df["to"]))

        if num_edges < 3 or num_nodes < 2:
            # Fallback: minimal structure, still store for continuity.
            daily_tda_cache[pd.Timestamp(current_date)] = {
                f"overlap{kwargs['over_lap']}-cube{kwargs['n_cube']}-cls{kwargs['cls']}":
                    [num_nodes, num_edges, num_nodes, num_nodes]
            }
            continue

        # Node-level feature construction.
        outgoing_wsum = sub_df.groupby("from")["value"].sum()
        incoming_wsum = sub_df.groupby("to")["value"].sum()
        outgoing_cnt = sub_df.groupby("from")["value"].count()
        incoming_cnt = sub_df.groupby("to")["value"].count()

        records = [{
            "nodeID": n,
            "outgoing_edge_weight_sum": outgoing_wsum.get(n, 0),
            "incoming_edge_weight_sum": incoming_wsum.get(n, 0),
            "outgoing_edge_count": outgoing_cnt.get(n, 0),
            "incoming_edge_count": incoming_cnt.get(n, 0)
        } for n in set(sub_df["from"]).union(sub_df["to"])]

        X = pd.DataFrame(records).drop(columns=["nodeID"], errors="ignore")
        if X.shape[0] < 3:
            daily_tda_cache[pd.Timestamp(current_date)] = {
                f"overlap{kwargs['over_lap']}-cube{kwargs['n_cube']}-cls{kwargs['cls']}":
                    [num_nodes, num_edges, num_nodes, num_nodes]
            }
            continue

        # Normalize and project via t-SNE for lens construction.
        mapper = km.KeplerMapper()
        X_scaled = MinMaxScaler((0, 1)).fit_transform(X)
        perplexity = max(2, min(30, X_scaled.shape[0] // 3))
        try:
            lens = sklearn.manifold.TSNE(perplexity=perplexity, init="random", max_iter=500).fit_transform(X_scaled)
        except Exception:
            lens = X_scaled  # fallback: identity projection

        # Mapper graph computation.
        mapper_graph = mapper.map(
            lens,
            X_scaled,
            clusterer=sklearn.cluster.KMeans(
                n_clusters=min(kwargs["cls"], max(1, X_scaled.shape[0] // 2)),
                random_state=1618033,
            ),
            cover=km.Cover(n_cubes=kwargs["n_cube"], perc_overlap=kwargs["over_lap"]),
        )

        # Extract structural metrics.
        n_nodes = len(mapper_graph["nodes"])
        n_edges = sum(len(v) for v in mapper_graph["links"].values())
        cluster_sizes = [len(v) for v in mapper_graph["nodes"].values()]
        avg_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_size = max(cluster_sizes) if cluster_sizes else 0

        daily_tda_cache[pd.Timestamp(current_date)] = {
            f"overlap{kwargs['over_lap']}-cube{kwargs['n_cube']}-cls{kwargs['cls']}":
                [n_nodes, n_edges, max_size, avg_size]
        }

    # Runtime logging.
    duration = time.perf_counter() - t0
    timings_writer.writerow({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": dataset_name,
        "variant":variant_name,
        "phase": "tda",
        "alpha": alpha,
        "beta": beta,
        "start_unix": f"{start_unix:.6f}",
        "end_unix": f"{time.time():.6f}",
        "seconds": f"{duration:.6f}",
    })

    print(f"[DONE] TDA {variant_name} in {duration:.2f}s\n")
    return daily_tda_cache


# ============================================================
# TEMPORAL LABELING / SEQUENCE GENERATION
# ============================================================
def create_time_series_rnn_sequence(file: str,
                                    dataset_name: str,
                                    alpha: float,
                                    beta: float,
                                    variant_name: str,
                                    daily_tda_cache: dict,
                                    timings_writer: csv.DictWriter,
                                    **kwargs) -> None:
    """
    Generate RNN training sequences with labels from precomputed daily TDA features.

    Each window aggregates Mapper features over 7 days (windowSize)
    and assigns a label depending on future network evolution.

    Args:
        file: CSV file name.
        dataset_name: Base dataset folder.
        alpha, beta: Experimental parameters for bookkeeping.
        variant_name: Dataset variant identifier.
        daily_tda_cache: Daily TDA feature cache.
        timings_writer: CSV writer for runtime metrics.
    """
    print(f"Processing {file}")
    start_unix = time.time()
    t0 = time.perf_counter()

    windowSize, gap, labelWindowSize = 7, 3, 7
    csv_path = os.path.join(RICCI_ROOT, dataset_name, file)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").sort_values()
    df["value"] = df["value"].astype(float)
    df = df.sort_values("timestamp")

    last_date, start_date = df["timestamp"].max(), df["timestamp"].min()
    num_windows = int((last_date - start_date).days - (windowSize + gap + labelWindowSize))

    for task_id, name, _ in [
        (1, "task1", "Sequence_task1"),
        (2, "task2", "Sequence_task2"),
        (3, "task3", "Sequence_task3"),
    ]:
        print(f"\n=== Running {name} labeling for {file} ===")
        seq_data, seq_labels = [], []
        ws_date = start_date

        for _ in tqdm(range(num_windows), desc=f"{file} - {name}", leave=False):
            w_end = ws_date + pd.Timedelta(days=windowSize)
            l_start = ws_date + pd.Timedelta(days=windowSize + gap)
            l_end = l_start + pd.Timedelta(days=labelWindowSize)

            df_win = df[(df["timestamp"] >= ws_date) & (df["timestamp"] < w_end)]
            df_label = df[(df["timestamp"] >= l_start) & (df["timestamp"] < l_end)]
            G = nx.from_pandas_edgelist(df_win, "from", "to", ["value"], create_using=nx.MultiDiGraph())

            if task_id == 1:
                label = label_task1(df_win, df_label)
            elif task_id == 2:
                label = label_task2(G, df_win)
            else:
                label = label_task3(G)

            # Aggregate daily TDA features inside this window.
            daily_feats = [daily_tda_cache.get(pd.Timestamp(ws_date + pd.Timedelta(days=i)).floor("D"))
                           for i in range(windowSize)
                           if pd.Timestamp(ws_date + pd.Timedelta(days=i)).floor("D") in daily_tda_cache]
            if not daily_feats:
                ws_date += pd.Timedelta(days=1)
                continue

            seq_data.append(merge_dicts(daily_feats))
            seq_labels.append(label)
            ws_date += pd.Timedelta(days=1)

        print(f"[OK] {variant_name} {name}: {len(seq_labels)} windows labeled.")
        gc.collect()

    duration = time.perf_counter() - t0
    timings_writer.writerow({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": dataset_name,
        "variant": variant_name,
        "phase": "rnn_labeling",
        "alpha": alpha,
        "beta": beta,
        "start_unix": f"{start_unix:.6f}",
        "end_unix": f"{time.time():.6f}",
        "seconds": f"{duration:.6f}",
    })

    print(f"[DONE] Labeling {variant_name} in {duration:.2f}s\n")


# ============================================================
# DRIVER ENTRY POINT
# ============================================================
overall_start = time.perf_counter()
timings_path = os.path.join(GRAPHPULSE_RESULTS_DIR, "run_times.csv")
timings_exists = os.path.exists(timings_path)
timings_f = open(timings_path, "a", newline="", encoding="utf-8")
timings_writer = csv.DictWriter(
    timings_f,
    fieldnames=["current_time", "dataset","variant", "phase", "alpha", "beta", "start_unix", "end_unix", "seconds"],
)

if not timings_exists:
    timings_writer.writeheader()

for dataset_name in args.datasets:
    dataset_prefix = f"{dataset_name}_TFR_a{ALPHA:.2f}_b{BETA:.2f}"
    # Main and binned variants.
    test_datasets = {dataset_prefix: {"over_lap": OVER_LAP, "n_cube": N_CUBE, "cls": CLS}}
    for i in range(1, BINS + 1):
        test_datasets[f"{dataset_prefix}_bin{i}"] = {"over_lap": OVER_LAP, "n_cube": N_CUBE, "cls": CLS}

    for variant_name, cfg in test_datasets.items():
        daily_tda_cache = precompute_all_daily_tda(
            dataset_name, variant_name, ALPHA, BETA, timings_writer,
            over_lap=cfg["over_lap"], n_cube=cfg["n_cube"], cls=cfg["cls"]
        )
        if daily_tda_cache is None:
            print(f"[SKIP] No TDA cache computed for {variant_name}")
            continue

        create_time_series_rnn_sequence(
            f"{variant_name}.csv", dataset_name, ALPHA, BETA, variant_name,
            daily_tda_cache, timings_writer,
            over_lap=cfg["over_lap"], n_cube=cfg["n_cube"], cls=cfg["cls"]
        )

timings_f.close()
print(f"\nAll datasets completed in {time.perf_counter() - overall_start:.2f} seconds.")
