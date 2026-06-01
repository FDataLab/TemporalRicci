#!/usr/bin/env python3
"""
GraphPulse multi-dataset driver with enhanced features for Tasks 2 & 3.

This script orchestrates:
  1. Loading Ricci curvature CSV datasets.
  2. Running daily Topological Data Analysis (TDA) via KeplerMapper.
  3. Generating temporal RNN training sequences and labels with enhanced features.
  4. Logging runtimes for each phase in Unix seconds.

Expected input file format inside each dataset folder:
  <DATASET>_full.csv
  <DATASET>_bin1.csv
  <DATASET>_bin2.csv
  ...
  <DATASET>_binK.csv
"""

import argparse
import json
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
    help="List of dataset names under RicciResults/ricci_values_windowed/ (e.g. tgbl-wiki LCC ETH)"
)
parser_args.add_argument("--overlap", type=float, default=0.2, help="Mapper overlap fraction.")
parser_args.add_argument("--ncube", type=int, default=2, help="Number of cubes for Mapper cover.")
parser_args.add_argument("--cls", type=int, default=5, help="Cluster count (k-means).")
parser_args.add_argument("--bins", type=int, default=5, help="Number of bin variants per dataset.")
args = parser_args.parse_args()

OVER_LAP = args.overlap
N_CUBE = args.ncube
CLS = args.cls
BINS = args.bins

# temporal parameters from GraphPulse
windowSize, gap, labelWindowSize = 7, 3, 7

# directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
GRAPHPULSE_RESULTS_DIR = os.path.join(base_dir, "GraphPulseResultsWindowed")
RICCI_ROOT = os.path.join(base_dir, "RicciResults/ricci_values_windowed")
os.makedirs(GRAPHPULSE_RESULTS_DIR, exist_ok=True)

# ============================================================
# SANITY CHECKS
# ============================================================
if not os.path.exists(RICCI_ROOT) or not os.listdir(RICCI_ROOT):
    print(
        f"[FATAL] Ricci curvature values are missing.\n"
        f"Expected: {RICCI_ROOT}\n"
        f"Ensure RicciResults/ricci_values_windowed/ contains dataset subfolders like 'tgbl-wiki/'."
    )
    raise SystemExit(1)


# ============================================================
# LABELING HELPERS
# ============================================================
def label_task1(
    selectedNetworkInGraphDataWindow: pd.DataFrame,
    selectedNetworkInLabelingWindow: pd.DataFrame
) -> int:
    """Return 1 if the labeling window has more edges than the data window."""
    return int(len(selectedNetworkInLabelingWindow) > len(selectedNetworkInGraphDataWindow))


def label_task2(G_current, G_next, k_ratio=0.10, threshold=0.30):
    """Predict whether the top-k degree set changes sufficiently in the next window."""
    if G_next is None or G_next.number_of_nodes() == 0:
        return 0

    deg_curr = dict(G_current.degree())
    deg_next = dict(G_next.degree())

    if len(deg_curr) == 0 or len(deg_next) == 0:
        return 0

    k = max(1, int(len(deg_curr) * k_ratio))

    top_k_curr = set(
        n for n, _ in sorted(deg_curr.items(), key=lambda x: x[1], reverse=True)[:k]
    )
    top_k_next = set(
        n for n, _ in sorted(deg_next.items(), key=lambda x: x[1], reverse=True)[:k]
    )

    new_nodes = top_k_next - top_k_curr
    ratio_new = len(new_nodes) / k

    return int(ratio_new > threshold)


def label_task3(
    df_data_window: pd.DataFrame,
    df_label_window: pd.DataFrame
) -> int:
    """
    Task 3: Network Participation Increase Prediction

    Returns:
        1 if label_window has more unique nodes than data_window
        0 otherwise
    """
    if len(df_data_window) < 1 or len(df_label_window) < 1:
        return 0

    data_nodes = set(df_data_window["from"]).union(set(df_data_window["to"]))
    label_nodes = set(df_label_window["from"]).union(set(df_label_window["to"]))

    return int(len(label_nodes) > len(data_nodes))


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
def precompute_all_daily_tda(
    dataset_name: str,
    variant_name: str,
    timings_writer: csv.DictWriter,
    **kwargs
) -> dict | None:
    """
    Compute daily Mapper-TDA features for a given dataset variant.

    variant_name examples:
      tgbl-wiki_full
      tgbl-wiki_bin3
    """
    csv_path = os.path.join(RICCI_ROOT, dataset_name, f"{variant_name}.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {dataset_name}/{variant_name}.csv: file not found.")
        return None

    blank_day_counter = 0
    print(f"[LOAD] Reading {csv_path}")
    df = pd.read_csv(csv_path)

    if df.shape[0] < 100:
        print(f"[SKIP] {variant_name}: only {df.shape[0]} edges (<100).")
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.floor("D")
    daily_tda_cache = {}
    all_dates = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="D")
    print(f"Dataset has earliest: {pd.Timestamp(all_dates[0])}, Latest: {pd.Timestamp(all_dates[-1])}")

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
                "mapper": [num_nodes, num_edges, num_nodes, num_nodes]
            }
            blank_day_counter += 1
            continue

        mapper = km.KeplerMapper()
        X_scaled = MinMaxScaler((0, 1)).fit_transform(X)
        perplexity = max(2, min(30, X_scaled.shape[0] // 3))
        try:
            lens = sklearn.manifold.TSNE(
                perplexity=perplexity,
                init="random",
                random_state=42,
                max_iter=500
            ).fit_transform(X_scaled)
        except Exception:
            lens = X_scaled

        mapper_graph = mapper.map(
            lens,
            X_scaled,
            clusterer=sklearn.cluster.KMeans(
                n_clusters=min(kwargs["cls"], max(1, X_scaled.shape[0] // 2)),
                random_state=42
            ),
            cover=km.Cover(n_cubes=kwargs["n_cube"], perc_overlap=kwargs["over_lap"])
        )

        nodes_in_map_graph = len(mapper_graph["nodes"])
        edges_in_map_graph = sum(len(v) for v in mapper_graph["links"].values())
        cluster_sizes = [len(v) for v in mapper_graph["nodes"].values()]
        avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        max_size = max(cluster_sizes) if cluster_sizes else 0

        daily_tda_cache[pd.Timestamp(current_date)] = {
            "mapper": [nodes_in_map_graph, edges_in_map_graph, max_size, avg_size]
        }

    duration = time.perf_counter() - t0
    timings_writer.writerow({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": dataset_name,
        "variant": variant_name,
        "phase": "tda",
        "alpha": "",
        "beta": "",
        "start_unix": f"{start_unix:.6f}",
        "end_unix": f"{time.time():.6f}",
        "seconds": f"{duration:.6f}",
    })

    print(
        f"[DONE] TDA {variant_name} with {blank_day_counter} empty days in "
        f"{len(all_dates)} days: {duration:.2f}s\n"
    )
    return daily_tda_cache


def to_builtin(obj):
    """Recursively convert numpy types to plain Python."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    return obj


# ============================================================
# SEQUENCE GENERATION AND LABELING WITH ENHANCED FEATURES
# ============================================================
def create_time_series_rnn_sequence(
    file: str,
    dataset_name: str,
    basis_dataset_file: str,
    variant_name: str,
    daily_tda_cache: dict,
    timings_writer: csv.DictWriter
) -> None:
    print(f"Processing {dataset_name}/{file}")
    start_unix = time.time()
    t0 = time.perf_counter()

    csv_path = os.path.join(RICCI_ROOT, dataset_name, file)
    csv_path_basis = os.path.join(RICCI_ROOT, dataset_name, basis_dataset_file)

    df = pd.read_csv(csv_path)
    df_basis = pd.read_csv(csv_path_basis)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("D")
    df_basis["timestamp"] = pd.to_datetime(df_basis["timestamp"], unit="s").dt.floor("D")

    df = df.sort_values("timestamp")
    df_basis = df_basis.sort_values("timestamp")
    df["value"] = df["value"].astype(float)

    start_date = df_basis["timestamp"].min()
    last_date = df_basis["timestamp"].max()
    num_windows = max(0, int((last_date - start_date).days - (windowSize + gap + labelWindowSize)))

    for task_id, name, task_folder in [
        (1, "task1", "Sequence_task1"),
        (2, "task2", "Sequence_task2"),
        (3, "task3", "Sequence_task3"),
    ]:
        print(f"\n=== Running {name} labeling for {file} ===")
        seq_tda, seq_raw, seq_labels = [], [], []
        ws_date = start_date

        for _ in tqdm(range(num_windows), desc=f"{file} - {name}", leave=False):
            w_end = ws_date + pd.Timedelta(days=windowSize)
            l_start = ws_date + pd.Timedelta(days=windowSize + gap)
            l_end = l_start + pd.Timedelta(days=labelWindowSize)

            df_win = df[(df["timestamp"] >= ws_date) & (df["timestamp"] < w_end)]
            df_label = df[(df["timestamp"] >= l_start) & (df["timestamp"] < l_end)]

            G = nx.from_pandas_edgelist(
                df_win, "from", "to", ["value"], create_using=nx.MultiDiGraph()
            )

            # ----- LABELS -----
            if task_id == 1:
                label = label_task1(df_win, df_label)
            elif task_id == 2:
                if df_label is not None and len(df_label) > 0:
                    G_next = nx.from_pandas_edgelist(
                        df_label, "from", "to", ["value"], create_using=nx.MultiDiGraph()
                    )
                else:
                    G_next = None
                label = label_task2(G, G_next)
            else:
                label = label_task3(df_win, df_label)

            seq_labels.append(label)

            # ----- FEATURES -----
            daily_feats = []
            daily_raws = []

            for i in range(windowSize):
                key_date = pd.Timestamp(ws_date + pd.Timedelta(days=i)).floor("D")

                if key_date in daily_tda_cache:
                    daily_feats.append(daily_tda_cache[key_date])

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

                    # Weighted degree distribution
                    wdeg = {
                        node: sum(data.get("value", 0) for _, _, data in G_day.edges(node, data=True))
                        for node in G_day.nodes()
                    }
                    total_wdeg = sum(wdeg.values())

                    if total_wdeg > 0 and len(wdeg) > 0:
                        k = max(1, int(len(wdeg) * 0.10))
                        top_k_wdeg = sum(sorted(wdeg.values(), reverse=True)[:k])
                        whale_dominance = top_k_wdeg / total_wdeg
                    else:
                        whale_dominance = 0

                    # Connectivity
                    H = G_day.to_undirected()
                    if H.number_of_nodes() > 0:
                        gcc_size = len(max(nx.connected_components(H), key=len))
                        num_components = nx.number_connected_components(H)
                    else:
                        gcc_size = 0
                        num_components = 0

                    # Volume
                    total_volume = sum(data.get("value", 0) for _, _, data in G_day.edges(data=True))
                    avg_transaction = total_volume / e if e > 0 else 0

                    daily_raws.append({
                        "raw": [
                            n,                 # num_nodes
                            e,                 # num_edges
                            avg_deg,           # average_degree
                            whale_dominance,   # weighted degree concentration
                            gcc_size,          # giant connected component size
                            num_components,    # number of connected components
                            total_volume,      # total transaction volume
                            avg_transaction    # average transaction size
                        ]
                    })
                else:
                    daily_feats.append({"mapper": [0, 0, 0, 0]})
                    daily_raws.append({"raw": [0, 0, 0, 0, 0, 0, 0, 0]})

            seq_tda.append(merge_dicts(daily_feats))
            seq_raw.append(merge_dicts(daily_raws))
            ws_date += pd.Timedelta(days=1)

        dataset_name_noext = os.path.splitext(file)[0]
        output_dir = os.path.join(GRAPHPULSE_RESULTS_DIR, task_folder, dataset_name_noext)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "seq_tda.txt"), "w", encoding="utf-8") as f_tda:
            json.dump(
                {"TDA_SEQUENCE": to_builtin(merge_dicts(seq_tda)), "LABELS": seq_labels},
                f_tda,
                indent=2
            )

        with open(os.path.join(output_dir, "seq_raw.txt"), "w", encoding="utf-8") as f_raw:
            json.dump(
                {"RAW_SEQUENCE": to_builtin(merge_dicts(seq_raw)), "LABELS": seq_labels},
                f_raw,
                indent=2
            )

        gc.collect()

    duration = time.perf_counter() - t0
    timings_writer.writerow({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": dataset_name,
        "variant": variant_name,
        "phase": "rnn_labeling",
        "alpha": "",
        "beta": "",
        "start_unix": f"{start_unix:.6f}",
        "end_unix": f"{time.time():.6f}",
        "seconds": f"{duration:.6f}",
    })
    print(f"[DONE] Labeling {variant_name} in {duration:.2f}s\n")


# ============================================================
# DRIVER ENTRY POINT
# ============================================================
overall_start = time.perf_counter()
timings_path = os.path.join(GRAPHPULSE_RESULTS_DIR, "run_times_sensitivity.csv")
timings_exists = os.path.exists(timings_path)

timings_f = open(timings_path, "a", newline="", encoding="utf-8")
timings_writer = csv.DictWriter(
    timings_f,
    fieldnames=[
        "current_time", "dataset", "variant", "phase",
        "alpha", "beta", "start_unix", "end_unix", "seconds"
    ],
)
if not timings_exists:
    timings_writer.writeheader()

for dataset_name in args.datasets:
    dataset_dir = os.path.join(RICCI_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        print(f"[WARN] Dataset folder not found: {dataset_dir}")
        continue

    dataset_prefix = dataset_name
    full_variant = f"{dataset_prefix}_full"

    full_csv_path = os.path.join(dataset_dir, f"{full_variant}.csv")
    if not os.path.exists(full_csv_path):
        print(f"[WARN] Missing base file: {full_csv_path}. Skipping dataset {dataset_name}.")
        continue

    print(f"\n[DATASET] {dataset_name} - base variant {full_variant}")

    test_datasets = {
        full_variant: {"over_lap": OVER_LAP, "n_cube": N_CUBE, "cls": CLS}
    }

    for i in range(1, BINS + 1):
        variant_name = f"{dataset_prefix}_bin{i}"
        variant_path = os.path.join(dataset_dir, f"{variant_name}.csv")
        if os.path.exists(variant_path):
            test_datasets[variant_name] = {
                "over_lap": OVER_LAP,
                "n_cube": N_CUBE,
                "cls": CLS,
            }
        else:
            print(f"[INFO] Missing optional bin file: {variant_path}")

    for variant_name, cfg in test_datasets.items():
        daily_tda_cache = precompute_all_daily_tda(
            dataset_name,
            variant_name,
            timings_writer,
            over_lap=cfg["over_lap"],
            n_cube=cfg["n_cube"],
            cls=cfg["cls"],
        )
        if daily_tda_cache is None:
            print(f"[SKIP] No TDA cache computed for {variant_name}")
            continue

        print(f"[RNN] {dataset_name}/{variant_name}.csv (basis: {full_variant}.csv)")
        create_time_series_rnn_sequence(
            f"{variant_name}.csv",
            dataset_name,
            f"{full_variant}.csv",
            variant_name,
            daily_tda_cache,
            timings_writer,
        )

timings_f.close()
print(f"\nAll datasets completed in {time.perf_counter() - overall_start:.2f} seconds.")

# ============================================================
# CREATE SUMMARY OUTPUT FILE (process_data_time.csv)
# ============================================================
process_time_path = os.path.join(GRAPHPULSE_RESULTS_DIR, "process_data_time.csv")

df = pd.read_csv(timings_path)
df["start_time"] = pd.to_datetime(df["start_unix"], unit="s", errors="coerce")
df["end_time"] = pd.to_datetime(df["end_unix"], unit="s", errors="coerce")
df["duration_sec"] = df["end_time"] - df["start_time"]
df["duration_sec"] = df["duration_sec"].dt.total_seconds().round(2)

summary = (
    df.groupby(["dataset", "variant"], as_index=False)
    .agg(
        start_datetime=("start_time", "min"),
        end_datetime=("end_time", "max"),
        total_duration_sec=("duration_sec", "sum")
    )
)

summary.to_csv(process_time_path, index=False)
print(f"[SAVED] Process timing summary written to {process_time_path}")

# ============================================================
# PRINT TOTAL RUNTIME
# ============================================================
total_runtime = time.perf_counter() - overall_start
hours = int(total_runtime // 3600)
minutes = int((total_runtime % 3600) // 60)
seconds = total_runtime % 60

print("\n============================================================")
print(f"TOTAL PIPELINE RUNTIME: {hours}h {minutes}m {seconds:.2f}s")
print("============================================================\n")