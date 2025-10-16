
import csv
import os

import pandas as pd
import datetime as dt

import numpy as np
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler

import sys
import warnings
import time
from tqdm import tqdm


warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================
# CONFIGURATION
# ============================================================
DATASET_NAME = "BEPRO"
ALPHA = 3.00
BETA = 1.00

RICCI_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../../../../RicciResults/ricci_values/{DATASET_NAME}")
)
GRAPHPULSE_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../GraphPulseResults")
)
os.makedirs(GRAPHPULSE_RESULTS_DIR, exist_ok=True)

# ============================================================
# CLASS
# ============================================================
class NetworkParser:
    timeseries_file_path = RICCI_RESULTS_DIR
    daily_tda_cache = {}



    # ============================================================
    # PRECOMPUTE DAILY MAPPERS
    # ============================================================
    def precompute_all_daily_tda(self, selectedNetwork, **kwargs):

        if not np.issubdtype(selectedNetwork['timestamp'].dtype, np.datetime64):
            selectedNetwork['timestamp'] = pd.to_datetime(selectedNetwork['timestamp'], errors='coerce')

        all_dates = sorted(selectedNetwork['timestamp'].dt.floor('D').unique())

        for current_date in tqdm(all_dates, desc="[PRECOMPUTE DAILY MAPPERS]"):
            daily_end = current_date + dt.timedelta(days=1)
            selectedDailyNetwork = selectedNetwork[
                (selectedNetwork['timestamp'] >= current_date) &
                (selectedNetwork['timestamp'] < daily_end)
            ]
            num_edges = len(selectedDailyNetwork)
            num_nodes = len(set(selectedDailyNetwork['from']).union(set(selectedDailyNetwork['to'])))


            if num_edges < 3 or num_nodes < 2:
                print(f"[DEBUG ⚠️] Day {current_date}: graph has {num_nodes} nodes and {num_edges} edges")

            #print(f"[DEBUG] Day {current_date}: graph has {num_nodes} nodes and {num_edges} edges")
            outgoing_weight_sum = selectedDailyNetwork.groupby('from')['value'].sum()
            incoming_weight_sum = selectedDailyNetwork.groupby('to')['value'].sum()
            outgoing_count = selectedDailyNetwork.groupby('from')['value'].count()
            incoming_count = selectedDailyNetwork.groupby('to')['value'].count()

            records = []
            for node in set(selectedDailyNetwork['from']).union(set(selectedDailyNetwork['to'])):
                records.append({
                    "nodeID": node,
                    "outgoing_edge_weight_sum": outgoing_weight_sum.get(node, 0),
                    "incoming_edge_weight_sum": incoming_weight_sum.get(node, 0),
                    "outgoing_edge_count": outgoing_count.get(node, 0),
                    "incoming_edge_count": incoming_count.get(node, 0)
                })
            nodeFeatures = pd.DataFrame(records)

            Xfilt = nodeFeatures.drop(columns=['nodeID'], errors='ignore')
            if Xfilt.shape[0] < 3 or Xfilt.shape[1] == 0:
                print(f"[DEBUG ⚠️] Day {current_date}: Mapper has {Xfilt.shape[0]} data points")
                maxClusterSize = num_nodes
                average_cluster_size = num_nodes
                self.daily_tda_cache[current_date] = {
                    f"default-overlap{kwargs.get('over_lap')}-cube{kwargs.get('n_cube')}-cls{kwargs.get('cls')}":
                        [num_nodes, num_edges, maxClusterSize, average_cluster_size]
                }
            else:
                mapper = km.KeplerMapper()
                scaler = MinMaxScaler(feature_range=(0, 1))
                Xfilt = scaler.fit_transform(Xfilt)
                perplexity = max(2, min(30, Xfilt.shape[0] // 3))
                try:
                    lens = sklearn.manifold.TSNE(perplexity=perplexity, init="random", max_iter=500).fit_transform(Xfilt)
                except Exception:
                    lens = Xfilt

                dailyFeatures = self.TDA_process(mapper, lens, Xfilt,
                                                 kwargs.get('over_lap'),
                                                 kwargs.get('n_cube'),
                                                 kwargs.get('cls'))
                self.daily_tda_cache[current_date] = dailyFeatures
                #print(dailyFeatures)
                # num_mapper_nodes = list(dailyFeatures.values())[0][0]
                # num_mapper_edges = list(dailyFeatures.values())[0][1]
                #print(f"[CACHE] Precomputed day {current_date}: {num_mapper_nodes} mapper nodes, {num_mapper_edges} mapper edges")

    # ============================================================
    # MAIN SEQUENCE CREATION
    # ============================================================

    # ============================================================
    # TDA PROCESSOR
    # ============================================================
    def TDA_process(self, mapper, lens, Xfilt, per_overlap, n_cubes, cls):
        dailyTdaGraph = mapper.map(
            lens,
            Xfilt,
            clusterer=sklearn.cluster.KMeans(n_clusters=min(cls, max(1, Xfilt.shape[0] // 2)), random_state=1618033),
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap))
        numberOfNodes = len(dailyTdaGraph['nodes'])
        numberOfEdges = sum(len(edges) for edges in dailyTdaGraph['links'].values())
        maxClusterSize = len(dailyTdaGraph["nodes"][max(dailyTdaGraph["nodes"], key=lambda k: len(dailyTdaGraph["nodes"][k]))]) if dailyTdaGraph["nodes"] else 0
        cluster_sizes = [len(nodes) for nodes in dailyTdaGraph["nodes"].values()] if dailyTdaGraph["nodes"] else []
        average_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        return {f"overlap{per_overlap}-cube{n_cubes}-cls{cls}": [numberOfNodes, numberOfEdges, maxClusterSize, average_cluster_size]}

    # ============================================================
    # MERGE DICTS
    # ============================================================
    def merge_dicts(self, list_of_dicts):
        merged_dict = {}
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                merged_dict.setdefault(key, []).append(value)
        return merged_dict

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    TIMINGS_PATH = os.path.join(GRAPHPULSE_RESULTS_DIR, f"{DATASET_NAME}_run_times.csv")
    os.makedirs(os.path.dirname(TIMINGS_PATH), exist_ok=True)

    timings_exists = os.path.exists(TIMINGS_PATH)
    timings_f = open(TIMINGS_PATH, 'a', newline='', encoding='utf-8')
    timings_writer = csv.DictWriter(
        timings_f,
        fieldnames=['dataset', 'task', 'start_iso', 'end_iso', 'seconds', 'status', 'error']
    )
    if not timings_exists:
        timings_writer.writeheader()

    overall_start = time.perf_counter()
    parser = NetworkParser()

    dataset_prefix = f"{DATASET_NAME}_TFR_a{ALPHA:.2f}_b{BETA:.2f}"
    test_datasets = {dataset_prefix: {"over_lap": 0.2, "n_cube": 2, "cls": 5}}
    for i in range(1, 11):
        test_datasets[f"{dataset_prefix}_bin{i}"] = {"over_lap": 0.2, "n_cube": 2, "cls": 5}

    for dataset, cfg in tqdm(test_datasets.items(), desc="Datasets", total=len(test_datasets), miniters= max(25, len(test_datasets)//5000),maxinterval=200, disable=True):
        # daily TDA precomputation (causal, one per day)
        csv_path = os.path.join(parser.timeseries_file_path, f"{dataset}.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] {dataset}: file not found.")
            continue

        print(f"[LOAD] Reading {csv_path}")
        selectedNetwork = pd.read_csv(csv_path)
        if selectedNetwork.shape[0] < 100:
            print(f"[SKIP] {dataset}: only {selectedNetwork.shape[0]} edges (<100).")
            continue

        # Ensure timestamp is datetime
        if not np.issubdtype(selectedNetwork['timestamp'].dtype, np.datetime64):
            selectedNetwork['timestamp'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s', errors='coerce')

        selectedNetwork['timestamp'] = selectedNetwork['timestamp'].dt.floor('D')  # Ensure proper daily granularity

        # Precompute daily TDA before tasks
        print(f"\n[PRECOMPUTE] Running TDA precomputation for {dataset}")
        parser.precompute_all_daily_tda(selectedNetwork, **cfg)


    timings_f.close()
    overall_seconds = time.perf_counter() - overall_start
    print(f"\nAll done. Total runtime: {overall_seconds:.2f} seconds")
