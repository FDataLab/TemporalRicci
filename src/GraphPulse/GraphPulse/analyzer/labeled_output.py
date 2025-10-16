
import csv
import os

import networkx as nx
import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import warnings
import time
from tqdm import tqdm
import gc

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
    # LABELING TASKS
    # ============================================================
    def label_task1(self, selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow):
        return 1 if (len(selectedNetworkInLbelingWindow) - len(selectedNetworkInGraphDataWindow)) > 0 else 0

    def label_task2(self, G, selectedNetwork, prevNetwork=None):
        volume_by_addr = (
            selectedNetwork.groupby("from")["value"].sum()
            .add(selectedNetwork.groupby("to")["value"].sum(), fill_value=0)
        )
        top_n = max(1, int(len(volume_by_addr) * 0.01))
        current_count = len(volume_by_addr.nlargest(top_n))

        if prevNetwork is not None:
            prev_volume = (
                prevNetwork.groupby("from")["value"].sum()
                .add(prevNetwork.groupby("to")["value"].sum(), fill_value=0)
            )
            prev_top_n = max(1, int(len(prev_volume) * 0.01))
            prev_count = len(prev_volume.nlargest(prev_top_n))
            return 1 if current_count > prev_count else 0
        return 0

    def label_task3(self, G, prevG=None):
        H = G.to_undirected()
        current_count = nx.number_connected_components(H) if len(H.nodes()) > 0 else 0
        if prevG is not None:
            prevH = prevG.to_undirected()
            prev_count = nx.number_connected_components(prevH) if len(prevH.nodes()) > 0 else 0
            return 1 if current_count > prev_count else 0
        return 0

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
                continue
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
                print(f"[DEBUG ⚠️] Day {current_date}: Mapper has {Xfilt.shape[0]} and {Xfilt.shape[1]}")
                self.daily_tda_cache[current_date] = {
                    f"default-overlap{kwargs.get('over_lap')}-cube{kwargs.get('n_cube')}-cls{kwargs.get('cls')}":
                        [0, 0, 0, 0]
                }
                continue

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
            # num_mapper_nodes = list(dailyFeatures.values())[0][0]
            # num_mapper_edges = list(dailyFeatures.values())[0][1]
            #print(f"[CACHE] Precomputed day {current_date}: {num_mapper_nodes} mapper nodes, {num_mapper_edges} mapper edges")

    # ============================================================
    # MAIN SEQUENCE CREATION
    # ============================================================
    def create_time_series_rnn_sequence(self, file, **kwargs):
        print(f"Processing {file}")
        windowSize, gap, labelWindowSize = 7, 3, 7

        selectedNetwork = pd.read_csv(os.path.join(self.timeseries_file_path, file))
        if selectedNetwork.shape[0] < 100:
            print(f"[SKIP] {file}: only {selectedNetwork.shape[0]} edges (<100).")
            return

        selectedNetwork['timestamp'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='timestamp')



        data_last_date = selectedNetwork['timestamp'].max()
        window_start_date = selectedNetwork['timestamp'].min()

        for task_id, task_label_func, task_folder in [
            (1, "task1", "Sequence_task1"),
            (2, "task2", "Sequence_task2"),
            (3, "task3", "Sequence_task3"),
        ]:
            print(f"\n=== Running {task_label_func} labeling for {file} ===")
            totalRnnSequenceData = []
            totalRnnLabelData = []
            ws_date = window_start_date

            num_windows = int((data_last_date - ws_date).days - (windowSize + gap + labelWindowSize))
            for _ in tqdm(range(num_windows), desc=f"{file} - {task_label_func}", leave=False):
                window_end_date = ws_date + dt.timedelta(days=windowSize)
                label_start_date = ws_date + dt.timedelta(days=windowSize + gap)
                label_end_date = label_start_date + dt.timedelta(days=labelWindowSize)

                selectedNetworkInGraphDataWindow = selectedNetwork[
                    (selectedNetwork['timestamp'] >= ws_date) &
                    (selectedNetwork['timestamp'] < window_end_date)
                    ]
                selectedNetworkInLabelingWindow = selectedNetwork[
                    (selectedNetwork['timestamp'] >= label_start_date) &
                    (selectedNetwork['timestamp'] < label_end_date)
                    ]

                G = nx.from_pandas_edgelist(
                    selectedNetworkInGraphDataWindow, 'from', 'to', ['value'],
                    create_using=nx.MultiDiGraph()
                )

                if task_id == 1:
                    label = self.label_task1(selectedNetworkInGraphDataWindow, selectedNetworkInLabelingWindow)
                elif task_id == 2:
                    label = self.label_task2(G, selectedNetworkInGraphDataWindow)
                else:
                    label = self.label_task3(G)

                # collect cached daily features
                daily_feats = []
                for day_offset in range(windowSize):
                    day = ws_date + dt.timedelta(days=day_offset)
                    if day in self.daily_tda_cache:
                        daily_feats.append(self.daily_tda_cache[day])
                if not daily_feats:
                    ws_date += dt.timedelta(days=1)
                    continue

                merged_daily = self.merge_dicts(daily_feats)
                totalRnnSequenceData.append(merged_daily)
                totalRnnLabelData.append(label)
                ws_date += dt.timedelta(days=1)

            if totalRnnSequenceData:
                total_merged_tda = self.merge_dicts(totalRnnSequenceData)
                final_tda = {"sequence": total_merged_tda, "label": totalRnnLabelData}
                directory = os.path.join(GRAPHPULSE_RESULTS_DIR, task_folder, file)
                os.makedirs(directory, exist_ok=True)
                tda_path = os.path.join(directory, "seq_tda.txt")
                with open(tda_path, 'wb') as f_tda:
                    pickle.dump(final_tda, f_tda)
                print(f"Saved {tda_path}")
            else:
                print(f"[ERROR] No valid TDA sequences for {file}, {task_label_func}")
            gc.collect()

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

    for dataset, cfg in tqdm(test_datasets.items(), desc="Datasets", total=len(test_datasets)):
        for task_id in [1, 2, 3]:
            task_name = f"task{task_id}"
            print(f"\n[RUN] Dataset: {dataset} | Task: {task_name}")
            start_iso = datetime.utcnow().isoformat()
            t0 = time.perf_counter()
            status = 'ok'
            err_msg = ''
            try:
                parser.create_time_series_rnn_sequence(f"{dataset}.csv", task_id=task_id, **cfg)
            except Exception as e:
                status = 'error'
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[ERROR] {dataset} {task_name}: {err_msg}")
            seconds = time.perf_counter() - t0
            end_iso = datetime.utcnow().isoformat()
            timings_writer.writerow({
                'dataset': dataset,
                'task': task_name,
                'start_iso': start_iso,
                'end_iso': end_iso,
                'seconds': f"{seconds:.6f}",
                'status': status,
                'error': err_msg
            })
            timings_f.flush()

    timings_f.close()
    overall_seconds = time.perf_counter() - overall_start
    print(f"\nAll done. Total runtime: {overall_seconds:.2f} seconds")
