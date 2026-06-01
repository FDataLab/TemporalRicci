import ast
import csv
import multiprocessing
import os
import shutil
from collections import defaultdict
import networkx as nx
import pandas as pd
import datetime as dt
from datetime import datetime
import contextlib, io, csv
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

from util.graph_util import from_networkx


# ============================================================
# CONFIGURATION
# ============================================================
DATASET_NAME = "BEPRO"
ALPHA = 3.00
BETA = 1.00

# base directories
RICCI_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../../../../RicciResults/ricci_values/{DATASET_NAME}")
)
GRAPHPULSE_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../GraphPulseResults")
)
os.makedirs(GRAPHPULSE_RESULTS_DIR, exist_ok=True)


class NetworkParser:
    # ============================================================
    # PATHS & PARAMETERS
    # ============================================================
    file_path = RICCI_RESULTS_DIR
    timeseries_file_path = RICCI_RESULTS_DIR
    timeseries_file_path_other = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../src/GraphPulse/GraphPulse/data/all_network/TimeSeries/Other/")
    )

    timeWindow = [7]
    networkValidationDuration = 20
    finalDataDuration = 5
    labelTreshholdPercentage = 10

    # ============================================================
    # LABELING TASKS
    # ============================================================
    def label_task1(self, selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow):
        """Task 1 – Network Growth Prediction"""
        return 1 if (len(selectedNetworkInLbelingWindow) - len(selectedNetworkInGraphDataWindow)) > 0 else 0

    def label_task2(self, G, selectedNetwork, prevNetwork=None):
        """Task 2 – Influential Node Count Prediction (Hydra-based, Binary)"""
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
            label = 1 if current_count > prev_count else 0
        else:
            label = 0
        return label

    def label_task3(self, G, prevG=None):
        """Task 3 – Connected Components Prediction (Hydra-based, Binary)"""
        H = G.to_undirected()
        current_count = nx.number_connected_components(H) if len(H.nodes()) > 0 else 0
        if prevG is not None:
            prevH = prevG.to_undirected()
            prev_count = nx.number_connected_components(prevH) if len(prevH.nodes()) > 0 else 0
            label = 1 if current_count > prev_count else 0
        else:
            label = 0
        return label

    # ============================================================
    # MAIN SEQUENCE CREATION
    # ============================================================
    def create_time_series_rnn_sequence(self, file, **kwargs):
        print(f"Processing {file}")
        windowSize, gap, lableWindowSize = 7, 3, 7

        selectedNetwork = pd.read_csv(os.path.join(self.timeseries_file_path, file))
        if selectedNetwork.shape[0] < 100:
            print(f"[SKIP] {file}: only {selectedNetwork.shape[0]} edges (<100).")
            return

        selectedNetwork['timestamp'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='timestamp')

        # Normalize transaction values
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())
        if max_transfer == min_transfer:
            selectedNetwork['value'] = 1.0
        else:
            selectedNetwork['value'] = selectedNetwork['value'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer)))
            )

        data_last_date = selectedNetwork['timestamp'].max()
        window_start_date = selectedNetwork['timestamp'].min()

        # Precompute static aggregates
        outgoing_weight_sum = selectedNetwork.groupby('from')['value'].sum()
        incoming_weight_sum = selectedNetwork.groupby('to')['value'].sum()
        outgoing_count = selectedNetwork.groupby('from')['value'].count()
        incoming_count = selectedNetwork.groupby('to')['value'].count()

        pool = multiprocessing.Pool(processes=os.cpu_count())

        for task_id, task_label_func, task_folder in [
            (1, "task1", "Sequence_task1"),
            (2, "task2", "Sequence_task2"),
            (3, "task3", "Sequence_task3"),
        ]:
            print(f"\n=== Running {task_label_func} labeling for {file} ===")
            totalRnnSequenceData = []
            totalRnnLabelData = []
            ws_date = window_start_date

            num_windows = int((data_last_date - ws_date).days - (windowSize + gap + lableWindowSize))
            for _ in tqdm(range(num_windows), desc=f"{file} - {task_label_func}", leave=False):
                window_end_date = ws_date + dt.timedelta(days=windowSize)
                label_start_date = ws_date + dt.timedelta(days=windowSize + gap)
                label_end_date = label_start_date + dt.timedelta(days=lableWindowSize)

                selectedNetworkInGraphDataWindow = selectedNetwork[
                    (selectedNetwork['timestamp'] >= ws_date) & (selectedNetwork['timestamp'] < window_end_date)
                ]
                selectedNetworkInLbelingWindow = selectedNetwork[
                    (selectedNetwork['timestamp'] >= label_start_date) &
                    (selectedNetwork['timestamp'] < label_end_date)
                ]

                # Graph construction
                G = nx.from_pandas_edgelist(
                    selectedNetworkInGraphDataWindow,
                    'from', 'to', ['value'],
                    create_using=nx.MultiDiGraph()
                )

                # Label selection
                if task_id == 1:
                    label = self.label_task1(selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow)
                elif task_id == 2:
                    label = self.label_task2(G, selectedNetworkInGraphDataWindow)
                else:
                    label = self.label_task3(G)

                # Node feature creation
                records = []
                for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                    records.append({
                        "nodeID": item["from"],
                        "outgoing_edge_weight_sum": outgoing_weight_sum.get(item['from'], 0),
                        "incoming_edge_weight_sum": incoming_weight_sum.get(item['from'], 0),
                        "outgoing_edge_count": outgoing_count.get(item['from'], 0),
                        "incoming_edge_count": incoming_count.get(item['from'], 0),
                    })
                node_features = pd.DataFrame(records).drop_duplicates('nodeID')

                # Extract TDA-based sequence
                timeWindowSequence = self.process_TDA_extracted_rnn_sequence(
                    selectedNetworkInGraphDataWindow, node_features, pool=pool, **kwargs)
                totalRnnSequenceData.append(timeWindowSequence)
                totalRnnLabelData.append(label)

                ws_date = ws_date + dt.timedelta(days=1)

            # === Merge and Save both TDA and RAW Sequences ===
            total_merged_tda = self.merge_dicts(totalRnnSequenceData)
            final_tda = {"sequence": total_merged_tda, "label": totalRnnLabelData}

            # Prepare RAW sequence (no TDA)
            totalRnnSequenceDataRaw = []
            ws_date_raw = window_start_date
            for _ in range(num_windows):
                window_end_date = ws_date_raw + dt.timedelta(days=windowSize)
                selectedNetworkInGraphDataWindow = selectedNetwork[
                    (selectedNetwork['timestamp'] >= ws_date_raw) &
                    (selectedNetwork['timestamp'] < window_end_date)
                ]
                try:
                    G_raw = nx.from_pandas_edgelist(
                        selectedNetworkInGraphDataWindow,
                        'from', 'to', ['value'],
                        create_using=nx.MultiDiGraph()
                    )
                    num_nodes = G_raw.number_of_nodes()
                    num_edges = G_raw.number_of_edges()
                    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
                    daily_raw_features = {"raw_features": [num_nodes, num_edges, avg_degree]}
                    totalRnnSequenceDataRaw.append(daily_raw_features)
                except Exception as e:
                    print(f"[RAW Skipped] {file} window {ws_date_raw}: {e}")
                ws_date_raw += dt.timedelta(days=1)

            total_merged_raw = self.merge_dicts(totalRnnSequenceDataRaw)
            final_raw = {"sequence": total_merged_raw, "label": totalRnnLabelData}

            # === Save outputs to GraphPulseResults ===
            directory = os.path.join(GRAPHPULSE_RESULTS_DIR, task_folder, file)
            os.makedirs(directory, exist_ok=True)
            tda_path = os.path.join(directory, "seq_tda.txt")
            raw_path = os.path.join(directory, "seq_raw.txt")

            with open(tda_path, 'wb') as f_tda:
                pickle.dump(final_tda, f_tda)
            with open(raw_path, 'wb') as f_raw:
                pickle.dump(final_raw, f_raw)

            print(f"Saved {tda_path} and {raw_path}")

            # Cleanup per task
            del totalRnnSequenceData, totalRnnSequenceDataRaw, totalRnnLabelData, node_features, G
            gc.collect()

            # Reset multiprocessing pool after each task
            pool.close()
            pool.join()
            pool = multiprocessing.Pool(processes=os.cpu_count())

        # Close pool after all tasks
        pool.close()
        pool.join()

    # ============================================================
    # TDA SEQUENCE + HELPER FUNCTIONS
    # ============================================================
    def process_TDA_extracted_rnn_sequence(self, timeFrameData, nodeFeatures, **kwargs):
        timeWindowSequence = []
        try:
            data_first_date = timeFrameData['timestamp'].min()
            data_last_date = timeFrameData['timestamp'].max()
            numberOfDays = (data_last_date - data_first_date).days
        except Exception as e:
            print(f"[SKIP] Could not parse timestamps: {e}")
            return {}

        start_date = data_first_date
        processingDay = 0

        while processingDay <= numberOfDays:
            daily_end_date = start_date + dt.timedelta(days=1)
            selectedDailyNetwork = timeFrameData[
                (timeFrameData['timestamp'] >= start_date) & (timeFrameData['timestamp'] < daily_end_date)
            ]

            daily_node_features = pd.DataFrame()
            for item in selectedDailyNetwork.to_dict(orient="records"):
                try:
                    to_match = nodeFeatures[nodeFeatures["nodeID"] == item["to"]]
                    from_match = nodeFeatures[nodeFeatures["nodeID"] == item["from"]]
                    if not to_match.empty:
                        new_row = pd.DataFrame({**{"nodeID": item["from"]},
                                                **to_match.drop("nodeID", axis=1).to_dict('records')[0]}, index=[0])
                        daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)
                    if not from_match.empty:
                        new_row = pd.DataFrame({**{"nodeID": item["to"]},
                                                **from_match.drop("nodeID", axis=1).to_dict('records')[0]}, index=[0])
                        daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)
                except Exception:
                    continue

            daily_node_features = daily_node_features.drop_duplicates(subset=['nodeID'])

            try:
                Xfilt = daily_node_features.drop(columns=['nodeID'], errors='ignore')
                if Xfilt.shape[0] == 0 or Xfilt.shape[1] == 0:
                    start_date += dt.timedelta(days=1)
                    processingDay += 1
                    continue

                mapper = km.KeplerMapper()
                scaler = MinMaxScaler(feature_range=(0, 1))
                Xfilt = scaler.fit_transform(Xfilt)
                lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())

                with multiprocessing.Pool(processes=2) as pool:
                    result = pool.apply_async(
                        self.TDA_process,
                        (mapper, lens, Xfilt,
                         kwargs.get('over_lap'),
                         kwargs.get('n_cube'),
                         kwargs.get('cls'))
                    )
                    dailyFeatures = result.get()

                if dailyFeatures and isinstance(dailyFeatures, dict):
                    timeWindowSequence.append(dailyFeatures)

            except Exception as e:
                print(f"[TDA Skipped] Day {processingDay} failed: {e}")

            start_date += dt.timedelta(days=1)
            processingDay += 1

        if not timeWindowSequence:
            print("[Skip] No daily features extracted.")
            return {}

        return self.merge_dicts(timeWindowSequence)

    def TDA_process(self, mapper, lens, Xfilt, per_overlap, n_cubes, cls):
        dailyTdaGraph = mapper.map(
            lens,
            Xfilt,
            clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap))
        numberOfNodes = len(dailyTdaGraph['nodes'])
        numberOfEdges = sum(len(edges) for edges in dailyTdaGraph['links'].values())
        try:
            maxClusterSize = len(
                dailyTdaGraph["nodes"][max(dailyTdaGraph["nodes"], key=lambda k: len(dailyTdaGraph["nodes"][k]))])
            cluster_sizes = [len(nodes) for nodes in dailyTdaGraph["nodes"].values()]
            average_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
            edge_weights = defaultdict(dict)
            for source_node, target_nodes in dailyTdaGraph['links'].items():
                for target_node in target_nodes:
                    common_indexes = len(
                        set(dailyTdaGraph['nodes'][source_node]) & set(dailyTdaGraph['nodes'][target_node]))
                    edge_weights[source_node][target_node] = common_indexes
            total_edge_weights = sum(
                weight for target_weights in edge_weights.values() for weight in target_weights.values())
            total_edges = sum(len(target_weights) for target_weights in edge_weights.values())
            average_edge_weight = total_edge_weights / total_edges if total_edges > 0 else 0
        except Exception:
            maxClusterSize = 0
            average_cluster_size = 0
            average_edge_weight = 0

        return {"overlap{}-cube{}-cls{}".format(per_overlap, n_cubes, cls): [
            numberOfNodes, numberOfEdges, maxClusterSize, average_cluster_size, average_edge_weight]}

    def merge_dicts(self, list_of_dicts):
        merged_dict = {}
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                merged_dict.setdefault(key, []).append(value)
        return merged_dict


if __name__ == '__main__':
    from datetime import datetime

    # Timing file location
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

    # Build dataset variants (base + bins 1–10)
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
