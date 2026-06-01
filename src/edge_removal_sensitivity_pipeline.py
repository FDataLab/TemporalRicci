# #!/usr/bin/env python3
# """
# edge_removal_sensitivity_pipeline.py
# ====================================
#
# Single end-to-end pipeline for edge-removal sensitivity analysis.
#
# For each selected dataset, this script:
#   1. Loads the raw temporal edge CSV.
#   2. Computes Temporal Forman--Ricci curvature independently inside
#      non-overlapping 7-day windows.
#   3. Creates sparse graph variants by removing the lowest-curvature
#      rho% of edges per window.
#   4. Runs the GraphPulse-style daily TDA + raw feature processing.
#   5. Trains the RNN model for Task 1 only.
#   6. Saves ROC-AUC results and the sensitivity plot.
#
# Sensitivity setting:
#   x-axis = removed edges (%)
#   y-axis = ROC-AUC for Task 1
#
# Default datasets:
#   ADX, BAG, BEPRO, DERC, DFRC, DINO, ETH2X-FLI, EVERMOON, GLM, HOICHI
#
# Expected raw input files:
#   PROJECT_ROOT/data/<DATASET>.csv
#
# Expected columns after flexible renaming:
#   from, to, timestamp, value
#
# Outputs:
#   PROJECT_ROOT/RicciResults/removal_sensitivity/<DATASET>/...
#   PROJECT_ROOT/GraphPulseResultsRemovalSensitivity/...
#   PROJECT_ROOT/results/removal_ratio_sensitivity_task1.png
#   PROJECT_ROOT/results/removal_ratio_sensitivity_task1.csv
# """
#
# # ============================================================
# # Imports
# # ============================================================
#
# import os
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# os.environ["WANDB_MODE"] = "disabled"
#
# import gc
# import json
# import time
# import warnings
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
#
# import numpy as np
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# import kmapper as km
# import sklearn
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import roc_auc_score
#
# import tensorflow as tf
# from keras.callbacks import Callback
# from keras.layers import LSTM, Dense, GRU
# from keras.models import Sequential
#
# warnings.filterwarnings("ignore")
#
#
# # ============================================================
# # CONFIG
# # ============================================================
#
# PROJECT_ROOT = Path(r"C:\Users\azadp\PycharmProjects\TemporalRicci")
#
# DATASETS = [
#     "ADX",
#     "BAG",
#     "BEPRO",
#     "DERC",
#     "DFRC",
#     "DINO",
#     "ETH2X-FLI",
#     "EVERMOON",
#     "GLM",
#     "HOICHI",
# ]
#
# INPUT_DIR = PROJECT_ROOT / "data"
#
# RICCI_OUT_ROOT = PROJECT_ROOT / "RicciResults" / "removal_sensitivity"
# PROCESS_OUT_ROOT = PROJECT_ROOT / "GraphPulseResultsRemovalSensitivity"
# RESULTS_DIR = PROJECT_ROOT / "results"
#
# RICCI_OUT_ROOT.mkdir(parents=True, exist_ok=True)
# PROCESS_OUT_ROOT.mkdir(parents=True, exist_ok=True)
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#
# RESULTS_CSV = RESULTS_DIR / "removal_ratio_sensitivity_task1.csv"
# PLOT_PATH = RESULTS_DIR / "removal_ratio_sensitivity_task1.png"
# RUNTIME_CSV = RESULTS_DIR / "removal_ratio_sensitivity_runtime.csv"
#
# # Sensitivity values: percentage of lowest-curvature edges removed
# REMOVAL_RATIOS = [0, 20, 40, 60, 80, 90]
#
# # Temporal curvature setting
# TAU_DAYS = 1.0
# WINDOW_DAYS = 7
# WINDOW_STEP_DAYS = 7
# CURV_COL = "Temporal Forman-Ricci value"
#
# # GraphPulse-style temporal settings
# WINDOW_SIZE = 7
# GAP = 3
# LABEL_WINDOW_SIZE = 7
#
# # Mapper/TDA settings from your processing script
# OVER_LAP = 0.2
# N_CUBE = 2
# CLS = 5
#
# # RNN settings from your rnn_methods.py
# EPOCHS = 10
# LEARNING_RATE = 1e-4
# NORMALIZER_MODE = "per_column"
# SEED = 1
#
# # Numerical constants
# EPS = 1e-12
# SECONDS_PER_DAY = 86400.0
# MIN_POSITIVE_WEIGHT = 1e-12
# VALUE_SCALE_QUANTILE = 0.99
# MIN_NORMALIZED_EDGE_WEIGHT = 0.05
# MAX_NORMALIZED_EDGE_WEIGHT = 1.0
# TIMESTAMP_TIE_BREAK = 1e-9
#
#
# # ============================================================
# # Utility Helpers
# # ============================================================
#
# def now_str() -> str:
#     return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#
#
# def dataset_variants(dataset: str) -> List[str]:
#     return list(dict.fromkeys([
#         dataset,
#         dataset.upper(),
#         dataset.lower(),
#         dataset.replace("-", "_"),
#         dataset.replace("-", "_").upper(),
#         dataset.replace("-", "_").lower(),
#         dataset.replace("_", "-"),
#         dataset.replace("_", "-").upper(),
#         dataset.replace("_", "-").lower(),
#     ]))
#
#
# def find_input_csv(dataset: str) -> Path:
#     for name in dataset_variants(dataset):
#         path = INPUT_DIR / f"{name}.csv"
#         if path.exists():
#             return path
#
#     raise FileNotFoundError(
#         f"Input CSV not found for dataset {dataset}. Tried variants in {INPUT_DIR}"
#     )
#
#
# def load_edge_csv(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#
#     df = df.rename(columns={
#         "source": "from",
#         "target": "to",
#         "from_address": "from",
#         "to_address": "to",
#         "block_timestamp": "timestamp",
#         "amount": "value",
#         "weight": "value",
#     })
#
#     required = {"from", "to", "timestamp", "value"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing columns in {path}: {missing}")
#
#     df = df[["from", "to", "timestamp", "value"]].copy()
#     df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
#     df = df.dropna(subset=["timestamp"])
#     df["timestamp"] = df["timestamp"].astype(np.int64)
#     df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
#     df["_date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.floor("D")
#     df = df.dropna(subset=["_date"])
#     return df
#
#
# def reset_random_seeds(seed: int = SEED) -> None:
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#
#
# # ============================================================
# # TFRC Computation
# # Adapted from TempRicciWindowed.py
# # ============================================================
#
# def compute_tfrc_window(win: pd.DataFrame, tau_days: float) -> np.ndarray:
#     n_edges = len(win)
#     if n_edges == 0:
#         return np.array([], dtype=np.float64)
#
#     tau_days = max(float(tau_days), EPS)
#
#     frm_raw = win["frm"].to_numpy()
#     to_raw = win["to"].to_numpy()
#
#     _, encoded = np.unique(
#         np.concatenate([frm_raw, to_raw]),
#         return_inverse=True
#     )
#
#     frm_arr = encoded[:n_edges].astype(np.int32)
#     to_arr = encoded[n_edges:].astype(np.int32)
#     n_nodes = int(max(frm_arr.max(), to_arr.max()) + 1) if n_edges else 0
#
#     val_arr_raw = win["value"].to_numpy(dtype=np.float64)
#     val_arr_raw = np.where(val_arr_raw > 0.0, val_arr_raw, MIN_POSITIVE_WEIGHT)
#
#     log_vals = np.log1p(val_arr_raw)
#     if n_edges > 1:
#         scale = float(np.quantile(log_vals, VALUE_SCALE_QUANTILE))
#     else:
#         scale = float(log_vals.max())
#     scale = max(scale, EPS)
#
#     val_arr = log_vals / scale
#     val_arr = np.clip(val_arr, MIN_NORMALIZED_EDGE_WEIGHT, MAX_NORMALIZED_EDGE_WEIGHT)
#
#     ts_arr_raw = win["timestamp"].to_numpy(dtype=np.float64)
#     ts_arr = (ts_arr_raw - ts_arr_raw.min()) / SECONDS_PER_DAY
#     order = np.arange(n_edges, dtype=np.float64)
#     ts_eff = ts_arr + order * TIMESTAMP_TIE_BREAK
#
#     sort_idx = np.argsort(ts_eff, kind="stable")
#     ts_sorted = ts_eff[sort_idx]
#     frm_sorted = frm_arr[sort_idx]
#     to_sorted = to_arr[sort_idx]
#     val_sorted = val_arr[sort_idx]
#
#     sqrt_val = np.sqrt(np.maximum(val_arr, EPS))
#     sqrt_val_sorted = np.sqrt(np.maximum(val_sorted, EPS))
#
#     strength = (
#         np.bincount(frm_arr, weights=val_arr, minlength=n_nodes) +
#         np.bincount(to_arr, weights=val_arr, minlength=n_nodes)
#     )
#
#     node_weight = np.log1p(strength)
#     node_weight = np.maximum(node_weight, EPS)
#
#     S_arr = val_arr * (
#         1.0 / node_weight[frm_arr] +
#         1.0 / node_weight[to_arr]
#     )
#
#     out_sorted_pos: List[List[int]] = [[] for _ in range(n_nodes)]
#     for pos in range(n_edges):
#         out_sorted_pos[frm_sorted[pos]].append(pos)
#
#     D_u = np.zeros(n_edges, dtype=np.float64)
#     D_v = np.zeros(n_edges, dtype=np.float64)
#
#     for i in range(n_edges):
#         u = frm_arr[i]
#         v = to_arr[i]
#         w_e = val_arr[i]
#         t_i = ts_eff[i]
#         sv_e = sqrt_val[i]
#
#         pos_list = out_sorted_pos[u]
#         if pos_list:
#             pos_arr = np.array(pos_list, dtype=np.int64)
#             pos_arr = pos_arr[to_sorted[pos_arr] != v]
#             if len(pos_arr) > 0:
#                 dt_days = np.maximum(np.abs(ts_sorted[pos_arr] - t_i), EPS)
#                 K = np.exp(-dt_days / tau_days)
#                 denom = np.maximum(sv_e * sqrt_val_sorted[pos_arr], EPS)
#                 D_u[i] = w_e * float(np.sum(K / denom)) / float(len(pos_arr))
#
#         pos_list = out_sorted_pos[v]
#         if pos_list:
#             pos_arr = np.array(pos_list, dtype=np.int64)
#             pos_arr = pos_arr[to_sorted[pos_arr] != u]
#             if len(pos_arr) > 0:
#                 dt_days = np.maximum(np.abs(ts_sorted[pos_arr] - t_i), EPS)
#                 K = np.exp(-dt_days / tau_days)
#                 denom = np.maximum(sv_e * sqrt_val_sorted[pos_arr], EPS)
#                 D_v[i] = w_e * float(np.sum(K / denom)) / float(len(pos_arr))
#
#     D_arr = 0.5 * (D_u + D_v)
#     return S_arr - D_arr
#
#
# def compute_tfrc_dataset(dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
#     df = raw_df.rename(columns={"from": "frm"}).copy()
#
#     date_min = df["_date"].min()
#     date_max = df["_date"].max()
#
#     window_td = pd.Timedelta(days=WINDOW_DAYS)
#     step_td = pd.Timedelta(days=WINDOW_STEP_DAYS)
#     window_starts = pd.date_range(date_min, date_max, freq=step_td)
#
#     chunks = []
#
#     for ws in tqdm(window_starts, desc=f"TFRC windows {dataset}"):
#         we = ws + window_td
#         mask = (df["_date"] >= ws) & (df["_date"] < we)
#         win_df = df.loc[mask].reset_index(drop=True)
#
#         if len(win_df) == 0:
#             continue
#
#         tfrc = compute_tfrc_window(win_df, TAU_DAYS)
#
#         chunk = win_df[["frm", "to", "timestamp", "value"]].copy()
#         chunk[CURV_COL] = tfrc
#         chunk["window_start"] = ws.strftime("%Y-%m-%d")
#         chunks.append(chunk)
#
#     if not chunks:
#         raise ValueError(f"No non-empty TFRC windows for {dataset}")
#
#     full = pd.concat(chunks, ignore_index=True)
#     full = full.rename(columns={"frm": "from"})
#     return full
#
#
# def make_removal_variant(full_tfrc: pd.DataFrame, removal_pct: int) -> pd.DataFrame:
#     """
#     Remove the lowest-curvature removal_pct fraction from each window.
#     removal_pct = 0 gives the full graph.
#     """
#     if removal_pct == 0:
#         return full_tfrc[["from", "to", "timestamp", "value"]].copy()
#
#     keep_ratio = max(0.0, min(1.0, 1.0 - removal_pct / 100.0))
#     kept_chunks = []
#
#     for _, grp in full_tfrc.groupby("window_start", sort=False):
#         n = len(grp)
#         if n == 0:
#             continue
#         n_keep = max(1, int(np.ceil(n * keep_ratio)))
#         kept = grp.sort_values(CURV_COL, ascending=False).head(n_keep)
#         kept_chunks.append(kept[["from", "to", "timestamp", "value"]])
#
#     if not kept_chunks:
#         return full_tfrc.iloc[0:0][["from", "to", "timestamp", "value"]].copy()
#
#     return pd.concat(kept_chunks, ignore_index=True)
#
#
# # ============================================================
# # GraphPulse-style Processing for Task 1
# # Adapted from process_dataset.py
# # ============================================================
#
# def mapper_features_for_day(sub_df: pd.DataFrame) -> List[float]:
#     num_edges = len(sub_df)
#     num_nodes = len(set(sub_df["from"]).union(set(sub_df["to"])))
#
#     if num_edges < 3 or num_nodes < 2:
#         return [num_nodes, num_edges, num_nodes, num_nodes]
#
#     outgoing_wsum = sub_df.groupby("from")["value"].sum()
#     incoming_wsum = sub_df.groupby("to")["value"].sum()
#     outgoing_cnt = sub_df.groupby("from")["value"].count()
#     incoming_cnt = sub_df.groupby("to")["value"].count()
#
#     nodes = set(sub_df["from"]).union(set(sub_df["to"]))
#     records = [{
#         "nodeID": n,
#         "outgoing_edge_weight_sum": outgoing_wsum.get(n, 0),
#         "incoming_edge_weight_sum": incoming_wsum.get(n, 0),
#         "outgoing_edge_count": outgoing_cnt.get(n, 0),
#         "incoming_edge_count": incoming_cnt.get(n, 0),
#     } for n in nodes]
#
#     X = pd.DataFrame(records).drop(columns=["nodeID"], errors="ignore")
#     if X.shape[0] < 3:
#         return [num_nodes, num_edges, num_nodes, num_nodes]
#
#     mapper = km.KeplerMapper()
#     X_scaled = MinMaxScaler((0, 1)).fit_transform(X)
#     perplexity = max(2, min(30, X_scaled.shape[0] // 3))
#
#     try:
#         lens = sklearn.manifold.TSNE(
#             perplexity=perplexity,
#             init="random",
#             random_state=42,
#             max_iter=500,
#         ).fit_transform(X_scaled)
#     except Exception:
#         lens = X_scaled
#
#     mapper_graph = mapper.map(
#         lens,
#         X_scaled,
#         clusterer=sklearn.cluster.KMeans(
#             n_clusters=min(CLS, max(1, X_scaled.shape[0] // 2)),
#             random_state=42,
#         ),
#         cover=km.Cover(n_cubes=N_CUBE, perc_overlap=OVER_LAP),
#     )
#
#     nodes_in_map_graph = len(mapper_graph["nodes"])
#     edges_in_map_graph = sum(len(v) for v in mapper_graph["links"].values())
#     cluster_sizes = [len(v) for v in mapper_graph["nodes"].values()]
#     avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
#     max_size = max(cluster_sizes) if cluster_sizes else 0
#
#     return [nodes_in_map_graph, edges_in_map_graph, max_size, avg_size]
#
#
# def raw_features_cumulative(df_day_window: pd.DataFrame) -> List[float]:
#     G_day = nx.from_pandas_edgelist(
#         df_day_window,
#         "from",
#         "to",
#         ["value"],
#         create_using=nx.MultiDiGraph(),
#     )
#
#     n = G_day.number_of_nodes()
#     e = G_day.number_of_edges()
#
#     avg_deg = 0 if n == 0 else sum(dict(G_day.to_undirected().degree()).values()) / n
#
#     wdeg = {
#         node: sum(data.get("value", 0) for _, _, data in G_day.edges(node, data=True))
#         for node in G_day.nodes()
#     }
#     total_wdeg = sum(wdeg.values())
#
#     if total_wdeg > 0 and len(wdeg) > 0:
#         k = max(1, int(len(wdeg) * 0.10))
#         top_k_wdeg = sum(sorted(wdeg.values(), reverse=True)[:k])
#         whale_dominance = top_k_wdeg / total_wdeg
#     else:
#         whale_dominance = 0
#
#     H = G_day.to_undirected()
#     if H.number_of_nodes() > 0:
#         gcc_size = len(max(nx.connected_components(H), key=len))
#         num_components = nx.number_connected_components(H)
#     else:
#         gcc_size = 0
#         num_components = 0
#
#     total_volume = sum(data.get("value", 0) for _, _, data in G_day.edges(data=True))
#     avg_transaction = total_volume / e if e > 0 else 0
#
#     return [
#         n,
#         e,
#         avg_deg,
#         whale_dominance,
#         gcc_size,
#         num_components,
#         total_volume,
#         avg_transaction,
#     ]
#
#
# def label_task1(df_data_window: pd.DataFrame, df_label_window: pd.DataFrame) -> int:
#     return int(len(df_label_window) > len(df_data_window))
#
#
# def build_task1_sequences(
#     variant_df: pd.DataFrame,
#     basis_df: pd.DataFrame,
# ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
#     """
#     Features are computed from variant_df.
#     Labels are computed from basis_df (full graph), matching your existing RNN setup.
#     """
#     df = variant_df.copy()
#     df_basis = basis_df.copy()
#
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.floor("D")
#     df_basis["timestamp"] = pd.to_datetime(df_basis["timestamp"], unit="s", errors="coerce").dt.floor("D")
#     df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
#     df_basis = df_basis.dropna(subset=["timestamp"]).sort_values("timestamp")
#
#     df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0).astype(float)
#     df_basis["value"] = pd.to_numeric(df_basis["value"], errors="coerce").fillna(0.0).astype(float)
#
#     start_date = df_basis["timestamp"].min()
#     last_date = df_basis["timestamp"].max()
#
#     if pd.isna(start_date) or pd.isna(last_date):
#         raise ValueError("Invalid date range for sequence generation")
#
#     num_windows = max(
#         0,
#         int((last_date - start_date).days - (WINDOW_SIZE + GAP + LABEL_WINDOW_SIZE)),
#     )
#
#     if num_windows <= 0:
#         raise ValueError("Not enough temporal span to create task sequences")
#
#     # Precompute daily TDA features for variant graph
#     all_dates = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="D")
#     daily_tda_cache: Dict[pd.Timestamp, List[float]] = {}
#
#     for current_date in tqdm(all_dates, desc="Daily TDA", leave=False):
#         next_day = current_date + pd.Timedelta(days=1)
#         sub_df = df[(df["timestamp"] >= current_date) & (df["timestamp"] < next_day)]
#         daily_tda_cache[pd.Timestamp(current_date)] = mapper_features_for_day(sub_df)
#
#     seq_tda = []
#     seq_raw = []
#     seq_labels = []
#     ws_date = start_date
#
#     for _ in tqdm(range(num_windows), desc="Task1 windows", leave=False):
#         w_end = ws_date + pd.Timedelta(days=WINDOW_SIZE)
#         l_start = ws_date + pd.Timedelta(days=WINDOW_SIZE + GAP)
#         l_end = l_start + pd.Timedelta(days=LABEL_WINDOW_SIZE)
#
#         # Labels from full/basis graph
#         basis_win = df_basis[(df_basis["timestamp"] >= ws_date) & (df_basis["timestamp"] < w_end)]
#         basis_label = df_basis[(df_basis["timestamp"] >= l_start) & (df_basis["timestamp"] < l_end)]
#         seq_labels.append(label_task1(basis_win, basis_label))
#
#         # Features from sparse/full variant
#         daily_tda = []
#         daily_raw = []
#
#         for i in range(WINDOW_SIZE):
#             key_date = pd.Timestamp(ws_date + pd.Timedelta(days=i)).floor("D")
#             daily_tda.append(daily_tda_cache.get(key_date, [0, 0, 0, 0]))
#
#             df_win_day = df[
#                 (df["timestamp"] >= ws_date) &
#                 (df["timestamp"] < ws_date + pd.Timedelta(days=i + 1))
#             ]
#             daily_raw.append(raw_features_cumulative(df_win_day))
#
#         seq_tda.append(daily_tda)
#         seq_raw.append(daily_raw)
#         ws_date += pd.Timedelta(days=1)
#
#     tda_np = np.array(seq_tda, dtype=np.float32)
#     raw_np = np.array(seq_raw, dtype=np.float32)
#     y = np.array(seq_labels, dtype=np.float32).reshape(-1, 1)
#
#     X = normalize_and_merge(tda_np, raw_np)
#
#     lbl0 = int((y == 0).sum())
#     lbl1 = int((y == 1).sum())
#
#     diag = {
#         "num_samples": len(y),
#         "label0": lbl0,
#         "label1": lbl1,
#         "label0_rate": lbl0 / max(len(y), 1),
#         "label1_rate": lbl1 / max(len(y), 1),
#     }
#
#     return X, y, diag
#
#
# def normalize_and_merge(tda_np: np.ndarray, raw_np: np.ndarray) -> np.ndarray:
#     def norm(arr: np.ndarray) -> np.ndarray:
#         if NORMALIZER_MODE == "per_column":
#             min_v = np.min(arr, axis=(0, 1), keepdims=True)
#             max_v = np.max(arr, axis=(0, 1), keepdims=True)
#         else:
#             min_v = np.min(arr)
#             max_v = np.max(arr)
#         scaled = (arr - min_v) / (max_v - min_v + 1e-10)
#         return np.nan_to_num(scaled)
#
#     return np.concatenate((norm(tda_np), norm(raw_np)), axis=2)
#
#
# # ============================================================
# # RNN Training
# # Adapted from rnn_methods.py
# # ============================================================
#
# class AUCCallback(Callback):
#     def __init__(self, validation_data):
#         super().__init__()
#         self.validation_data = validation_data
#         self.auc_scores: List[float] = []
#
#     def on_epoch_end(self, epoch, logs=None):
#         x_val, y_val = self.validation_data
#         y_pred = self.model.predict(x_val, verbose=0)
#         try:
#             auc_score = roc_auc_score(y_val, y_pred)
#         except ValueError:
#             auc_score = float("nan")
#         self.auc_scores.append(auc_score)
#
#     def get_auc_avg(self) -> float:
#         vals = [v for v in self.auc_scores if np.isfinite(v)]
#         return float(np.average(vals)) if vals else float("nan")
#
#     def get_auc_std(self) -> float:
#         vals = [v for v in self.auc_scores if np.isfinite(v)]
#         return float(np.std(vals)) if vals else float("nan")
#
#
# def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
#     model = Sequential()
#     model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
#     model.add(LSTM(32, activation="relu", return_sequences=True))
#     model.add(GRU(32, activation="relu", return_sequences=False))
#     model.add(Dense(64, activation="relu"))
#     model.add(Dense(1, activation="sigmoid"))
#     return model
#
#
# def train_task1_model(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
#     reset_random_seeds()
#
#     n = len(X)
#     if n < 10:
#         raise ValueError(f"Too few samples for RNN training: {n}")
#
#     label_unique = np.unique(y)
#     if len(label_unique) < 2:
#         raise ValueError("Degenerate labels: only one class exists")
#
#     n_train = int(0.7 * n)
#     n_val = int(0.85 * n)
#
#     X_tr, y_tr = X[:n_train], y[:n_train]
#     X_val, y_val = X[n_train:n_val], y[n_train:n_val]
#     X_te, y_te = X[n_val:], y[n_val:]
#
#     # If val/test labels are degenerate, roc_auc_score may fail.
#     # The row is still saved with NaN rather than crashing the whole pipeline.
#     model = build_lstm_model(input_shape=(WINDOW_SIZE, X.shape[2]))
#     opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#     model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", "AUC"])
#
#     auc_cb = AUCCallback(validation_data=(X_val, y_val))
#
#     t0 = time.perf_counter()
#     model.fit(
#         X_tr,
#         y_tr,
#         epochs=EPOCHS,
#         validation_data=(X_val, y_val),
#         callbacks=[auc_cb],
#         verbose=0,
#     )
#
#     y_pred = model.predict(X_te, verbose=0)
#     train_seconds = time.perf_counter() - t0
#
#     try:
#         roc_auc = float(roc_auc_score(y_te, y_pred))
#     except ValueError:
#         roc_auc = float("nan")
#
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
#     loss = float(loss_fn(y_te, y_pred).numpy())
#     acc = float(np.mean((y_pred >= 0.5) == y_te))
#
#     tf.keras.backend.clear_session()
#     gc.collect()
#
#     return {
#         "loss": loss,
#         "accuracy": acc,
#         "roc_auc": roc_auc,
#         "auc_avg_val": auc_cb.get_auc_avg(),
#         "auc_std_val": auc_cb.get_auc_std(),
#         "train_seconds": train_seconds,
#     }
#
#
# # ============================================================
# # One Dataset Pipeline
# # ============================================================
#
# def run_dataset(dataset: str) -> List[Dict[str, object]]:
#     rows = []
#     dataset_start = time.perf_counter()
#
#     print("\n" + "=" * 80)
#     print(f"DATASET: {dataset}")
#     print("=" * 80)
#
#     input_csv = find_input_csv(dataset)
#     raw_df = load_edge_csv(input_csv)
#
#     print(f"Input: {input_csv}")
#     print(f"Edges: {len(raw_df):,}")
#     print(f"Date range: {raw_df['_date'].min().date()} -> {raw_df['_date'].max().date()}")
#
#     # 1) Curvature computation once per dataset
#     t0 = time.perf_counter()
#     full_tfrc = compute_tfrc_dataset(dataset, raw_df)
#     tfrc_seconds = time.perf_counter() - t0
#
#     dataset_out_dir = RICCI_OUT_ROOT / dataset
#     dataset_out_dir.mkdir(parents=True, exist_ok=True)
#
#     # Save full TFRC scored file
#     full_tfrc_path = dataset_out_dir / f"{dataset}_tfrc_scored.csv"
#     full_tfrc.drop(columns=["window_start"], errors="ignore").to_csv(full_tfrc_path, index=False)
#
#     basis_df = raw_df[["from", "to", "timestamp", "value"]].copy()
#
#     for removal_pct in REMOVAL_RATIOS:
#         print("\n" + "-" * 70)
#         print(f"{dataset} | removal = {removal_pct}%")
#         print("-" * 70)
#
#         variant_start = time.perf_counter()
#         status = "ok"
#         err_msg = ""
#
#         try:
#             # 2) Sparsification variant
#             t_sparsify = time.perf_counter()
#             variant_df = make_removal_variant(full_tfrc, removal_pct)
#             sparsify_seconds = time.perf_counter() - t_sparsify
#
#             variant_name = f"remove{removal_pct:02d}"
#             variant_path = dataset_out_dir / f"{dataset}_{variant_name}.csv"
#             variant_df.to_csv(variant_path, index=False)
#
#             kept_edges = len(variant_df)
#             original_edges = len(basis_df)
#             kept_pct = 100.0 * kept_edges / max(original_edges, 1)
#
#             # 3) Processing: TDA + sequence creation
#             t_process = time.perf_counter()
#             X, y, diag = build_task1_sequences(variant_df, basis_df)
#             processing_seconds = time.perf_counter() - t_process
#
#             # Save sequences for reproducibility
#             seq_dir = PROCESS_OUT_ROOT / "Sequence_task1" / f"{dataset}_{variant_name}"
#             seq_dir.mkdir(parents=True, exist_ok=True)
#             np.save(seq_dir / "X.npy", X)
#             np.save(seq_dir / "y.npy", y)
#
#             # 4) Prediction
#             metrics = train_task1_model(X, y)
#
#             row = {
#                 "dataset": dataset,
#                 "task": "task1",
#                 "removal_pct": removal_pct,
#                 "kept_pct": round(kept_pct, 4),
#                 "original_edges": original_edges,
#                 "kept_edges": kept_edges,
#                 "roc_auc": metrics["roc_auc"],
#                 "accuracy": metrics["accuracy"],
#                 "loss": metrics["loss"],
#                 "auc_avg_val": metrics["auc_avg_val"],
#                 "auc_std_val": metrics["auc_std_val"],
#                 "num_samples": diag["num_samples"],
#                 "label0": diag["label0"],
#                 "label1": diag["label1"],
#                 "label0_rate": diag["label0_rate"],
#                 "label1_rate": diag["label1_rate"],
#                 "tfrc_seconds_once": tfrc_seconds,
#                 "sparsify_seconds": sparsify_seconds,
#                 "processing_seconds": processing_seconds,
#                 "prediction_seconds": metrics["train_seconds"],
#                 "variant_total_seconds": time.perf_counter() - variant_start,
#                 "status": status,
#                 "error": err_msg,
#                 "variant_csv": str(variant_path),
#             }
#
#         except Exception as e:
#             status = "failed"
#             err_msg = str(e)
#             print(f"[ERROR] {dataset} removal {removal_pct}% failed: {err_msg}")
#
#             row = {
#                 "dataset": dataset,
#                 "task": "task1",
#                 "removal_pct": removal_pct,
#                 "kept_pct": np.nan,
#                 "original_edges": len(basis_df),
#                 "kept_edges": np.nan,
#                 "roc_auc": np.nan,
#                 "accuracy": np.nan,
#                 "loss": np.nan,
#                 "auc_avg_val": np.nan,
#                 "auc_std_val": np.nan,
#                 "num_samples": np.nan,
#                 "label0": np.nan,
#                 "label1": np.nan,
#                 "label0_rate": np.nan,
#                 "label1_rate": np.nan,
#                 "tfrc_seconds_once": tfrc_seconds,
#                 "sparsify_seconds": np.nan,
#                 "processing_seconds": np.nan,
#                 "prediction_seconds": np.nan,
#                 "variant_total_seconds": time.perf_counter() - variant_start,
#                 "status": status,
#                 "error": err_msg,
#                 "variant_csv": "",
#             }
#
#         rows.append(row)
#         pd.DataFrame(rows).to_csv(dataset_out_dir / f"{dataset}_sensitivity_partial.csv", index=False)
#
#         # Also append global file incrementally for safety
#         append_results_row(row)
#
#     runtime_row = pd.DataFrame([{
#         "dataset": dataset,
#         "start_time": now_str(),
#         "total_dataset_seconds": time.perf_counter() - dataset_start,
#     }])
#     if RUNTIME_CSV.exists():
#         runtime_row.to_csv(RUNTIME_CSV, mode="a", header=False, index=False)
#     else:
#         runtime_row.to_csv(RUNTIME_CSV, index=False)
#
#     return rows
#
#
# def append_results_row(row: Dict[str, object]) -> None:
#     df_row = pd.DataFrame([row])
#     if RESULTS_CSV.exists():
#         df_row.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
#     else:
#         df_row.to_csv(RESULTS_CSV, index=False)
#
#
# # ============================================================
# # Summary + Plot
# # ============================================================
#
# def summarize_and_plot() -> None:
#     if not RESULTS_CSV.exists():
#         print(f"[WARN] Results CSV not found: {RESULTS_CSV}")
#         return
#
#     df = pd.read_csv(RESULTS_CSV)
#     df_ok = df[df["status"] == "ok"].copy()
#     df_ok["roc_auc"] = pd.to_numeric(df_ok["roc_auc"], errors="coerce")
#
#     summary = (
#         df_ok.groupby("removal_pct", as_index=False)
#         .agg(
#             mean_roc_auc=("roc_auc", "mean"),
#             std_roc_auc=("roc_auc", "std"),
#             n_datasets=("dataset", "nunique"),
#         )
#         .sort_values("removal_pct")
#     )
#
#     summary_path = RESULTS_DIR / "removal_ratio_sensitivity_task1_summary.csv"
#     summary.to_csv(summary_path, index=False)
#
#     print("\nSensitivity summary:")
#     print(summary.to_string(index=False))
#
#     fig, ax = plt.subplots(figsize=(6.5, 4.2))
#
#     x = summary["removal_pct"].to_numpy(dtype=float)
#     y = summary["mean_roc_auc"].to_numpy(dtype=float)
#     yerr = summary["std_roc_auc"].fillna(0).to_numpy(dtype=float)
#
#     ax.plot(
#         x,
#         y,
#         marker="o",
#         linewidth=2.6,
#         markersize=6,
#         color="#6A0DAD",
#         label="Our method",
#     )
#
#     ax.fill_between(
#         x,
#         y - yerr,
#         y + yerr,
#         color="#6A0DAD",
#         alpha=0.12,
#         linewidth=0,
#         label="Std. across datasets",
#     )
#
#     ax.set_xlabel("Removed Edges (%)", fontsize=12)
#     ax.set_ylabel("ROC-AUC", fontsize=12)
#     ax.set_xticks(REMOVAL_RATIOS)
#     ax.set_xlim(min(REMOVAL_RATIOS), max(REMOVAL_RATIOS))
#
#     finite_y = y[np.isfinite(y)]
#     if len(finite_y) > 0:
#         y_min = max(0.0, float(np.nanmin(finite_y)) - 0.05)
#         y_max = min(1.0, float(np.nanmax(finite_y)) + 0.05)
#         if y_max - y_min < 0.1:
#             y_min = max(0.0, y_min - 0.05)
#             y_max = min(1.0, y_max + 0.05)
#         ax.set_ylim(y_min, y_max)
#     else:
#         ax.set_ylim(0.0, 1.0)
#
#     ax.grid(True, linestyle="--", alpha=0.35)
#     ax.legend(fontsize=10, loc="best", frameon=True)
#     ax.tick_params(axis="both", labelsize=10)
#
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.0)
#
#     plt.tight_layout()
#     plt.savefig(PLOT_PATH, dpi=600, bbox_inches="tight")
#     plt.close()
#
#     print(f"[SAVED] Results     -> {RESULTS_CSV}")
#     print(f"[SAVED] Summary     -> {summary_path}")
#     print(f"[SAVED] Plot        -> {PLOT_PATH}")
#
#
# # ============================================================
# # Main
# # ============================================================
#
# if __name__ == "__main__":
#     overall_start = time.perf_counter()
#
#     print("=" * 80)
#     print("Edge-removal sensitivity pipeline")
#     print("=" * 80)
#     print(f"Project root : {PROJECT_ROOT}")
#     print(f"Input dir    : {INPUT_DIR}")
#     print(f"Datasets     : {DATASETS}")
#     print(f"Removal %    : {REMOVAL_RATIOS}")
#     print(f"Output CSV   : {RESULTS_CSV}")
#     print(f"Output plot  : {PLOT_PATH}")
#     print("=" * 80)
#
#     # Start fresh for this sensitivity run
#     if RESULTS_CSV.exists():
#         backup = RESULTS_CSV.with_suffix(f".backup_{int(time.time())}.csv")
#         RESULTS_CSV.rename(backup)
#         print(f"[INFO] Existing results backed up to: {backup}")
#
#     for dataset in DATASETS:
#         try:
#             run_dataset(dataset)
#         except Exception as e:
#             print(f"[FATAL DATASET ERROR] {dataset}: {e}")
#             append_results_row({
#                 "dataset": dataset,
#                 "task": "task1",
#                 "removal_pct": np.nan,
#                 "kept_pct": np.nan,
#                 "original_edges": np.nan,
#                 "kept_edges": np.nan,
#                 "roc_auc": np.nan,
#                 "accuracy": np.nan,
#                 "loss": np.nan,
#                 "auc_avg_val": np.nan,
#                 "auc_std_val": np.nan,
#                 "num_samples": np.nan,
#                 "label0": np.nan,
#                 "label1": np.nan,
#                 "label0_rate": np.nan,
#                 "label1_rate": np.nan,
#                 "tfrc_seconds_once": np.nan,
#                 "sparsify_seconds": np.nan,
#                 "processing_seconds": np.nan,
#                 "prediction_seconds": np.nan,
#                 "variant_total_seconds": np.nan,
#                 "status": "dataset_failed",
#                 "error": str(e),
#                 "variant_csv": "",
#             })
#
#     summarize_and_plot()
#
#     total_seconds = time.perf_counter() - overall_start
#     h = int(total_seconds // 3600)
#     m = int((total_seconds % 3600) // 60)
#     s = total_seconds % 60
#     print("\n" + "=" * 80)
#     print(f"TOTAL RUNTIME: {h}h {m}m {s:.2f}s")
#     print("=" * 80)

import matplotlib.pyplot as plt
import numpy as np

removal_pct = [0, 20, 40, 60, 80, 90]

mean_roc_auc = [
    0.789346,
    0.791111,
    0.761917,
    0.774231,
    0.768046,
    0.641474
]

std_roc_auc = [
    0.039,
    0.047,
    0.052,
    0.043,
    0.055,
    0.075
]

x = np.array(removal_pct)
y = np.array(mean_roc_auc)
std = np.array(std_roc_auc)

fig, ax = plt.subplots(figsize=(6.0, 3.8))

ax.plot(
    x,
    y,
    marker="o",
    linewidth=2.2,
    markersize=6,
    color="#6A5ACD",
    label="Mean ROC-AUC"
)

ax.fill_between(
    x,
    y - std,
    y + std,
    color="#6A5ACD",
    alpha=0.15,
    linewidth=0,
    label="Std. across datasets"
)

ax.set_xlabel("Removed Edges (%)", fontsize=12)
ax.set_ylabel("Mean ROC-AUC", fontsize=12)
ax.set_xticks(x)
ax.set_ylim(0.55, 0.85)

ax.grid(linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=10)

ax.legend(frameon=False, fontsize=9)

plt.tight_layout()

plt.savefig(
    "edge_removal_sensitivity.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()