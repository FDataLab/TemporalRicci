import json

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import pickle
import time
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import LSTM, Dense, GRU
from keras.models import Sequential
from sklearn.metrics import roc_auc_score

import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Script that takes a base directory as argument.")
parser.add_argument("--base_dir", type=str, required=True,
                    help="Absolute path to the project base directory.")
args = parser.parse_args()

BASE_DIR = os.path.abspath(args.base_dir)
print("Using base directory:", BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "data/output")
RESULTS_FILE = os.path.join(RESULTS_DIR, "RNNResultsAllTasks.csv")

TASK_FOLDERS = {
    "task1": {"pretty": "Network Growth Prediction", "folder": "Sequence_task1"},
    "task2": {"pretty": "Influential Node Prediction", "folder": "Sequence_task2"},
    "task3": {"pretty": "Connected Component Prediction", "folder": "Sequence_task3"},
}

NETWORKS = [
    "BEPRO_TFR_a3.00_b1.00",
    "BEPRO_TFR_a3.00_b1.00_bin1", "BEPRO_TFR_a3.00_b1.00_bin2",
    "BEPRO_TFR_a3.00_b1.00_bin3", "BEPRO_TFR_a3.00_b1.00_bin4",
    "BEPRO_TFR_a3.00_b1.00_bin5", "BEPRO_TFR_a3.00_b1.00_bin6",
    "BEPRO_TFR_a3.00_b1.00_bin7", "BEPRO_TFR_a3.00_b1.00_bin8",
    "BEPRO_TFR_a3.00_b1.00_bin9", "BEPRO_TFR_a3.00_b1.00_bin10",
]

LEARNING_RATE_DEFAULT = 0.0001
NORMALIZER_MODE = "all"   # or "per_column"

# ============================================================
# UTILITIES
# ============================================================

class AUCCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.auc_scores: List[float] = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val, verbose=0)
        try:
            auc_score = roc_auc_score(y_val, y_pred)
        except ValueError:
            auc_score = float("nan")
        self.auc_scores.append(auc_score)
        print(f"{task_key}Epoch {epoch + 1} - Validation AUC: {auc_score:.4f}")

    def get_auc_std(self) -> float:
        vals = [v for v in self.auc_scores if np.isfinite(v)]
        return float(np.std(vals)) if vals else float("nan")

    def get_auc_avg(self) -> float:
        vals = [v for v in self.auc_scores if np.isfinite(v)]
        return float(np.average(vals)) if vals else float("nan")


def reset_random_seeds(seed: int = 1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def ensure_results_header():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_FILE) or os.path.getsize(RESULTS_FILE) == 0:
        with open(RESULTS_FILE, "w") as fh:
            fh.write(
                "Task,TaskPretty,Network,Spec,Loss,Accuracy,AUC,ROC_AUC,AUC_AVG,AUC_STD,"
                "TrainTimeSec,NumSamples,LearningRate,Label0,Label1,Label0Rate,Label1Rate,"
                "AllZeroTDA,AllZeroRAW,NonEmptyEither,NonEmptyBoth\n"
            )


def pad_sequence(seq, target_len=7, feature_dim=None):
    seq = np.array(seq, dtype=np.float32)
    if feature_dim is None:
        feature_dim = max(len(row) for row in seq)
    padded = np.zeros((target_len, feature_dim), dtype=np.float32)
    for i in range(min(len(seq), target_len)):
        row = np.array(seq[i], dtype=np.float32)
        padded[i, :len(row)] = row
    return padded


def normalize_pair(np_data, np_data_raw, mode: str):
    if mode == "per_column":
        min_values = np.min(np_data, axis=(0, 1))
        max_values = np.max(np_data, axis=(0, 1))
        normalized_data_arr = (np_data - min_values) / (max_values - min_values + 1e-10)
        normalized_data_arr = np.nan_to_num(normalized_data_arr)

        min_values = np.min(np_data_raw, axis=(0, 1))
        max_values = np.max(np_data_raw, axis=(0, 1))
        normalized_raw_data_arr = (np_data_raw - min_values) / (max_values - min_values + 1e-10)
        normalized_raw_data_arr = np.nan_to_num(normalized_raw_data_arr)
    else:
        gmin = np.min(np_data)
        gmax = np.max(np_data)
        normalized_data_arr = (np_data - gmin) / (gmax - gmin + 1e-10)
        normalized_data_arr = np.nan_to_num(normalized_data_arr)

        gmin = np.min(np_data_raw)
        gmax = np.max(np_data_raw)
        normalized_raw_data_arr = (np_data_raw - gmin) / (gmax - gmin + 1e-10)
        normalized_raw_data_arr = np.nan_to_num(normalized_raw_data_arr)

    return normalized_data_arr, normalized_raw_data_arr


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation="relu", return_sequences=True))
    model.add(GRU(32, activation="relu", return_sequences=False))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# DATA LOADING
# ============================================================

def assemble_data(task_key, network, base_network, normalizer="all"):
    tinfo = TASK_FOLDERS[task_key]
    base_dir = os.path.join(BASE_DIR, tinfo["folder"], network)
    print(f"\n--- Assembling {tinfo['pretty']} | {base_dir}")

    with open(os.path.join(base_dir, "seq_tda.txt"), "r") as f:
        seq_tda = json.load(f)

    with open(os.path.join(base_dir, "seq_raw.txt"), "r") as f:
        seq_raw = json.load(f)
    with open(os.path.join(os.path.join(BASE_DIR, tinfo["folder"], base_network), "seq_raw.txt"), "r") as f:
        seq_raw_base = json.load(f)

    # Directly access the lists
    y = np.array(seq_raw_base["LABELS"], dtype=np.float32)

    tda_sequences = seq_tda["TDA_SEQUENCE"]["mapper"]
    raw_sequences = seq_raw["RAW_SEQUENCE"]["raw"]
    # --- Sanity check: ensure TDA and RAW sequences have same length ---
    if len(tda_sequences) != len(raw_sequences):
        print(f"[FATAL ❌] Length mismatch: TDA has {len(tda_sequences)} sequences, RAW has {len(raw_sequences)}.")
        print("These should be aligned per temporal window. Please verify preprocessing.")
        exit(1)





    assert all(len(seq) == 7 for seq in tda_sequences)
    assert all(len(seq) == 7 for seq in raw_sequences)
    tda_np = np.array(tda_sequences, dtype=np.float32)
    raw_np = np.array(raw_sequences, dtype=np.float32)


    n = len(tda_sequences)
    # Diagnostics
    all_zero_tda = int((np.abs(tda_np).sum(axis=(1, 2)) == 0).sum())
    all_zero_raw = int((np.abs(raw_np).sum(axis=(1, 2)) == 0).sum())
    lbl0 = int((y == 0).sum())
    lbl1 = int((y == 1).sum())
    diag = {
        "Label0": lbl0, "Label1": lbl1,
        "Label0Rate": lbl0 / n, "Label1Rate": lbl1 / n,
        "AllZeroTDA": all_zero_tda, "AllZeroRAW": all_zero_raw,
        "NonEmptyEither": n - min(all_zero_tda, all_zero_raw),
        "NonEmptyBoth": n - (all_zero_tda + all_zero_raw),
    }

    # Normalize and merge
    norm_tda, norm_raw = normalize_pair(tda_np, raw_np, mode=normalizer)
    X = np.concatenate((norm_tda, norm_raw), axis=2)
    y = y.reshape(-1, 1)

    print(f"  X shape: {X.shape}, y shape: {y.shape}, labels: 0={lbl0}, 1={lbl1}")
    return X, y, diag


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_and_log(task_key, network, X, y, diag):
    ensure_results_header()
    reset_random_seeds()

    # Split
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_te, y_te = X[n_val:], y[n_val:]

    model = build_lstm_model(input_shape=(7, X.shape[2]))
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DEFAULT)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", "AUC"])

    auc_cb = AUCCallback(validation_data=(X_val, y_val))

    t0 = time.time()
    model.fit(X_tr, y_tr, epochs=10, validation_data=(X_val, y_val),
              callbacks=[auc_cb], verbose=0)
    elapsed = time.time() - t0

    y_pred = model.predict(X_te, verbose=0)
    try:
        roc_auc = roc_auc_score(y_te, y_pred)
    except ValueError:
        roc_auc = float("nan")
    loss, acc, auc = model.evaluate(X_te, y_te, verbose=0)

    # Save RicciResults
    with open(RESULTS_FILE, "a") as f:
        f.write("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.2f},{},{},"
                "{},{:.4f},{:.4f},{},{},{},{}\n".format(
            task_key,
            TASK_FOLDERS[task_key]["pretty"],
            network,
            loss, acc, auc, roc_auc,
            auc_cb.get_auc_avg(), auc_cb.get_auc_std(),
            elapsed, len(X),
            LEARNING_RATE_DEFAULT,
            diag["Label0"], diag["Label1"],
            diag["Label0Rate"], diag["Label1Rate"],
            diag["AllZeroTDA"], diag["AllZeroRAW"],
            diag["NonEmptyEither"], diag["NonEmptyBoth"]
        ))
    print(f"[{TASK_FOLDERS[task_key]['pretty']}] {network} | loss={loss:.4f} acc={acc:.4f} AUC={auc:.4f} ROC_AUC={roc_auc:.4f}")


# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":
    for task_key in ["task1", "task2", "task3"]:
        base_network = NETWORKS[0]
        print("We are using labels of ", base_network)
        for network in NETWORKS:
            # try:
                X, y, diag = assemble_data(task_key, network, base_network, NORMALIZER_MODE)
                train_and_log(task_key, network, X, y, diag)
            # except Exception as e:
            #     print(f"[Error] {task_key} - {network}: {e}")
