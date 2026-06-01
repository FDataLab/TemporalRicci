"""
rnn_mog.py
==========

RNN training for MoG Effective Resistance baseline.
Reads sequences produced by process_mog.py from GraphPulseResultsMoG/.

No command-line arguments needed — just set DATASET below and run.

File lives at: TemporalRicci/src/MoG/rnn_mog.py
"""

import json
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import LSTM, Dense, GRU
from keras.models import Sequential
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIG
# ============================================================

DATASET = "tgbl-comment"

# ============================================================
# PATHS
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent      # src/MoG/
_BASE_DIR = _THIS_DIR.parent.parent              # TemporalRicci/

GRAPHPULSE_MOG_DIR = str(_BASE_DIR / "GraphPulseResultsMoG")

# Results output
RESULTS_DIR  = str(_BASE_DIR / "GraphPulseResultsMoG" / "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, f"RNNResults_MoG_{DATASET}.csv")
RUNTIME_FILE = os.path.join(RESULTS_DIR, f"RNN_Runtime_MoG_{DATASET}.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# NETWORKS
# ============================================================

NETWORKS = [f"{DATASET}_mog_er"]

# The base network whose labels are used (same variant — labels come
# from the raw data in process_mog.py, stored in the same seq_raw.txt)
BASE_NETWORK = NETWORKS[0]

# ============================================================
# TASK FOLDERS
# ============================================================

TASK_FOLDERS = {
    "task1": {"pretty": "Network Growth Prediction",      "folder": "Sequence_task1"},
    "task2": {"pretty": "Influential Node Prediction",    "folder": "Sequence_task2"},
    "task3": {"pretty": "Connected Component Prediction", "folder": "Sequence_task3"},
}

# ============================================================
# TRAINING SETTINGS
# ============================================================

LEARNING_RATE_DEFAULT = 0.0001
NORMALIZER_MODE       = "all"
EPOCHS                = 10

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
        print(f"  Epoch {epoch + 1} - Val AUC: {auc_score:.4f}")

    def get_auc_std(self) -> float:
        vals = [v for v in self.auc_scores if np.isfinite(v)]
        return float(np.std(vals)) if vals else float("nan")

    def get_auc_avg(self) -> float:
        vals = [v for v in self.auc_scores if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")


def reset_random_seeds(seed: int = 1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def ensure_results_header():
    if not os.path.exists(RESULTS_FILE) or os.path.getsize(RESULTS_FILE) == 0:
        with open(RESULTS_FILE, "w") as f:
            f.write(
                "Task,Network,Loss,Accuracy,ROC_AUC,AUC_AVG,AUC_STD,"
                "TrainTimeSec,NumSamples,LearningRate,Label0,Label1,"
                "Label0Rate,Label1Rate,AllZeroTDA,AllZeroRAW,"
                "NonEmptyEither,NonEmptyBoth,StartDateTime,EndDateTime\n"
            )


def ensure_runtime_header():
    if not os.path.exists(RUNTIME_FILE) or os.path.getsize(RUNTIME_FILE) == 0:
        with open(RUNTIME_FILE, "w") as f:
            f.write("Task,Network,StartDateTime,EndDateTime,TotalSeconds\n")


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation="relu", return_sequences=True))
    model.add(GRU(32, activation="relu", return_sequences=False))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_pair(np_data, np_data_raw, mode="per_column"):
    def norm(arr):
        min_v = np.min(arr, axis=(0, 1), keepdims=True)
        max_v = np.max(arr, axis=(0, 1), keepdims=True)
        scaled = (arr - min_v) / (max_v - min_v + 1e-10)
        return np.nan_to_num(scaled)
    return norm(np_data), norm(np_data_raw)


# ============================================================
# DATA LOADING
# ============================================================

def assemble_data(task_key, network, base_network, normalizer="all"):
    tinfo    = TASK_FOLDERS[task_key]
    seq_dir  = os.path.join(GRAPHPULSE_MOG_DIR, tinfo["folder"], network)
    base_dir = os.path.join(GRAPHPULSE_MOG_DIR, tinfo["folder"], base_network)

    print(f"\n--- Assembling {tinfo['pretty']} | {seq_dir}")

    with open(os.path.join(seq_dir, "seq_tda.txt"), "r") as f:
        seq_tda = json.load(f)

    with open(os.path.join(seq_dir, "seq_raw.txt"), "r") as f:
        seq_raw = json.load(f)

    with open(os.path.join(base_dir, "seq_raw.txt"), "r") as f:
        seq_raw_base = json.load(f)

    y = np.array(seq_raw_base["LABELS"], dtype=np.float32)

    tda_sequences = seq_tda["TDA_SEQUENCE"]["mapper"]
    raw_sequences = seq_raw["RAW_SEQUENCE"]["raw"]

    if len(tda_sequences) != len(raw_sequences):
        print(f"[FATAL] Length mismatch: TDA={len(tda_sequences)}, RAW={len(raw_sequences)}")
        raise SystemExit(1)

    assert all(len(seq) == 7 for seq in tda_sequences)
    assert all(len(seq) == 7 for seq in raw_sequences)

    tda_np = np.array(tda_sequences, dtype=np.float32)
    raw_np = np.array(raw_sequences, dtype=np.float32)

    n = len(tda_sequences)
    all_zero_tda = int((np.abs(tda_np).sum(axis=(1, 2)) == 0).sum())
    all_zero_raw = int((np.abs(raw_np).sum(axis=(1, 2)) == 0).sum())
    lbl0 = int((y == 0).sum())
    lbl1 = int((y == 1).sum())

    diag = {
        "Label0": lbl0, "Label1": lbl1,
        "Label0Rate": lbl0 / n, "Label1Rate": lbl1 / n,
        "AllZeroTDA": all_zero_tda, "AllZeroRAW": all_zero_raw,
        "NonEmptyEither": n - min(all_zero_tda, all_zero_raw),
        "NonEmptyBoth":   n - (all_zero_tda + all_zero_raw),
    }

    norm_tda, norm_raw = normalize_pair(tda_np, raw_np, mode=normalizer)
    X = np.concatenate((norm_tda, norm_raw), axis=2)
    y = y.reshape(-1, 1)

    print(f"  X shape: {X.shape}, y shape: {y.shape}, labels: 0={lbl0}, 1={lbl1}")
    return X, y, diag


# ============================================================
# TRAINING
# ============================================================

def train_and_log(task_key, network, X, y, diag):
    ensure_results_header()
    ensure_runtime_header()
    reset_random_seeds()

    # Train / val / test split
    n       = len(X)
    n_train = int(0.7  * n)
    n_val   = int(0.85 * n)
    X_tr, y_tr   = X[:n_train],        y[:n_train]
    X_val, y_val = X[n_train:n_val],   y[n_train:n_val]
    X_te,  y_te  = X[n_val:],          y[n_val:]

    model = build_lstm_model(input_shape=(7, X.shape[2]))
    opt   = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DEFAULT)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy", "AUC"])

    auc_cb = AUCCallback(validation_data=(X_val, y_val))

    start_unix     = time.time()
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_unix))

    model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[auc_cb],
        verbose=0
    )

    y_pred = model.predict(X_te, verbose=0)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss    = float(loss_fn(y_te, y_pred).numpy())
    acc     = float(np.mean((y_pred >= 0.5) == y_te))
    auc     = float(tf.keras.metrics.AUC()(y_te, y_pred).numpy())
    roc_auc = float(roc_auc_score(y_te, y_pred))

    end_unix     = time.time()
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_unix))
    total_seconds = end_unix - start_unix

    # Save runtime
    with open(RUNTIME_FILE, "a") as f:
        f.write(f"{task_key},{network},{start_datetime},{end_datetime},{total_seconds:.2f}\n")

    # Save results CSV
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{task_key},{network},{loss:.4f},{acc:.4f},{roc_auc:.4f},"
            f"{auc_cb.get_auc_avg():.4f},{auc_cb.get_auc_std():.4f},{total_seconds:.2f},"
            f"{len(X)},{LEARNING_RATE_DEFAULT},{diag['Label0']},{diag['Label1']},"
            f"{diag['Label0Rate']:.4f},{diag['Label1Rate']:.4f},{diag['AllZeroTDA']},"
            f"{diag['AllZeroRAW']},{diag['NonEmptyEither']},{diag['NonEmptyBoth']},"
            f"{start_datetime},{end_datetime}\n"
        )

    print(f"[{TASK_FOLDERS[task_key]['pretty']}] {network} | "
          f"loss={loss:.4f} acc={acc:.4f} ROC_AUC={roc_auc:.4f} | "
          f"{start_datetime} -> {end_datetime}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"Dataset  : {DATASET}")
    print(f"Networks : {NETWORKS}")
    print(f"Sequences: {GRAPHPULSE_MOG_DIR}")
    print(f"Results  : {RESULTS_FILE}")
    print()

    for task_key in ["task1", "task2", "task3"]:
        tinfo = TASK_FOLDERS[task_key]
        print(f"\n{'='*60}")
        print(f"Task: {tinfo['pretty']}")
        print(f"{'='*60}")

        for network in NETWORKS:
            X, y, diag = assemble_data(
                task_key, network, BASE_NETWORK, NORMALIZER_MODE
            )
            train_and_log(task_key, network, X, y, diag)