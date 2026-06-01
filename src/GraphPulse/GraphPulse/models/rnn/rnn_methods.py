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
import wandb
import re
from wandb.integration.keras import WandbMetricsLogger

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
RESULTS_FILE = os.path.join(RESULTS_DIR, "RNNResultsAllTasks_tgbl-comment_Sensitivity.csv")

RUNTIME_FILE = os.path.join(RESULTS_DIR, "RNN_PredictionRuntime_tgbl-comment.csv")

TASK_FOLDERS = {

    "task1": {"pretty": "Network Growth Prediction", "folder": "Sequence_task1"},
    "task2": {"pretty": "Influential Node Prediction", "folder": "Sequence_task2"},
    "task3": {"pretty": "Connected Component Prediction", "folder": "Sequence_task3"},
}

NETWORKS = [
"tgbl-comment_full",
"tgbl-comment_bin1",
"tgbl-comment_bin2",
"tgbl-comment_bin3",
"tgbl-comment_bin4",
"tgbl-comment_bin5",
]

def ensure_runtime_header():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if (not os.path.exists(RUNTIME_FILE)) or os.path.getsize(RUNTIME_FILE) == 0:
        with open(RUNTIME_FILE, "w") as f:
            f.write("Task,Network,StartDateTime,EndDateTime,TotalSeconds\n")


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
        print(f"{task_key} Epoch {epoch + 1} - Validation AUC: {auc_score:.4f}")

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
                "Task,Network,Loss,Accuracy,ROC_AUC,AUC_AVG,AUC_STD,"
                "TrainTimeSec,NumSamples,LearningRate,Label0,Label1,Label0Rate,Label1Rate,"
                "AllZeroTDA,AllZeroRAW,NonEmptyEither,NonEmptyBoth,"
                "StartDateTime,EndDateTime\n"
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


def normalize_pair_old (np_data, np_data_raw, mode: str):
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

def normalize_pair(np_data, np_data_raw, mode="per_column"):
    # normalize per feature column across all samples and time steps
    def norm(arr):
        min_v = np.min(arr, axis=(0, 1), keepdims=True)
        max_v = np.max(arr, axis=(0, 1), keepdims=True)
        scaled = (arr - min_v) / (max_v - min_v + 1e-10)
        return np.nan_to_num(scaled)

    return norm(np_data), norm(np_data_raw)


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
        print(f"[FATAL âŒ] Length mismatch: TDA has {len(tda_sequences)} sequences, RAW has {len(raw_sequences)}.")
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

def short_net_name(network: str) -> str:
    # remove the numeric specifier part
    cleaned = network.replace("TFR_a3.00_b1.00_", "")
    return cleaned


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_and_log(task_key, network, X, y, diag):
    epochs = 10

    tags = [task_key, network.split("_")[-1], "RNN", "7day"]
    short_net = network
    run_name = f"{task_key}{short_net}"

    wandb.init(
        project="GraphPulse_RNN",
        group=task_key,
        name=run_name,
        job_type="training",
        tags=tags,
        config={
            "learning_rate": LEARNING_RATE_DEFAULT,
            "epochs": epochs,
            "sequence_length": 7,
            "normalizer_mode": NORMALIZER_MODE,
            "task": task_key,
            "network": short_net
        },
        reinit=True
    )

    wandb_logger = WandbMetricsLogger()
    ensure_results_header()
    ensure_runtime_header()   # <--- ADD THIS
    reset_random_seeds()

    # Split dataset
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_te, y_te = X[n_val:], y[n_val:]

    # Build and compile model
    model = build_lstm_model(input_shape=(7, X.shape[2]))
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DEFAULT)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", "AUC"])

    auc_cb = AUCCallback(validation_data=(X_val, y_val))

    # ===== Total runtime: train + test + metrics =====
    start_unix = time.time()
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_unix))

    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[auc_cb, wandb_logger],
        verbose=0
    )

    y_pred = model.predict(X_te, verbose=0)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = float(loss_fn(y_te, y_pred).numpy())
    acc = float(np.mean((y_pred >= 0.5) == y_te))
    auc = float(tf.keras.metrics.AUC()(y_te, y_pred).numpy())
    roc_auc = float(roc_auc_score(y_te, y_pred))

    end_unix = time.time()
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_unix))
    total_seconds = end_unix - start_unix

    # <--- ADD THIS: write runtime row
    with open(RUNTIME_FILE, "a") as f:
        f.write(f"{task_key},{network},{start_datetime},{end_datetime},{total_seconds:.2f}\n")

    # ---- W&B logging ----
    wandb.log({
        "test/loss": loss,
        "test/accuracy": acc,
        "test/auc_keras": auc,
        "test/roc_auc": roc_auc,
        "validation/auc_mean": auc_cb.get_auc_avg(),
        "validation/auc_std": auc_cb.get_auc_std(),
        "runtime/train_test_total_sec": total_seconds,
        "dataset/num_samples": len(X),
        "dataset/label0_rate": diag["Label0Rate"],
        "dataset/label1_rate": diag["Label1Rate"],
        "dataset/all_zero_tda": diag["AllZeroTDA"],
        "dataset/all_zero_raw": diag["AllZeroRAW"]
    })

    # ---- Save main results ----
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{task_key},{network},{loss:.4f},{acc:.4f},{roc_auc:.4f},"
            f"{auc_cb.get_auc_avg():.4f},{auc_cb.get_auc_std():.4f},{total_seconds:.2f},"
            f"{len(X)},{LEARNING_RATE_DEFAULT},{diag['Label0']},{diag['Label1']},"
            f"{diag['Label0Rate']:.4f},{diag['Label1Rate']:.4f},{diag['AllZeroTDA']},"
            f"{diag['AllZeroRAW']},{diag['NonEmptyEither']},{diag['NonEmptyBoth']},"
            f"{start_datetime},{end_datetime}\n"
        )

    wandb.finish()

    print(f"[{TASK_FOLDERS[task_key]['pretty']}] {network} | "
          f"loss={loss:.4f} acc={acc:.4f} AUC={auc:.4f} ROC_AUC={roc_auc:.4f} | "
          f"Started: {start_datetime} Ended: {end_datetime}")


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