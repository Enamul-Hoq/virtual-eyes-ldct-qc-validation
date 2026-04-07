import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def scan_npy_folder(
    folder_path: str,
    label: int,
    input_dim: Optional[int] = None,
    patient_id_from_filename: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels, patient_ids = [], [], []

    if not os.path.exists(folder_path):
        print(f"[WARN] Folder not found: {folder_path}")
        return np.empty((0, input_dim or 0)), np.array([]), np.array([])

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

    for fname in files:
        path = os.path.join(folder_path, fname)
        data = np.load(path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if input_dim is not None and data.shape[1] != input_dim:
            raise ValueError(
                f"Unexpected feature dimension in {fname}: got {data.shape[1]}, expected {input_dim}"
            )

        pid = fname.split("_")[0] if patient_id_from_filename else Path(fname).stem

        embeddings.append(data)
        labels.extend([label] * len(data))
        patient_ids.extend([pid] * len(data))

    if not embeddings:
        return np.empty((0, input_dim or 0)), np.array([]), np.array([])

    return np.vstack(embeddings), np.array(labels), np.array(patient_ids)


def combine_folders(
    pos_folder: str,
    neg_folder: str,
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_pos, y_pos, p_pos = scan_npy_folder(pos_folder, label=1, input_dim=input_dim)
    x_neg, y_neg, p_neg = scan_npy_folder(neg_folder, label=0, input_dim=input_dim)

    if len(x_pos) == 0 and len(x_neg) == 0:
        return np.empty((0, input_dim)), np.array([]), np.array([])

    if len(x_pos) == 0:
        return x_neg, y_neg, p_neg

    if len(x_neg) == 0:
        return x_pos, y_pos, p_pos

    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg], axis=0)
    pids = np.concatenate([p_pos, p_neg], axis=0)
    return x, y, pids


def patient_level_labels(pids: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"pid": pids, "label": y})
    grouped = df.groupby("pid", as_index=False)["label"].first()
    return grouped


def split_patients(
    pids: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    patient_df = patient_level_labels(pids, y)
    patient_ids = patient_df["pid"].values
    patient_labels = patient_df["label"].values

    train_pids, temp_pids, train_labels, temp_labels = train_test_split(
        patient_ids,
        patient_labels,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=patient_labels,
    )

    relative_test = test_size / (val_size + test_size)
    val_pids, test_pids, _, _ = train_test_split(
        temp_pids,
        temp_labels,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": train_pids,
        "val": val_pids,
        "test": test_pids,
    }


def subset_by_patient_ids(
    x: np.ndarray,
    y: np.ndarray,
    pids: np.ndarray,
    patient_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isin(pids, patient_ids)
    return x[mask], y[mask], pids[mask]


def build_mlp(
    input_dim: int,
    hidden_dims: List[int] = [512, 256],
    dropout: float = 0.3,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for hidden_dim in hidden_dims:
        model.add(tf.keras.layers.Dense(hidden_dim, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    results_dir: str,
    model_name: str,
    epochs: int = 40,
    batch_size: int = 128,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    ensure_dir(results_dir)

    ckpt_path = os.path.join(results_dir, f"{model_name}_best.weights.h5")

    callback_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )
    return model, history


def patient_pooling_predictions(
    y_true_slices: np.ndarray,
    y_pred_probs: np.ndarray,
    pids: np.ndarray,
) -> pd.DataFrame:
    rows = []
    unique_pids = np.unique(pids)

    for pid in unique_pids:
        mask = pids == pid
        probs = y_pred_probs[mask].reshape(-1)
        labels = y_true_slices[mask].reshape(-1)

        rows.append(
            {
                "pid": pid,
                "label": int(labels[0]),
                "mean_prob": float(np.mean(probs)),
                "max_prob": float(np.max(probs)),
                "top3_prob": float(np.mean(np.sort(probs)[-3:])) if len(probs) >= 3 else float(np.mean(probs)),
            }
        )

    return pd.DataFrame(rows)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["pr_auc"] = float(auc(recall, precision))
    return metrics


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        scores.append(roc_auc_score(y_b, p_b))

    if not scores:
        return float("nan"), float("nan")

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def save_history_plot(history: tf.keras.callbacks.History, results_dir: str, prefix: str) -> None:
    hist = history.history

    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{prefix}_loss_curve.png"), dpi=200)
    plt.close()

    if "auc" in hist and "val_auc" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["auc"], label="train_auc")
        plt.plot(hist["val_auc"], label="val_auc")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{prefix} Training AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{prefix}_auc_curve.png"), dpi=200)
        plt.close()


def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, results_dir: str, prefix: str) -> None:
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{prefix}_roc.png"), dpi=200)
    plt.close()
