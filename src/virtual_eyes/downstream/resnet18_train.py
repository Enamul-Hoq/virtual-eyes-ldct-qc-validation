import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from virtual_eyes.downstream.common import (
    bootstrap_auc_ci,
    compute_binary_metrics,
    combine_folders,
    ensure_dir,
    patient_pooling_predictions,
    save_history_plot,
    save_json,
    save_roc_plot,
    set_seed,
    split_patients,
    subset_by_patient_ids,
)

# ---------------------------
# EDIT THESE PATHS
# ---------------------------
DRIVE_BASE = "/content/drive/MyDrive/MIDL_PAPER"
RESULTS_DIR = os.path.join(DRIVE_BASE, "resnet18_mlp_results")

PATHS = {
    "PREPROC_CANCER": os.path.join(DRIVE_BASE, "Preprocessed_Cancer_ResNet18"),
    "PREPROC_NONCANCER": os.path.join(DRIVE_BASE, "Preprocessed_No_Cancer_ResNet18"),
    "RAW_CANCER": os.path.join(DRIVE_BASE, "Raw_Cancer_ResNet18"),
    "RAW_NONCANCER": os.path.join(DRIVE_BASE, "Raw_No_Cancer_ResNet18"),
}

INPUT_DIM = 512
BATCH_SIZE = 32
SEED = 42


def build_resnet_head(input_dim: int = INPUT_DIM) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def run_single_condition(prefix: str, cancer_dir: str, noncancer_dir: str) -> dict:
    x, y, pids = combine_folders(cancer_dir, noncancer_dir, input_dim=INPUT_DIM)
    if len(x) == 0:
        raise RuntimeError(f"No data found for condition: {prefix}")

    split = split_patients(pids, y, val_size=0.2, test_size=0.2, seed=SEED)
    x_train, y_train, p_train = subset_by_patient_ids(x, y, pids, split["train"])
    x_val, y_val, p_val = subset_by_patient_ids(x, y, pids, split["val"])
    x_test, y_test, p_test = subset_by_patient_ids(x, y, pids, split["test"])

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_resnet_head(INPUT_DIM)

    ckpt_path = os.path.join(RESULTS_DIR, f"{prefix}_best.weights.h5")
    callback_list = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
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
        epochs=40,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        class_weight=class_weight,
        verbose=1,
    )

    y_prob_test = model.predict(x_test, verbose=0).reshape(-1)

    slice_metrics = compute_binary_metrics(y_test, y_prob_test)
    auc_ci = bootstrap_auc_ci(y_test, y_prob_test)

    patient_df = patient_pooling_predictions(y_test, y_prob_test, p_test)
    mean_metrics = compute_binary_metrics(patient_df["label"].values, patient_df["mean_prob"].values)
    max_metrics = compute_binary_metrics(patient_df["label"].values, patient_df["max_prob"].values)
    top3_metrics = compute_binary_metrics(patient_df["label"].values, patient_df["top3_prob"].values)

    save_history_plot(history, RESULTS_DIR, prefix)
    save_roc_plot(y_test, y_prob_test, RESULTS_DIR, f"{prefix}_slice")
    save_roc_plot(patient_df["label"].values, patient_df["mean_prob"].values, RESULTS_DIR, f"{prefix}_patient_mean")
    patient_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_patient_predictions.csv"), index=False)

    summary = {
        "prefix": prefix,
        "slice_metrics": slice_metrics,
        "slice_auc_ci_95": {"lower": auc_ci[0], "upper": auc_ci[1]},
        "patient_mean_metrics": mean_metrics,
        "patient_max_metrics": max_metrics,
        "patient_top3_metrics": top3_metrics,
    }

    save_json(summary, os.path.join(RESULTS_DIR, f"{prefix}_metrics.json"))
    return summary


def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)

    print("Starting ResNet18 pipeline...")

    preproc_summary = run_single_condition(
        prefix="preproc",
        cancer_dir=PATHS["PREPROC_CANCER"],
        noncancer_dir=PATHS["PREPROC_NONCANCER"],
    )

    raw_summary = run_single_condition(
        prefix="raw",
        cancer_dir=PATHS["RAW_CANCER"],
        noncancer_dir=PATHS["RAW_NONCANCER"],
    )

    save_json(
        {"preproc": preproc_summary, "raw": raw_summary},
        os.path.join(RESULTS_DIR, "resnet18_final_summary.json"),
    )
    print("ResNet18 pipeline done.")


if __name__ == "__main__":
    main()
