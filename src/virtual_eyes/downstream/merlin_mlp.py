import os

import numpy as np
from scipy.stats import ks_2samp

from virtual_eyes.downstream.common import (
    bootstrap_auc_ci,
    build_mlp,
    combine_folders,
    compute_binary_metrics,
    ensure_dir,
    patient_pooling_predictions,
    save_history_plot,
    save_json,
    save_roc_plot,
    set_seed,
    split_patients,
    subset_by_patient_ids,
    train_model,
)

# ---------------------------
# EDIT THESE PATHS
# ---------------------------
DRIVE_BASE = "/content/drive/MyDrive/MIDL_PAPER"
RESULTS_DIR = os.path.join(DRIVE_BASE, "merlin_mlp_results_final")

PATHS = {
    "PREPROC_CANCER": os.path.join(DRIVE_BASE, "Preprocessed_Cancer_Merlin"),
    "PREPROC_NONCANCER": os.path.join(DRIVE_BASE, "Preprocessed_No_Cancer_Merlin"),
    "RAW_CANCER": os.path.join(DRIVE_BASE, "Raw_Cancer_Merlin"),
    "RAW_NONCANCER": os.path.join(DRIVE_BASE, "Raw_No_Cancer_Merlin"),
}

INPUT_DIM = 2048
SEED = 42


def run_single_condition(prefix: str, cancer_dir: str, noncancer_dir: str) -> dict:
    x, y, pids = combine_folders(cancer_dir, noncancer_dir, input_dim=INPUT_DIM)
    if len(x) == 0:
        raise RuntimeError(f"No data found for condition: {prefix}")

    split = split_patients(pids, y, val_size=0.2, test_size=0.2, seed=SEED)

    x_train, y_train, p_train = subset_by_patient_ids(x, y, pids, split["train"])
    x_val, y_val, p_val = subset_by_patient_ids(x, y, pids, split["val"])
    x_test, y_test, p_test = subset_by_patient_ids(x, y, pids, split["test"])

    model = build_mlp(input_dim=INPUT_DIM, hidden_dims=[512, 256], dropout=0.3, learning_rate=1e-4)
    model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        results_dir=RESULTS_DIR,
        model_name=prefix,
        epochs=40,
        batch_size=128,
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
        "num_train_slices": int(len(x_train)),
        "num_val_slices": int(len(x_val)),
        "num_test_slices": int(len(x_test)),
        "num_train_patients": int(len(np.unique(p_train))),
        "num_val_patients": int(len(np.unique(p_val))),
        "num_test_patients": int(len(np.unique(p_test))),
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

    print("Running Merlin MLP pipeline...")
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

    ks_stat, ks_p = ks_2samp(
        [preproc_summary["patient_mean_metrics"]["auc"]],
        [raw_summary["patient_mean_metrics"]["auc"]],
    )

    final_summary = {
        "preproc": preproc_summary,
        "raw": raw_summary,
        "ks_summary_note": {
            "statistic": float(ks_stat),
            "pvalue": float(ks_p),
            "note": "This placeholder KS comparison is at summary level. Replace with score-distribution KS if you have saved raw patient probabilities from both conditions on matched cohorts."
        },
    }

    save_json(final_summary, os.path.join(RESULTS_DIR, "merlin_final_summary.json"))
    print("Merlin pipeline finished.")


if __name__ == "__main__":
    main()
