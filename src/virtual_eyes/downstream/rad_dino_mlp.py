import os

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

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
RESULTS_DIR = os.path.join(DRIVE_BASE, "rad_dino_mlp_results")

PATHS = {
    "PREPROC_CANCER": os.path.join(DRIVE_BASE, "Preprocessed_Cancer_RadDino"),
    "PREPROC_NONCANCER": os.path.join(DRIVE_BASE, "Preprocessed_No_Cancer_RadDino"),
    "RAW_CANCER": os.path.join(DRIVE_BASE, "Raw_Cancer_RadDino"),
    "RAW_NONCANCER": os.path.join(DRIVE_BASE, "Raw_No_Cancer_RadDino"),
}

INPUT_DIM = 768
SEED = 42


def run_umap_visualization(x, domains, out_path):
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
    embedding = reducer.fit_transform(x)

    plt.figure(figsize=(10, 8))
    for dom in np.unique(domains):
        mask = domains == dom
        plt.scatter(embedding[mask, 0], embedding[mask, 1], label=str(dom), s=5, alpha=0.5)
    plt.title("RAD-DINO Embeddings: Preprocessed vs Raw")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_single_condition(prefix: str, cancer_dir: str, noncancer_dir: str) -> dict:
    x, y, pids = combine_folders(cancer_dir, noncancer_dir, input_dim=INPUT_DIM)
    if len(x) == 0:
        raise RuntimeError(f"No data found for condition: {prefix}")

    split = split_patients(pids, y, val_size=0.2, test_size=0.2, seed=SEED)

    x_train, y_train, p_train = subset_by_patient_ids(x, y, pids, split["train"])
    x_val, y_val, p_val = subset_by_patient_ids(x, y, pids, split["val"])
    x_test, y_test, p_test = subset_by_patient_ids(x, y, pids, split["test"])

    model = build_mlp(input_dim=INPUT_DIM, hidden_dims=[512], dropout=0.3, learning_rate=1e-3)
    model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        results_dir=RESULTS_DIR,
        model_name=prefix,
        epochs=30,
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

    print("Running RAD-DINO pipeline...")

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

    x_pre, _, _ = combine_folders(PATHS["PREPROC_CANCER"], PATHS["PREPROC_NONCANCER"], input_dim=INPUT_DIM)
    x_raw, _, _ = combine_folders(PATHS["RAW_CANCER"], PATHS["RAW_NONCANCER"], input_dim=INPUT_DIM)

    n_pre = min(5000, len(x_pre))
    n_raw = min(5000, len(x_raw))
    x_all = np.vstack([x_pre[:n_pre], x_raw[:n_raw]])
    domains = np.array(["Preproc"] * n_pre + ["Raw"] * n_raw)
    run_umap_visualization(x_all, domains, os.path.join(RESULTS_DIR, "umap_rad_dino.png"))

    save_json(
        {"preproc": preproc_summary, "raw": raw_summary},
        os.path.join(RESULTS_DIR, "rad_dino_final_summary.json"),
    )
    print("RAD-DINO pipeline finished.")


if __name__ == "__main__":
    main()
