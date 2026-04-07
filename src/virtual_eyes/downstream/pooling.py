import numpy as np
from sklearn.metrics import roc_auc_score


def patient_pooling_eval(y_true_slices, y_pred_probs, pids):
    results = {}
    unique_pids = np.unique(pids)

    y_true_patient = []
    y_pred_mean = []
    y_pred_max = []
    y_pred_top3 = []

    for pid in unique_pids:
        mask = pids == pid
        probs = np.asarray(y_pred_probs)[mask].reshape(-1)
        labels = np.asarray(y_true_slices)[mask].reshape(-1)

        y_true_patient.append(int(labels[0]))
        y_pred_mean.append(float(np.mean(probs)))
        y_pred_max.append(float(np.max(probs)))
        y_pred_top3.append(float(np.mean(np.sort(probs)[-3:])) if len(probs) >= 3 else float(np.mean(probs)))

    y_true_patient = np.array(y_true_patient)
    y_pred_mean = np.array(y_pred_mean)
    y_pred_max = np.array(y_pred_max)
    y_pred_top3 = np.array(y_pred_top3)

    if len(np.unique(y_true_patient)) > 1:
        results["Mean_AUC"] = float(roc_auc_score(y_true_patient, y_pred_mean))
        results["Max_AUC"] = float(roc_auc_score(y_true_patient, y_pred_max))
        results["Top3_AUC"] = float(roc_auc_score(y_true_patient, y_pred_top3))
    else:
        results["Mean_AUC"] = float("nan")
        results["Max_AUC"] = float("nan")
        results["Top3_AUC"] = float("nan")

    return results
