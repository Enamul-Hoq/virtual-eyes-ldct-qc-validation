#!/usr/bin/env python3
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import pydicom
from skimage import measure, morphology

warnings.filterwarnings("ignore")

RAW_BASE = "/home/mhoq/moffit_nlst_sybil/data/nlst_root/nlst"
PATIENT_LIST_CSV = "/home/mhoq/virtual_eyes/experiments/nlst_exp1_qc/outputs/manifests/nlst_qc_patient_list_T0_labelT0.csv"
OUT_DIR = "/home/mhoq/virtual_eyes/experiments/nlst_exp1_qc/outputs"

BLOCK_DIR = os.path.join(OUT_DIR, "accepted_lung_blocks_npy")
MANIFEST_DIR = os.path.join(OUT_DIR, "manifests")
os.makedirs(BLOCK_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)


class CTQualityConfig:
    LUNG_HU_MIN = -950
    LUNG_HU_MAX = -700
    MIN_LUNG_VOLUME_RATIO = 0.05
    MIN_RAW_IMAGES = 64
    MIN_LUNG_SCORE_FOR_SLICE = 0.15
    MIN_LUNG_BLOCK_SIZE = 20


def safe_get(ds, name, default=None):
    try:
        return getattr(ds, name, default)
    except Exception:
        return default


def sort_slices(dsets):
    def zpos(ds):
        ipp = safe_get(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
        sl = safe_get(ds, "SliceLocation", None)
        if sl is not None:
            return float(sl)
        inst = safe_get(ds, "InstanceNumber", 0)
        return float(inst)
    return sorted(dsets, key=zpos)


def to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(safe_get(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(safe_get(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def load_series(dicom_files):
    dsets = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            if hasattr(ds, "pixel_array"):
                dsets.append(ds)
        except Exception:
            continue

    if not dsets:
        return None

    dsets = sort_slices(dsets)
    shapes = [ds.pixel_array.shape for ds in dsets]
    mode_shape = max(set(shapes), key=shapes.count)
    dsets = [ds for ds in dsets if ds.pixel_array.shape == mode_shape]

    if len(dsets) == 0:
        return None

    vol = np.stack([to_hu(ds) for ds in dsets], axis=0)
    return vol


def improved_lung_detection_hu(hu_slice):
    lung_mask1 = (hu_slice >= CTQualityConfig.LUNG_HU_MIN) & (hu_slice <= CTQualityConfig.LUNG_HU_MAX)
    body_mask = hu_slice > -500
    lung_mask2 = (hu_slice >= -1000) & (hu_slice <= -400) & body_mask
    lung_mask = lung_mask1 | lung_mask2

    if np.sum(lung_mask) == 0:
        return np.zeros_like(hu_slice, dtype=bool)

    labeled = measure.label(lung_mask)
    regions = measure.regionprops(labeled)
    min_area = hu_slice.size * CTQualityConfig.MIN_LUNG_VOLUME_RATIO
    clean = np.zeros_like(lung_mask, dtype=bool)

    for r in regions:
        if r.area >= min_area and r.eccentricity < 0.95:
            clean[r.coords[:, 0], r.coords[:, 1]] = True

    if np.sum(clean) > 0:
        clean = morphology.binary_opening(clean, morphology.disk(1))
        clean = morphology.binary_closing(clean, morphology.disk(2))

    return clean


def enhanced_lung_presence_check_hu(hu_slice):
    lung_mask = improved_lung_detection_hu(hu_slice)
    total = hu_slice.size
    lung_area_ratio = float(np.sum(lung_mask) / total) if total > 0 else 0.0

    h, w = hu_slice.shape
    mid = w // 2
    left = lung_mask[:, :mid]
    right = lung_mask[:, mid:]

    left_cov = float(np.sum(left) / left.size) if left.size else 0.0
    right_cov = float(np.sum(right) / right.size) if right.size else 0.0

    bilateral_score = min(left_cov, right_cov) * 2
    bilateral_score = min(bilateral_score, 1.0)

    area_score = min(lung_area_ratio / 0.15, 1.0)
    final_score = (area_score * 0.6 + bilateral_score * 0.4)

    keep = (
        lung_area_ratio >= CTQualityConfig.MIN_LUNG_VOLUME_RATIO and
        bilateral_score >= 0.1 and
        final_score >= CTQualityConfig.MIN_LUNG_SCORE_FOR_SLICE
    )
    return keep, final_score, lung_area_ratio


def find_lung_blocks(slice_results):
    blocks = []
    cur = None

    for i, r in enumerate(slice_results):
        if r["keep"]:
            if cur is None:
                cur = {"start": i, "end": i, "slices": [i]}
            else:
                cur["end"] = i
                cur["slices"].append(i)
        else:
            if cur is not None:
                blocks.append(cur)
                cur = None

    if cur is not None:
        blocks.append(cur)

    for b in blocks:
        b["length"] = b["end"] - b["start"] + 1
        scores = [slice_results[j]["lung_score"] for j in b["slices"]]
        areas = [slice_results[j]["lung_area"] for j in b["slices"]]
        b["median_score"] = float(np.median(scores))
        b["median_area"] = float(np.median(areas))

    return blocks


def discover_patient_series(raw_base, patient_ids):
    raw_base = Path(raw_base)
    all_series = []

    for pid in tqdm(patient_ids, desc="Discovering series"):
        pdir = raw_base / str(pid)
        if not pdir.exists():
            continue

        for sd in [d for d in pdir.rglob("*") if d.is_dir()]:
            dcm_files = sorted([str(x) for x in sd.glob("*.dcm")])
            if len(dcm_files) == 0:
                continue

            try:
                hdr = pydicom.dcmread(dcm_files[0], stop_before_pixels=True, force=True)
                series_uid = str(safe_get(hdr, "SeriesInstanceUID", sd.name))
                series_desc = str(safe_get(hdr, "SeriesDescription", ""))
                modality = str(safe_get(hdr, "Modality", ""))
            except Exception:
                series_uid = sd.name
                series_desc = ""
                modality = ""

            if modality.upper() != "CT":
                continue

            all_series.append({
                "pid": str(pid),
                "series_uid": series_uid,
                "series_name": series_desc,
                "dicom_files": dcm_files,
                "series_dir": str(sd),
            })

    return all_series


def process_series(s):
    vol = load_series(s["dicom_files"])
    if vol is None:
        return None

    if vol.shape[0] < CTQualityConfig.MIN_RAW_IMAGES:
        return {
            "pid": s["pid"],
            "series_uid": s["series_uid"],
            "series_name": s["series_name"],
            "status": "REJECTED",
            "reason": f"Insufficient slices: {vol.shape[0]}",
            "original_slices": vol.shape[0],
            "kept_slices": 0,
            "lung_block_npy_path": ""
        }

    if vol.shape[1:] != (512, 512):
        return {
            "pid": s["pid"],
            "series_uid": s["series_uid"],
            "series_name": s["series_name"],
            "status": "REJECTED",
            "reason": f"Non-512x512 resolution: {vol.shape[1:]}",
            "original_slices": vol.shape[0],
            "kept_slices": 0,
            "lung_block_npy_path": ""
        }

    slice_results = []
    for i in range(vol.shape[0]):
        keep, score, area = enhanced_lung_presence_check_hu(vol[i])
        slice_results.append({
            "slice_idx": i,
            "keep": bool(keep),
            "lung_score": float(score),
            "lung_area": float(area),
        })

    blocks = find_lung_blocks(slice_results)
    if not blocks:
        return {
            "pid": s["pid"],
            "series_uid": s["series_uid"],
            "series_name": s["series_name"],
            "status": "REJECTED",
            "reason": "No valid lung blocks found",
            "original_slices": vol.shape[0],
            "kept_slices": 0,
            "lung_block_npy_path": ""
        }

    block = max(blocks, key=lambda b: b["length"])
    if block["length"] < CTQualityConfig.MIN_LUNG_BLOCK_SIZE:
        return {
            "pid": s["pid"],
            "series_uid": s["series_uid"],
            "series_name": s["series_name"],
            "status": "REJECTED",
            "reason": f"Lung block too small: {block['length']}",
            "original_slices": vol.shape[0],
            "kept_slices": block["length"],
            "lung_block_npy_path": ""
        }

    lung_block = vol[block["start"]:block["end"] + 1].copy()
    out_dir = os.path.join(BLOCK_DIR, s["pid"], s["series_uid"].replace(".", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(out_dir, "lung_block.npy")
    np.save(out_npy, lung_block.astype(np.float32))

    return {
        "pid": s["pid"],
        "series_uid": s["series_uid"],
        "series_name": s["series_name"],
        "status": "ACCEPTED",
        "reason": "N/A",
        "original_slices": vol.shape[0],
        "kept_slices": block["length"],
        "lung_start_idx": block["start"],
        "lung_end_idx": block["end"],
        "lung_block_length": block["length"],
        "median_lung_score": block["median_score"],
        "median_lung_area": block["median_area"],
        "lung_block_npy_path": out_npy,
    }


def main():
    plist = pd.read_csv(PATIENT_LIST_CSV)
    patient_ids = plist["pid"].astype(str).tolist()

    all_series = discover_patient_series(RAW_BASE, patient_ids)
    print(f"Discovered CT series: {len(all_series)}")

    rows = []
    for s in tqdm(all_series, desc="QC processing"):
        try:
            row = process_series(s)
            if row is not None:
                rows.append(row)
        except Exception as e:
            traceback.print_exc()
            rows.append({
                "pid": s["pid"],
                "series_uid": s["series_uid"],
                "series_name": s["series_name"],
                "status": "REJECTED",
                "reason": f"Unhandled exception: {str(e)}",
                "original_slices": 0,
                "kept_slices": 0,
                "lung_block_npy_path": ""
            })

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(MANIFEST_DIR, "qc_results_nlst_T0_labelT0.csv")
    results_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("\nStatus counts:")
    print(results_df["status"].value_counts())
    acc = results_df[results_df["status"] == "ACCEPTED"]
    print("\nAccepted patients:", acc["pid"].nunique())
    print("Accepted series:", len(acc))
    if not acc.empty:
        print("\nAccepted kept_slices summary:")
        print(acc["kept_slices"].describe())


if __name__ == "__main__":
    main()

