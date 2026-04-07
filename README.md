# Virtual-Eyes: Quantitative Validation of a Lung CT Quality-Control Pipeline for Foundation-Model Cancer Risk Prediction

Official repository for the MIDL 2026 paper:

**Virtual-Eyes: Quantitative Validation of a Lung CT Quality-Control Pipeline for Foundation-Model Cancer Risk Prediction**

---

## Overview

Virtual-Eyes is a deterministic, lung-aware 16-bit CT quality-control pipeline for low-dose CT (LDCT) lung cancer screening. This repository provides a **two-stage reproducible pipeline**:

1. **Quality control / preprocessing (Virtual-Eyes)**
2. **Downstream evaluation** of RAD-DINO, Merlin, Sybil, and ResNet-18

![Virtual-Eyes pipeline](assets/midl_prep_pipeline.jpeg)

---

## Data

NLST low-dose CT data are available from **The Cancer Imaging Archive (TCIA)**.

This repository does **not** distribute NLST images or patient data.

---

## Repository structure

```text
virtual-eyes-ldct-qc-validation/
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
│   └── midl_prep_pipeline.jpeg
├── scripts/
│   ├── 02_run_virtual_eyes_qc.py
│   ├── 04_run_rad_dino_mlp.py
│   ├── 05_run_merlin_mlp.py
│   ├── 06_run_sybil_eval.py
│   └── 07_run_resnet18_train.py
├── src/
│   └── virtual_eyes/
│       ├── __init__.py
│       ├── qc/
│       │   ├── __init__.py
│       │   └── run_qc.py
│       └── downstream/
│           ├── __init__.py
│           ├── common.py
│           ├── pooling.py
│           ├── rad_dino_mlp.py
│           ├── merlin_mlp.py
│           ├── sybil_eval.py
│           └── resnet18_train.py
├── docs/
│   └── REPRODUCING_NLST_EXPERIMENTS.md
└── data/
    └── README.md