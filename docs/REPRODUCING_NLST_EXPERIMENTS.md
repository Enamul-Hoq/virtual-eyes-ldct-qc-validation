# Reproducing the NLST Experiments

This guide follows the two-stage pipeline described in the Virtual-Eyes MIDL 2026 paper.

---

## Stage 1: Quality control / preprocessing

The Virtual-Eyes QC stage performs:

- CT series discovery
- rejection of short/non-diagnostic scans
- enforcement of 512 × 512 resolution
- lung detection (HU-based)
- contiguous lung block extraction
- saving `lung_block.npy`
- QC CSV generation

### Default QC parameters

- lung HU range: `[-950, -700]`
- minimum lung volume ratio: `0.05`
- minimum raw images: `64`
- minimum lung score for slice: `0.15`
- minimum lung block size: `20`

### Run QC

```bash
python scripts/02_run_virtual_eyes_qc.py