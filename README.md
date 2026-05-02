# MA-RAFT
This repository provides the official implementation of MA-RAFT, a morphology-aware optical-flow framework for tracer-particle velocimetry under non-ideal imaging conditions.

MA-RAFT is designed for robust dense velocity-field estimation from consecutive particle images, especially when the tracer images suffer from particle morphology variation, photometric fluctuation, background contamination, blur, agglomeration, or other imaging-chain degradation.

<p align="center">
  <img src="assets/demo.mp4" width="700">
</p>

---

## Highlights

- **Morphology-aware PIV estimation** under non-ideal tracer-particle imaging.
- **Adaptive Particle Enhancement Module (APEM)** for suppressing background interference and enhancing degraded particle features.
- **Deformable Cross-Attention (DCA)** for geometry-adaptive correspondence matching.
- **RAFT-style recurrent refinement** for dense optical-flow / velocity-field prediction.
- Supports evaluation on:
  - **Non-Ideal PIV Dataset**
  - **PIV Dataset I**
  - simulated particle-image pairs

---

## Repository Structure

```text
MA-RAFT/
├── README.md
├── requirements.txt
├── train.py
├── test_simulated.py
├── core/
│   ├── model.py
│   ├── apem.py
│   ├── dca.py
│   └── ...
├── weights/
│   ├── ma_raft_non_ideal.pth
│   └── ma_raft_pivdataset1.pth
├── datasets/
│   └── README.md
├── assets/
│   └── demo.gif
└── results/
    └── examples/
```
---

## Datasets

### 1. Non-Ideal PIV Dataset

The Non-Ideal PIV Dataset is designed for evaluating particle image velocimetry under non-ideal imaging conditions, including particle morphology variation, photometric fluctuation, background contamination, blur, and particle agglomeration.

You can download the dataset from Quark Drive:

| Dataset | Download Link | Extraction Code |
|---|---|---|
| Non-Ideal PIV Dataset | [Download](https://pan.quark.cn/s/6070cd255eb1?pwd=sx1b) | `sx1b` |
