# MA-RAFT
This repository provides the official implementation of MA-RAFT, a morphology-aware optical-flow framework for tracer-particle velocimetry under non-ideal imaging conditions.

MA-RAFT is designed for robust dense velocity-field estimation from consecutive particle images, especially when the tracer images suffer from particle morphology variation, photometric fluctuation, background contamination, blur, agglomeration, or other imaging-chain degradation.

<p align="center">
  <img src="assets/demo.gif" width="700">
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

## Overview

Particle image velocimetry (PIV) estimates flow motion from consecutive tracer-particle images. Conventional deep optical-flow models are usually trained on ideal synthetic particles, where particles are compact, stable, and approximately Gaussian. However, real measurements often contain non-ideal effects such as particle deformation, agglomeration, photometric fluctuation, background contamination, and imaging blur.

MA-RAFT addresses this problem by jointly improving feature representation and correspondence estimation:

1. **Adaptive Particle Enhancement Module (APEM)**  
   Learns spatially varying particle support and suppresses low-confidence background or scattering artifacts.

2. **Deformable Cross-Attention (DCA)**  
   Replaces rigid correlation lookup with deformable, multi-scale correspondence aggregation, allowing the matching region to adapt to local particle morphology.

3. **Recurrent Flow Update**  
   Iteratively refines the dense displacement / velocity field in a RAFT-style framework.

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
