# GSoC_DeepLense_PreTask
**ML4SCI | Sourish Kumar Dutta**

This repository contains my evaluation task submissions for the DeepLense project under ML4SCI (Google Summer of Code 2026). I completed both the mandatory Common Test I and the project-specific Test V.

---

## Common Test I — Multi-Class Substructure Classification

The task is to classify simulated strong lensing images into three categories: no substructure, spherical subhalo substructure, and vortex substructure. The vortex class is physically interesting — it would be direct evidence for superfluid dark matter outside the standard WIMP paradigm.

**Model:** EfficientNet-B0 with ImageNet pretrained weights, adapted for single-channel input by averaging RGB weights across the channel dimension. Classifier head replaced with a Linear(1280→3) layer.

**Training:** Two-phase fine-tuning — 3 epochs with frozen backbone (LR=1e-3), then 12 epochs with full unfreezing (LR=1e-4). WeightedRandomSampler for class balance. AdamW + cosine annealing + mixed precision.

**Results:**

| Metric | Value |
|---|---|
| Macro AUC | 0.9893 |
| Overall Accuracy | 93.44% |
| AUC — No Substructure | 0.993 |
| AUC — Sphere Substructure | 0.982 |
| AUC — Vortex Substructure | 0.993 |
| Sphere class recall | 86% |

The sphere class had the lowest recall throughout training. This is expected because spherical subhalos produce diffuse signatures that overlap with the no-substructure class in feature space.

---

## Specific Test V — Gravitational Lens Finding (Real Data)

This task uses real observational HSC galaxy images (3-channel gri, 64×64). The defining challenge is a roughly 100:1 class imbalance between non-lenses and lenses — not just a machine learning problem, but a reflection of how rare lenses actually are in the sky.

**Model:** ResNet18 pretrained on ImageNet. Final layer replaced with Linear(512→1) for binary output.

**Key decisions:**
- BCEWithLogitsLoss with positive class weight set to the empirical non-lens:lens ratio (~100), to penalise false negatives more heavily
- WeightedRandomSampler combined with augmentation on the minority class
- Decision threshold chosen on the validation set to maximise F1, not the default 0.5
- 8 epochs, AdamW, cosine annealing, mixed precision

**Results:**

| Metric | Value |
|---|---|
| Train AUC | 0.9983 |
| Validation AUC | 0.9685 |
| Test AUC | 0.9502 |
| Lenses correctly identified | 143 / 195 (73.3%) |
| False positives | 692 / 19,455 (3.6%) |

The train-to-test AUC gap is a domain shift issue, not a training failure. For context, Parul et al. (2024) showed a simulation-trained ENN dropping from AUROC 0.995 to 0.921 on real HSC data. This ResNet18 baseline at 0.9502 — without any domain adaptation — already exceeds their unadapted real-data result.

---


## Dependencies
```
torch torchvision numpy pandas matplotlib scikit-learn
```

---

*Submitted as part of the GSoC 2026 application to ML4SCI — DeepLense.*
