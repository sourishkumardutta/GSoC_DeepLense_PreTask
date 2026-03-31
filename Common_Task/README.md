# Common Test I — Multi-Class Substructure Classification

Three-class classification of simulated strong gravitational lensing images: no substructure, 
spherical subhalo substructure, and vortex substructure. The vortex class is a signature of 
superfluid dark matter condensates — detecting it would be evidence outside the standard 
WIMP paradigm.

## Contents

- `Final Common Task.ipynb` — full walkthrough: data exploration, training, evaluation, and plots
- `CommonTask.pdf` — PDF export of the notebook for easy viewing without running any code
- `Common_Task_Parameters.pth` — saved best model checkpoint (by validation macro-AUC)
- `auc_history.png` — validation macro-AUC across epochs
- `results.png` — loss curves, accuracy curves, ROC curves, and confusion matrix
- `val_predictions.csv` — raw validation set predictions and ground truth labels

## Model

EfficientNet-B0 pretrained on ImageNet, adapted for single-channel input by averaging the 
RGB weights across the channel dimension. Classifier head replaced with Linear(1280→3).

Two-phase training:
- Phase 1 (3 epochs): backbone frozen, only head trained at LR=1e-3
- Phase 2 (12 epochs): full model unfrozen at LR=1e-4

AdamW optimiser, cosine annealing schedule, mixed precision (torch.cuda.amp).
WeightedRandomSampler used to handle mild class imbalance in the sphere class.

## Results

| Metric | Value |
|---|---|
| Macro AUC | 0.9893 |
| Overall Accuracy | 93.44% |
| AUC — No Substructure | 0.993 |
| AUC — Sphere Substructure | 0.982 |
| AUC — Vortex Substructure | 0.993 |
| Sphere class recall | 86% (2162 / 2500) |

The sphere class consistently had the lowest recall. Spherical subhalos produce diffuse, 
extended lensing signatures that partially overlap with the no-substructure class in feature 
space.

## Reference

Alexander et al. (2019) — arXiv:1909.07346
