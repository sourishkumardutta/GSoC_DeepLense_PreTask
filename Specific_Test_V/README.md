# Specific Test V — Gravitational Lens Finding

Binary classification on real observational HSC galaxy images: lens vs non-lens.
Each image is 3-channel (g, r, i filters), 64×64 pixels.

The central challenge is a ~100:1 class imbalance between non-lenses and lenses.
This is not just a machine learning nuisance — it reflects the actual rarity of lenses 
in real surveys, where even a 1% false positive rate produces an unmanageable 
number of false positives for human reviewers.

## Contents

- `Final Task 5.ipynb` — full walkthrough: data analysis, training, ROC curve, confusion matrix
- `Task5.pdf` — PDF export of the notebook for easy viewing without running any code
- `Model_Weights.pth` — saved best model checkpoint (by validation AUC)
- `ROC-AUC Curve.png` — ROC curves for train, validation, and test sets
- `confusion_matrix.png` — confusion matrix on the test set
- `training-epoch-curves.png` — loss, accuracy, and AUC across 8 training epochs

## Model

ResNet18 pretrained on ImageNet. The three input channels map naturally to the three 
survey filters (g, r, i). Final fully connected layer replaced with Linear(512→1) for 
binary output.

Key decisions:
- BCEWithLogitsLoss with positive class weight set to the empirical non-lens:lens ratio (~100),
  directly penalising false negatives more than false positives during training
- WeightedRandomSampler to create roughly balanced batches, combined with augmentation
  applied to the minority lens class
- Decision threshold selected on the validation set to maximise F1 — the default 0.5 
  is almost never optimal under severe class imbalance
- 8 epochs, AdamW, cosine annealing, mixed precision

## Results

| Metric | Value |
|---|---|
| Train AUC | 0.9983 |
| Validation AUC | 0.9685 |
| Test AUC | 0.9502 |
| Lenses correctly identified | 143 / 195 (73.3%) |
| False positives | 692 / 19,455 (3.6%) |

The train-to-test AUC drop reflects domain shift, not overfitting. Parul et al. (2024) 
showed that a simulation-trained ENN falls from AUROC 0.995 to 0.921 on real HSC 
data without domain adaptation. This baseline at 0.9502 already exceeds that, and 
serves as the starting point for the domain adaptation work planned during GSoC.


## Reference

Parul et al. (2024) — NeurIPS ML4PS Workshop
