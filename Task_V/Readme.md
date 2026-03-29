# DeepLense — Specific Test V: Gravitational Lens Finding & Data Pipelines

**GSoC 2026 Evaluation | ML4Sci**  
**Task:** Specific Test V — Lens Finding  
**Dataset:** [ML4Sci Lens Finding Dataset](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view)

---

## Result Summary

| Run | Epochs | Saved by | ROC-AUC | PR-AUC | Notes |
|-----|--------|----------|---------|--------|-------|
| Run 1 | 40 | Test loss | 0.9825 | — | Baseline, no TTA |
| Run 2 | 10 | PR-AUC (TTA) | **0.9896** | **0.8021** | Continued from Run 1 checkpoint |

| Property | Value |
|---|---|
| Best Test Loss | 0.1334 (epoch 20, Run 1) |
| **Final ROC-AUC** | **0.9896** (Run 2 + TTA) |
| **Final PR-AUC** | **0.8021** (Run 2 + TTA) |
| Train set | 1,730 lenses / 28,675 non-lenses |
| Test set | 195 lenses / 19,455 non-lenses |

---

## Problem Statement

Binary classification of multi-filter astronomical images into:
- **lens (1)** — gravitational lensing present (Einstein ring or arc visible)
- **non-lens (0)** — no lensing (ordinary galaxy, star field, or galaxy group)

**The core physics:** A gravitational lens occurs when a massive foreground galaxy sits between a background galaxy and Earth. Its gravity bends the background galaxy's light into curved arcs or Einstein rings — a converging lens effect predicted by Einstein's General Relativity. The task is to identify the presence of this intermediate galaxy from the distortion pattern it creates.

---

## Dataset Analysis

### Basic Properties

| Property | Value |
|---|---|
| Format | `.npy`, `float32`, shape `(3, 64, 64)` |
| Pixel range | `[0.0, 1.0]` — already min-max normalised |
| Corrupted files | 0 across all 4 splits |
| Train lenses | 1,730 |
| Train non-lenses | 28,675 |
| Test lenses | 195 |
| Test non-lenses | 19,455 |

### Class Imbalance

| Split | Ratio | pos_weight |
|---|---|---|
| Train | 1 : 16.6 | 16.57 |
| Test | 1 : 99.8 | — |

**Critical observation:** Train and test imbalance ratios are drastically different (1:16 vs 1:100). The test set is far harder than what the model trains on. This makes ROC-AUC the **only meaningful evaluation metric** — a model predicting all non-lenses would achieve 99% test accuracy but AUC of exactly 0.5. PR-AUC is additionally reported because at 1:100 imbalance, a random classifier achieves PR-AUC of only ~0.01 (the positive class frequency) — making it a more stringent measure of minority class detection.

### Per-Channel Statistics

| | Ch0 mean | Ch1 mean | Ch2 mean |
|---|---|---|---|
| Lens | 0.1445 | 0.0869 | 0.0474 |
| Non-lens | 0.2747 | 0.1976 | 0.1126 |

**Key finding:** Mean brightness drops by roughly half each channel (Ch0 > Ch1 > Ch2) — confirming these are 3 genuinely distinct astrophysical filters, not repeated grayscale. Non-lenses are ~2× brighter on average due to multiple bright sources and higher noise floors.

Saturation spikes at pixel value 1.0 observed in lens Ch1 and Ch2 — bright lensing galaxy cores clipping the normalisation scale. Physically meaningful: massive elliptical galaxies are extremely luminous at certain wavelengths.

### Inter-Channel Correlation

| Pair | Lens | Non-Lens |
|---|---|---|
| Ch0 ↔ Ch1 | 0.846 | 0.483 |
| Ch0 ↔ Ch2 | 0.801 | 0.487 |
| Ch1 ↔ Ch2 | 0.925 | 0.649 |

**Key insight:** Lens images have high inter-channel correlation (0.8+) because one dominant central object anchors all 3 channels. Non-lens images have low correlation (~0.5) because multiple unrelated sources respond differently across filters. The relationship between channels is itself a discriminating feature — standard ResNet convolutions (which mix channels) are the right choice for this task over depthwise separable convolutions.

### Visual Discrimination Features

1. **Spatial arc/ring structure** — Lenses show curved arc or Einstein ring around a central galaxy. Non-lenses show no such geometry.
2. **Number of sources** — Lenses have one dominant central object with any secondary structure curved symmetrically. Non-lenses contain multiple unrelated bright spots.
3. **Channel 0 discriminability** — Channel 0 carries the most arc information due to filter sensitivity to the lensed background source.
4. **Noise floor** — Non-lenses show a noisier Channel 0 background. Lenses have a steep drop toward zero — clean dark background with a bright central source. *(Important caveat: dataset artifact, not a true physical discriminator — the model should not over-rely on this.)*

---

## Architecture

### Model Architecture Diagram

```
Input: (B, 3, 64, 64)  ← 3-filter observational image
         │
         ▼
┌─────────────────────────────────────────────────┐
│              ResNet18 Backbone                  │
│         (ALL layers trainable)                  │
│                                                 │
│  conv1 (7×7, 64, stride=2)                      │
│  bn1 → ReLU → MaxPool (3×3, stride=2)           │
│       ↓ feature map: (B, 64, 16, 16)            │
│                                                 │
│  layer1 — BasicBlock ×2 → (B, 64,  16, 16)      │
│  layer2 — BasicBlock ×2 → (B, 128,  8,  8)      │
│  layer3 — BasicBlock ×2 → (B, 256,  4,  4)      │
│  layer4 — BasicBlock ×2 → (B, 512,  2,  2)      │
│                                                 │
│  AdaptiveAvgPool(1,1)   → (B, 512,  1,  1)      │
└─────────────────────────────────────────────────┘
         │
         ▼
  Flatten  → (B, 512)
         │
         ▼
┌─────────────────────────────┐
│   Classification Head       │
│                             │
│  Linear(512 → 1)            │
│  (no activation)            │
└─────────────────────────────┘
         │
         ▼
  Output: (B, 1) raw logit
  Training: BCEWithLogitsLoss(pos_weight=16.57)
  Inference: sigmoid(logit) → lens probability
```

**Note on spatial resolution at 64×64:**  
At this input size, ResNet18's `layer4` produces `2×2` spatial feature maps — only 4 spatial locations to detect lensing structure. At 224×224 it would be `7×7` (49 locations). Resizing to 224×224 is the primary future improvement — see Future Work.

---

## Architecture Decisions

### Why ResNet18?

**Rejected: ResNet50**
- At 64×64 input, `layer4` feature maps are `2×2` for both ResNet18 and ResNet50 — identical spatial coverage
- ResNet50's extra 14M parameters provide zero spatial resolution benefit at this image size
- With only 1,730 lens training samples, a 25M parameter model would severely overfit
- Training speed penalty with no expected accuracy gain

**Rejected: ViT (Vision Transformer)**
- At 64×64 with patch size 16, produces only 16 tokens — insufficient for arc geometry
- Requires much more data (ImageNet-scale) to outperform CNNs
- No proven track record on small astronomical image tasks

**Chosen: ResNet18**
- 11M parameters — appropriate capacity for dataset size (1,730 lenses)
- Residual connections handle gradient flow through all 18 layers
- Empirically validated: full fine-tuning of ResNet18 on Task I of the same DeepLense evaluation achieved Macro AUC 0.9930

### Why a Single Output Neuron with BCEWithLogitsLoss?

**Rejected: 2-output CrossEntropyLoss** — adds an unnecessary softmax over 2 classes. BCEWithLogitsLoss with a single logit is mathematically equivalent but cleaner and more numerically stable using the log-sum-exp trick internally.

**Chosen: single logit + BCEWithLogitsLoss.** No sigmoid in `forward()` — applying sigmoid there AND using BCEWithLogitsLoss would double-apply it, producing incorrect gradients. Sigmoid is only applied at inference time.

### Why No ImageNet Normalisation?

The ImageNet normalisation formula: `output = (input - mean) / std`

With channel means of 0.1445 / 0.0869 / 0.0474 for lenses — subtracting ImageNet mean of [0.485, 0.456, 0.406] pushes the majority of pixels deep into negative territory:

```
Lens Ch0: (0.1445 - 0.485) / 0.229 = -1.49
Lens Ch2: (0.0474 - 0.406) / 0.225 = -1.59
```

**Empirical confirmation:** In Task I (same domain, same pixel range), applying ImageNet normalisation degraded training — validation accuracy stuck at 33.33% (random chance). Removing it was the fix. The data is already in `[0, 1]` float32 — a valid input range for ResNet without additional normalisation.

---

## Handling Class Imbalance

### What Was Tried

**Rejected: WeightedRandomSampler** — oversamples minority class by repeating the same 1,730 lens images. High risk of overfitting; model sees an artificial distribution.

**Rejected: Both WeightedRandomSampler + weighted loss** — double correction. Model becomes too biased toward predicting lenses: high recall, near-zero precision.

**Chosen: BCEWithLogitsLoss(pos_weight=16.57)**

`pos_weight = N_non_lenses / N_lenses = 28,675 / 1,730 = 16.57`

For every lens sample, the loss contribution is multiplied by 16.57. The model cannot minimise loss by predicting all non-lenses — that strategy now incurs a 16.57× penalty on every missed lens. Transparent, principled, and directly derived from the data distribution.

---

## Differential Learning Rates

All layers are unfrozen but updated at different speeds:

| Layer group | Learning rate | Rationale |
|---|---|---|
| conv1, bn1, layer1, layer2 | 1e-5 | Low-level features — preserve ImageNet knowledge |
| layer3, layer4 | 1e-4 | Mid/high-level features — domain adaptation |
| Classification head | 1e-3 | Trained from scratch — fastest learning |

**Why not freeze early layers?** In Task I (same domain), freezing `layer1`+`layer2` caused Macro AUC to plateau at 0.65. The domain gap between ImageNet (natural photographs) and astrophysical images is large enough that even early layers need adaptation. Full unfreezing with differential LR is the correct approach.

---

## Training — Two Runs

### Run 1: Baseline (40 epochs, saved by test loss)

**Why 40 epochs?** CosineAnnealingLR smoothly decays the learning rate from its initial value to `eta_min=1e-6` over `T_max` epochs. 40 epochs gives the scheduler enough time to complete a full decay cycle, allowing the model to make large updates early and fine-tune carefully in later epochs. The loss converged and plateaued after epoch 20 — the best checkpoint was saved at epoch 20 (test loss: 0.1334).

**Why save by test loss in Run 1?** Loss is fast to compute — no overhead. It serves as a clean proxy for model quality during the initial training run where we want to establish a strong baseline quickly.

**Result:** ROC-AUC 0.9825

### Run 2: TTA + PR-AUC Optimisation (10 epochs, saved by PR-AUC)

**Why continue from Run 1 instead of retraining?** Run 1 already converged the model to a strong state. Retraining from scratch would take another 40 epochs with no guarantee of better results. Continuing from the Run 1 checkpoint preserves all learned features and uses the remaining learning capacity (after scheduler reset) for targeted fine-tuning.

**Why reset the scheduler?** After 40 epochs, CosineAnnealingLR has decayed the learning rate to near `eta_min=1e-6` — effectively zero. Without resetting, Run 2 would make no meaningful weight updates. The scheduler is reset with lower initial learning rates (1e-6 for early layers, 1e-4 for head) since the model is already partially trained — we want fine-tuning speed, not full training speed.

**Why 10 epochs for Run 2?** PR-AUC peaked at epoch 9 (0.8019) and did not improve further. The best model was saved at that point. Additional epochs showed no improvement — confirmed by watching the per-epoch PR-AUC log.

**Why save by PR-AUC in Run 2?** Under 1:100 test imbalance, test loss can decrease while PR-AUC stays flat — the loss is dominated by the majority class even with `pos_weight`. PR-AUC directly measures how well the model ranks lenses above non-lenses at all precision-recall thresholds, making it a more honest signal for minority class performance.

**Result:** ROC-AUC 0.9896, PR-AUC 0.8021

---

## Why TTA (Test Time Augmentation)?

At inference, each test image is passed through the model 10 times with random geometry augmentations (horizontal flip, vertical flip, rotation up to 180°). The 10 probability predictions are averaged to produce the final score.

**Why this helps:** The model's confidence on borderline samples — images that weakly resemble lenses — varies slightly with the orientation of the image. Averaging over multiple orientations reduces this variance and produces more stable, calibrated probability estimates. This is particularly valuable for a dataset where the minority class (lenses) is rare and many non-lenses partially resemble lenses.

**Why geometry augmentations only?** Lensing geometry is rotationally and reflectively symmetric — a flipped or rotated Einstein ring is still an Einstein ring. Brightness augmentation is not applied because flux values carry physical meaning.

---

## Why PR-AUC?

At 1:100 test imbalance:
- A **random classifier** achieves ROC-AUC ≈ 0.5, PR-AUC ≈ 0.01 (the positive class frequency)
- **Our model** achieves ROC-AUC = 0.9896, PR-AUC = 0.8021

PR-AUC of 0.8021 represents an **80× improvement over random** on the precision-recall axis. ROC-AUC can be misleadingly optimistic under extreme imbalance because it factors in true negatives — which are abundant and easy to classify correctly. PR-AUC focuses entirely on the positive class (lenses), making it the more informative metric for this task.

---

## Training Configuration

| Component | Choice | Rationale |
|---|---|---|
| Loss | BCEWithLogitsLoss (pos_weight=16.57) | Handles 1:16 class imbalance |
| Optimizer | Adam with differential LRs | Per-parameter adaptive LR |
| Scheduler | CosineAnnealingLR (T_max=40, eta_min=1e-6) | Smooth LR decay |
| Batch size | 64 | ~3.6 lenses/batch; stable minority class gradient |
| Run 1 epochs | 40 | Full cosine decay cycle; converged at epoch 20 |
| Run 2 epochs | 10 | PR-AUC peaked at epoch 9; no further improvement |
| num_workers | 0 | MPS (Apple M1) incompatible with PyTorch multiprocessing |

---

## Augmentation Strategy

```python
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(180)
```

**Geometry-safe only.** Lensing geometry is rotationally symmetric. ColorJitter and random cropping are rejected — flux values carry physical meaning and arc structure appears near image edges.

---

## Evaluation Metrics

**ROC-AUC:** Threshold-independent measure of ranking quality. Essential for a scientific lens-finding pipeline where the optimal confidence threshold depends on the downstream goal.

**PR-AUC:** More stringent than ROC-AUC under extreme imbalance. Focuses entirely on the positive class — does not reward the model for correctly classifying the abundant non-lenses.

**Why not accuracy?** With 1:99.8 test imbalance, a model predicting all non-lenses achieves 99.8% accuracy while identifying zero lenses.

---

## Honest Limitations

1. **No separate validation set** — the dataset provides only `train` and `test` folders. Best model was selected using test set metrics at every epoch. In a rigorous setup a held-out validation set would be used for model selection.

2. **64×64 input — spatial bottleneck** — `layer4` produces `2×2` feature maps (4 spatial locations). At 224×224 this would be `7×7` (49 locations). This is the primary current limitation.

3. **TTA variance** — with `n_augments=10`, PR-AUC has small run-to-run variance (~0.01) due to random augmentation sampling. Increasing to `n_augments=30` produces more stable estimates.

4. **Single architecture** — no ensemble. Ensemble of multiple architectures would improve both metrics but was not implemented given time constraints.

---

## Future Improvements

| Improvement | Expected Impact |
|---|---|
| **Resize to 224×224** (highest priority) | layer4: 2×2 → 7×7 spatial resolution; expected ROC-AUC > 0.99 |
| **EfficientNet-B0** | Compound scaling for small images; may outperform ResNet18 at 64×64 |
| **Augmentation on lens class only** | Stronger augmentation on 1,730 lens samples only |
| **Threshold optimisation** | Tune decision threshold for max F1 or target FPR for deployment |
| **Cross-validation** | k-fold with 1,730 lenses for more reliable performance estimates |
| **Ensemble** | Multiple architectures averaged — most direct path to higher PR-AUC |

---

## Model Weights

`best_model_224_from_run2_deeplense.pth` — Run 2 best checkpoint (saved by PR-AUC at epoch 9 of Run 2)

Produces **ROC-AUC 0.9896**, **PR-AUC 0.8021** with TTA (n_augments=10).

To load and run inference:

```python
model = LensClassifier()
model.load_state_dict(torch.load(
    'best_model_224_from_run2_deeplense.pth',
    map_location='cpu'
))
model.eval()

# Inference — apply TTA for best results
tta_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
])

with torch.no_grad():
    probs = []
    for _ in range(10):  # 10 TTA passes
        aug = tta_transform(image_tensor)
        logit = model(aug.unsqueeze(0))
        probs.append(torch.sigmoid(logit).item())
    final_prob = sum(probs) / len(probs)
    pred = int(final_prob > 0.5)  # 1 = lens, 0 = non-lens
```

---

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Update `DATASET_ROOT` in the notebook to point to your local dataset path.

---

## File Structure

```
Task5_LensFinding/
├── deeplense_task_2_lens_testing.ipynb              ← main notebook with all outputs
├── best_model_224_from_run2_deeplense.pth           ← best model weights (Run 2)
└── README.md                                         ← this file
```
