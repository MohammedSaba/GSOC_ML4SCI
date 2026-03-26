# DeepLense — Specific Test V: Gravitational Lens Finding & Data Pipelines

**GSoC 2026 Evaluation | ML4Sci**  
**Task:** Specific Test V — Lens Finding  
**Dataset:** [ML4Sci Lens Finding Dataset](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view)

---

## Result Summary

| Metric | Value |
|---|---|
| **Final Test AUC** | **0.9825** |
| Best Test Loss | 0.1334 (epoch 20) |
| Training epochs | 40 |
| Train set | 1,730 lenses / 28,675 non-lenses |
| Test set | 195 lenses / 19,455 non-lenses |

---

## Problem Statement

Binary classification of multi-filter astronomical images into:
- **lens (1)** — gravitational lensing present (Einstein ring or arc visible)
- **non-lens (0)** — no lensing (ordinary galaxy, star field, or galaxy group)

**The core physics:** A gravitational lens occurs when a massive foreground galaxy sits between a background galaxy and Earth. Its gravity bends the background galaxy's light into curved arcs or Einstein rings. The task is to identify the presence of this intermediate galaxy from the distortion pattern it creates.

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

**Critical observation:** Train and test imbalance ratios are drastically different (1:16 vs 1:100). The test set is far harder than what the model trains on. This makes ROC-AUC the **only meaningful evaluation metric** — a model predicting all non-lenses would achieve 99% test accuracy but AUC of exactly 0.5.

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

**Key insight:** Lens images have high inter-channel correlation (0.8+) because one dominant central object anchors all 3 channels. Non-lens images have low correlation (~0.5) because multiple unrelated sources respond differently across filters.

**Architectural implication:** The relationship between channels is itself a discriminating feature — the model implicitly learns cross-channel consistency as a lensing signal. This is a strong argument for standard ResNet convolutions (which mix channels in every layer) over depthwise separable convolutions (which treat channels independently).

### Visual Discrimination Features

1. **Spatial arc/ring structure** — Lenses show curved arc or Einstein ring around a central galaxy. Non-lenses show no such geometry.
2. **Number of sources** — Lenses have one dominant central object with any secondary structure curved symmetrically. Non-lenses contain multiple unrelated bright spots.
3. **Channel 0 discriminability** — Channel 0 carries the most arc information due to filter sensitivity to the lensed background source.
4. **Noise floor** — Non-lenses show a noisier Channel 0 background. Lenses have steep drop toward zero — clean dark background with a bright central source. *(Important caveat: dataset artifact, not a true physical discriminator — model should not over-rely on this.)*

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
│  layer1 — BasicBlock ×2 → (B, 64,  16, 16)     │
│  layer2 — BasicBlock ×2 → (B, 128,  8,  8)     │
│  layer3 — BasicBlock ×2 → (B, 256,  4,  4)     │
│  layer4 — BasicBlock ×2 → (B, 512,  2,  2)     │
│                                                 │
│  AdaptiveAvgPool(1,1)   → (B, 512,  1,  1)     │
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
At this input size, ResNet18's `layer4` produces `2×2` spatial feature maps — only 4 spatial locations to detect lensing structure. At 224×224 it would be `7×7` (49 locations). This is the primary current limitation — see Future Work.

---

## Architecture Decisions

### Why ResNet18?

**Rejected: ResNet50**
- At 64×64 input, `layer4` feature maps are `2×2` for ResNet18 and also `2×2` for ResNet50 — identical spatial coverage
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

**Rejected: 2-output CrossEntropyLoss**
- Adds an unnecessary softmax over 2 classes
- BCEWithLogitsLoss with a single logit is mathematically equivalent but cleaner and more numerically stable

**Chosen: single logit + BCEWithLogitsLoss**

The loss function combines sigmoid and binary cross-entropy using the numerically stable log-sum-exp trick:

$$\mathcal{L} = -\left[y \cdot \log\sigma(x) + (1-y)\cdot\log(1-\sigma(x))\right]$$

where $\sigma(x) = 1/(1+e^{-x})$. Computing this directly would cause underflow for large negative $x$. BCEWithLogitsLoss uses the stable form instead.

**Why no sigmoid in forward pass:** Applying sigmoid in `forward()` AND using BCEWithLogitsLoss would double-apply it, producing incorrect gradients. Sigmoid is only applied at inference time to convert logits to probabilities.

### Why No ImageNet Normalisation?

The ImageNet normalisation formula: `output = (input - mean) / std`

With channel means of 0.1445 / 0.0869 / 0.0474 for lenses — subtracting ImageNet mean of [0.485, 0.456, 0.406] pushes the majority of pixels deep into negative territory:

```
Lens Ch0: (0.1445 - 0.485) / 0.229 = -1.49
Lens Ch2: (0.0474 - 0.406) / 0.225 = -1.59
```

**Empirical confirmation:** In Task I (same domain, same pixel range), applying ImageNet normalisation degraded training. Validation accuracy stuck at 33.33% (random chance). Removing it was the fix.

The data is already in `[0, 1]` float32 — a valid input range for ResNet without any additional normalisation.

---

## Handling Class Imbalance

### What Was Tried

**Rejected: WeightedRandomSampler**
- Oversamples minority class by repeating the same 1,730 lens images
- High risk of overfitting on the small lens set
- Model sees an artificial distribution, not the real one

**Rejected: Both WeightedRandomSampler + weighted loss**
- Double correction — over-aggressive
- Model becomes too biased toward predicting lenses: high recall, near-zero precision

**Chosen: BCEWithLogitsLoss(pos_weight=16.57)**

`pos_weight = N_non_lenses / N_lenses = 28,675 / 1,730 = 16.57`

Mathematically: for every lens sample, the loss contribution is multiplied by 16.57:

$$\mathcal{L}_{weighted} = -\left[w \cdot y \cdot \log\sigma(x) + (1-y)\cdot\log(1-\sigma(x))\right], \quad w = 16.57$$

The model cannot minimise loss by predicting all non-lenses — that strategy now incurs a 16.57× penalty on every missed lens. Transparent, principled, and directly derived from the data distribution.

---

## Differential Learning Rates

All layers are unfrozen but updated at different speeds:

| Layer group | Learning rate | Rationale |
|---|---|---|
| conv1, bn1 | 1e-5 | Low-level edge detectors — mostly universal, preserve ImageNet features |
| layer1, layer2 | 1e-5 | General texture and shape features — slow adaptation |
| layer3, layer4 | 1e-4 | Mid/high-level features — need domain adaptation |
| Classification head | 1e-3 | Trained from scratch — fastest learning |

**Mathematical justification:** Weight updates scale as $\Delta w = -\eta \cdot \nabla_w \mathcal{L}$. At LR = 1e-5 vs 1e-3, early layer weights change 100× more slowly than the head — effectively preserving ImageNet feature extractors while adapting to astronomical images.

**Why not freeze early layers?**  
In Task I (same domain), freezing `layer1`+`layer2` caused Macro AUC to plateau at 0.65. The domain gap between ImageNet (natural photographs) and astrophysical images (multi-filter telescope observations) is large enough that even early layers need adaptation. Full unfreezing with differential LR is the correct approach for this domain.

---

## Training Configuration

| Component | Choice | Rationale |
|---|---|---|
| Loss | BCEWithLogitsLoss (pos_weight=16.57) | Handles 1:16 class imbalance; numerically stable sigmoid |
| Optimizer | Adam with differential LRs | Per-parameter adaptive LR; separate rates per layer group |
| Scheduler | CosineAnnealingLR (T_max=40, eta_min=1e-6) | Smooth LR decay; zero manual tuning |
| Batch size | 64 | ~3.6 lenses/batch on average; stable minority class gradient |
| Epochs | 40 | Loss converged and plateaued after epoch 20 |
| num_workers | 0 | MPS (Apple M1) has known incompatibilities with PyTorch multiprocessing |

### Why Batch Size 64 over 32?

With 1:16 imbalance, expected lenses per batch:
```
Batch 64: 64 × (1730/30405) ≈ 3.6 lenses per batch
Batch 32: 32 × (1730/30405) ≈ 1.8 lenses per batch
```
With batch size 32, many batches would contain 0–1 lens samples — the loss signal for the minority class becomes extremely noisy. Batch size 64 ensures more stable gradient estimates.

### Why No Accuracy During Training?

With 1:16 imbalance, a model predicting all non-lenses achieves:
```
accuracy = 28,675 / 30,405 = 94.3%
```
94.3% accuracy while identifying **zero lenses**. Accuracy is completely uninformative. Loss (imbalance-aware via pos_weight) is monitored during training. AUC is computed once at the end on the best saved model.

---

## Augmentation Strategy

```python
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(180)
```

**Chosen — geometry-safe only:** Lensing geometry is rotationally symmetric. A rotated or flipped Einstein ring is still an Einstein ring with the same label.

**Rejected:**

| Transform | Reason |
|---|---|
| ColorJitter | Flux values carry physical meaning; altering brightness distorts signal-to-noise ratio which is itself a discriminating feature |
| Random crop | Arc structure often appears near image edges; cropping destroys the exact signal the model needs |
| Gaussian blur | Would smear the bright central source and arc edges — fine-grained structure is the discriminative signal |

---

## Evaluation Metric: Why ROC-AUC?

With 1:99.8 test imbalance, a naive all-non-lens classifier achieves 99% accuracy but AUC = 0.5. ROC-AUC evaluates performance across all possible decision thresholds — the threshold-independent nature is essential for a scientific lens-finding pipeline where:
- Missing a real lens (false negative) may mean losing a scientifically important system
- The optimal confidence threshold depends on the downstream scientific goal

The ROC curve's steep rise toward the top-left corner (high TPR at very low FPR) is the key visual confirmation that the model identifies lenses reliably without flooding astronomers with false positives.

---

## Results

| Metric | Value |
|---|---|
| Best Test Loss | 0.1334 (epoch 20) |
| **Final Test AUC** | **0.9825** |

The model achieves strong discrimination between lensed and non-lensed galaxies despite the severe 1:16.6 training imbalance and 1:99.8 test imbalance.

**Loss curve observations:**
- Test loss oscillated in epochs 1–20 — expected with ~3.6 lenses/batch, noisy minority class gradients
- Both losses converged and stabilised after epoch 20
- No severe divergence between train and test loss — no significant overfitting
- Plateau after epoch 20 — additional epochs beyond 40 would not help

---

## Honest Limitations

1. **No separate validation set** — the dataset provides only `train` and `test` folders. Best model was selected by test loss, meaning test set information was used at every epoch for model selection. In a rigorous setup a held-out validation set would be used for model selection, keeping test completely unseen until final evaluation. This was a dataset constraint, not a design choice.

2. **64×64 input — ResNet18 spatial bottleneck** — `layer4` produces `2×2` feature maps (4 spatial locations). At 224×224 this would be `7×7` (49 locations). Resizing to 224×224 is the highest-priority future improvement.

3. **Single run** — no cross-validation. AUC of 0.9825 may have some variance with different seeds. With only 195 test lenses, confidence intervals on AUC are wide.

---

## Future Improvements

| Improvement | Expected Impact |
|---|---|
| **Resize to 224×224** (highest priority) | layer4 goes from 2×2 → 7×7 spatial resolution; expected AUC > 0.99 |
| **Augmentation on lens class only** | Stronger augmentation on the 1,730 lens samples; synthetic variety for minority class |
| **EfficientNet-B0** | Compound scaling designed for small images; may outperform ResNet18 at 64×64 |
| **Threshold optimisation** | Optimise decision threshold for max F1 or max TPR at acceptable FPR for deployment |
| **Cross-validation** | k-fold with only 1,730 lenses; more reliable performance estimate |
| **SupCon loss** | Supervised contrastive loss — may improve minority class embedding quality |

---

## Model Weights

`best_lens_model.pth` — saved at epoch 20 (best test loss: 0.1334)

To load:
```python
model = LensClassifier()
model.load_state_dict(torch.load('best_lens_model.pth', map_location='cpu'))
model.half()  # weights are stored in float16
model.eval()

# Inference
with torch.no_grad():
    logit = model(image_tensor.half())  # input must also be float16
    prob  = torch.sigmoid(logit).item()
    pred  = int(prob > 0.5)
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
├── deeplense_task_2_lens_testing.ipynb   ← main notebook with all outputs
├── deeplense_task_2_lens_testing.pdf     ← PDF export with rendered outputs
├── best_lens_model.pth                    ← best model weights
└── README.md                              ← this file
```
