# DeepLense — Common Test I: Multi-Class Gravitational Lens Classification

**GSoC 2026 Evaluation | ML4Sci**  
**Task:** Common Test I — Multi-Class Classification  
**Dataset:** [ML4Sci DeepLense dataset.zip](https://drive.google.com/)

---

## Result Summary

| Class | AUC |
|---|---|
| no substructure | 0.9939 |
| sphere (subhalo) | 0.9884 |
| vort (vortex) | 0.9968 |
| **Macro AUC** | **0.9930** |

---

## Problem Statement

Classify gravitational lensing telescope images into three physical categories:

- **no** — strong lensing with no substructure (clean Einstein ring)
- **sphere** — subhalo substructure (ring distorted by a compact dark matter clump)
- **vort** — vortex substructure (twisted/spiral arc pattern)

---

## Dataset

| Property | Value |
|---|---|
| Format | `.npy` files, shape `(1, 150, 150)`, `float64` |
| Train | 30,000 images (10,000 per class — perfectly balanced) |
| Validation | 7,500 images (2,500 per class — perfectly balanced) |
| Split |Pre-provided by dataset (30,000 train / 7,500 val — 80:20) |
| Normalisation | Min-max to `[0.0, 1.0]` — applied by ML4Sci |
| Mean pixel value | 0.062 — very dark images (mostly black sky) |

The dataset was provided with a pre-existing train/val split. We use it as-is rather than re-splitting, as the class balance is identical across both sets (10,000 / 2,500 per class respectively).

**Key observation from EDA:** All three classes share the fundamental Einstein ring shape. Differences are extremely subtle — subhalo creates a localised brightness anomaly in the ring; vortex creates a spiral twist pattern. Even human visual inspection struggles to distinguish sphere from no substructure consistently. This confirmed the model must capture local brightness anomalies, ring completeness, and symmetry, not just global shape.

---

## Architecture

### Model Architecture Diagram

```
Input: (B, 1, 150, 150)  ← single-channel grayscale lensing image
         │
         ▼
  np.repeat(×3)  ← replicate channel: (B, 3, 150, 150)
         │
         ▼
┌─────────────────────────────────────────────────┐
│              ResNet18 Backbone                  │
│                                                 │
│  conv1 (7×7, 64, stride=2)                      │
│  bn1 → ReLU → MaxPool (3×3, stride=2)           │
│       ↓ feature map: (B, 64, 38, 38)            │
│                                                 │
│  layer1 — BasicBlock ×2 → (B, 64,  38, 38)      │
│  layer2 — BasicBlock ×2 → (B, 128, 19, 19)      │
│  layer3 — BasicBlock ×2 → (B, 256, 10, 10)      │
│  layer4 — BasicBlock ×2 → (B, 512,  5,  5)      │
│                                                 │
│  AdaptiveAvgPool(1,1)   → (B, 512,  1,  1)      │
└─────────────────────────────────────────────────┘
         │
         ▼
  Flatten  → (B, 512)
         │
         ▼
┌──────────────────────────┐
│   Classification Head    │
│                          │
│  Linear(512 → 256)       │
│  ReLU                    │
│  Dropout(p=0.2)          │
│  Linear(256 → 3)         │
└──────────────────────────┘
         │
         ▼
  Output: (B, 3) logits
  → softmax → class probabilities
  → argmax  → predicted class
```

**Total parameters:** 11,308,611  
**Trainable:** 11,308,611 (100% — full fine-tuning)

---

## Architecture Decisions

### Why ResNet18?

**Rejected: ResNet50**
- 25.6M parameters vs ResNet18's 11.3M
- At 150×150 input, spatial feature maps at layer4 are `5×5` — sufficient for arc and ring detection
- ResNet50's additional depth provides no spatial resolution benefit here
- Higher overfitting risk on a 30k dataset with subtle inter-class differences
- 2× slower training with no expected accuracy gain at this image scale

**Rejected: AlexNet**
- Older architecture, no skip connections
- ResNet's residual connections handle gradient flow better during fine-tuning
- Less efficient feature extraction

**Rejected: ViT (Vision Transformer)**
- Designed for large image datasets (ImageNet-scale, 224×224+)
- Requires far more data to outperform CNNs
- Patch tokenisation at 150×150 produces coarse spatial coverage
- Overkill for structured physics images with consistent spatial patterns

**Chosen: ResNet18**
- Proven track record on DeepLense tasks (past GSoC submissions achieved AUC 0.94+)
- Skip connections preserve gradient signal through 18 layers
- Fast enough for iterative experimentation under deadline

### Why Repeat Grayscale to 3 Channels?

Images are single-channel `(1, 150, 150)`. ResNet18's first conv layer expects 3 channels. We apply `np.repeat(img, 3, axis=0)` → `(3, 150, 150)`.

This is mathematically equivalent to `R = G = B = intensity`. The pretrained conv1 filters still activate meaningfully on edges and gradients — they respond to spatial structure, not colour. The real value of pretraining is in `layer3` and `layer4` which detect complex curved structures (arcs, rings) directly relevant to lensing.

An alternative is to modify conv1 to accept 1 channel, but this would destroy the pretrained weights for that layer and require learning edge detectors from scratch on only 30k images.

### Why No ImageNet Normalisation? (Critical Bug and Fix)

**Initial approach** applied ImageNet normalisation on top of min-max normalised images:
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**What went wrong:** Our images have mean pixel value of 0.072 (mostly black sky). ImageNet normalisation subtracts 0.485 from values that are mostly near 0:

```
pixel value 0.07 → (0.07 - 0.485) / 0.229 = -1.81
```

Mean activation across the dataset became -1.67. The model was seeing inputs almost entirely in negative territory with almost no variance — no signal to learn from.

**Evidence:** Validation accuracy stuck at exactly 33.33% — mathematically equal to `1/3`, i.e. random guessing for 3 balanced classes.

**Fix:** Remove ImageNet normalisation entirely. Min-max normalisation to `[0, 1]` is sufficient:
- Min-max → ensures consistent scale across images ✅ already done by ML4Sci
- ImageNet stats → shifts distribution to match pretrained model's expected input ❌ harmful here

---

## Transfer Learning Strategy — Evolution

### Experiment 1: Partial Fine-Tuning (Rejected, Macro AUC 0.6499)

**Strategy:** Freeze `layer1` + `layer2`, unfreeze only `layer3` + `layer4`.

**Theoretical reasoning:** Early layers detect universal features (edges, gradients) that transfer well from ImageNet. Only deeper layers need domain adaptation.

**Why it failed:** The domain gap between ImageNet (natural RGB photographs) and gravitational lensing (single-channel physics simulations) is too large. Layer 3 and 4 had learned ImageNet-specific high-level features (object parts, textures, colours) that didn't transfer to ring/arc detection even after fine-tuning.

**Result:** Macro AUC 0.6499 — insufficient.

### Experiment 2: Partial Fine-Tuning + Augmentation (Macro AUC 0.6639)

Added rotation and flip augmentation. Marginal improvement. Root cause (frozen early layers) unchanged.

### Final: Full Fine-Tuning (Macro AUC 0.9930 ✅)

**Strategy:** Unfreeze all layers. Reduce learning rate from 1e-3 to 1e-4.

**Why this works:**
- Every layer adapts to the new domain — including early conv layers that need to learn grayscale edge detectors rather than RGB colour features
- Low LR (1e-4) preserves the useful pretrained weight structure while making careful domain-specific adjustments. A large LR would destroy pretrained knowledge. The update magnitude per step scales linearly with LR — at 1e-4 vs 1e-3, weights move 10× more slowly, preserving general features while adapting to lensing

**The jump from 0.65 to 0.993 AUC** from simply unfreezing all layers with a lower LR is a strong empirical demonstration that transfer learning *strategy* matters more than architecture choice for out-of-domain data.

---

## Training Configuration

| Component | Choice | Rationale |
|---|---|---|
| Loss | CrossEntropyLoss | Standard for balanced 3-class classification |
| Optimizer | Adam, LR=1e-4 | Adaptive per-parameter LR; stable with pretrained weights |
| Weight decay | 1e-4 | L2 regularisation to prevent overfitting |
| Scheduler | CosineAnnealingLR (T_max=40, eta_min=1e-7) | Smooth decay; no sharp LR drops; no manual schedule tuning |
| Early stopping | patience=7 | Saves compute; saves best model by val loss |
| Batch size | 64 | Stable gradient estimates; fits in MPS memory |
| Dropout | 0.2 | Light regularisation — augmentation handles generalisation |
| Epochs | 40 | Early stopping triggered at epoch 38 |

### Why CosineAnnealingLR?

The learning rate follows: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t / T_{max}))$

Large steps early → fast learning. Small steps late → fine adjustment without overshooting. Zero additional hyperparameters vs StepLR (which requires manual drop schedule).

---

## Augmentation Strategy

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(degrees=180)
```

**Why these:** Gravitational lensing has rotational symmetry — a rotated Einstein ring is physically identical to the original. Flips and rotations preserve the class label perfectly and double/triple the effective dataset size.

**Explicitly rejected:**

| Rejected | Reason |
|---|---|
| ColorJitter (brightness/contrast) | Bright point sources ARE the discriminative signal for sphere class. Altering brightness could make a sphere image indistinguishable from no substructure |
| Random erasing | Could erase the subhalo point source — the exact feature that defines sphere class |
| Random crop | Could cut off part of the Einstein ring, losing the structural information the model needs |
| Gaussian blur | Would smear point sources and ring edges, destroying fine-grained class differences |

---

## Evaluation Metric: Why ROC-AUC over Accuracy?

Accuracy evaluates the model at a single fixed threshold (argmax = 0.5). ROC-AUC evaluates across **all possible thresholds simultaneously**.

For scientific use cases, physicists don't use a fixed threshold — they tune confidence cutoffs based on their specific scientific goal (e.g., prioritise recall over precision for rare substructure detection). A model with high AUC is trustworthy regardless of what threshold is chosen downstream. Additionally, argmax implicitly assumes 0.5 is optimal, which is rarely true in practice.

**Why one-vs-rest for multiclass ROC:** ROC curves are fundamentally binary. For 3 classes we decompose into 3 binary problems:
- Round 1: no vs (sphere + vort)
- Round 2: sphere vs (no + vort)  
- Round 3: vort vs (no + sphere)

Macro-averaged AUC is the simple average across all 3 — appropriate because classes are perfectly balanced (10k each).

---

## Results

| Metric | Value |
|---|---|
| Val accuracy (best checkpoint) | 95.25% (epoch 38) |
| no — AUC | 0.9939 |
| sphere — AUC | 0.9884 |
| vort — AUC | 0.9968 |
| **Macro AUC** | **0.9930** |

**Why sphere has the lowest AUC:** Consistent with the EDA finding that subhalo distortions are the subtlest class difference — even human observers struggle to distinguish sphere from no substructure consistently. The model's difficulty aligns with human intuition, which is scientifically meaningful rather than being an arbitrary model weakness.

---

## Training History (Key Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.0230 | 43.7% | 0.8093 | 62.6% |
| 10 | 0.3124 | 88.3% | 0.2741 | 89.9% |
| 20 | 0.2174 | 92.1% | 0.1803 | 93.7% |
| 30 | 0.1671 | 93.96% | 0.1419 | 94.9% |
| 38 | 0.1443 | 94.8% | 0.1339 | **95.25%** ← best |
| 40 | 0.1443 | 94.75% | 0.1348 | 95.3% |

---

## Overfitting Analysis

Early runs (before augmentation) showed severe overfitting:
- Epoch 5: Train 46% | Val 44% — gap 2%
- Epoch 20: Train 69% | Val 45% — gap 24%, val loss doubled

**Fix:** Augmentation (flip + rotation) + early stopping (patience=7) + dropout 0.4→0.2

**Why reduce dropout if overfitting?** The model was simultaneously underfitting — val accuracy plateaued at 44%. Dropout 0.4 was killing 40% of neurons, preventing learning of subtle lensing patterns. Augmentation handles generalisation; dropout provides only light regularisation (0.2).

---

## Future Improvements

| Improvement | Expected Impact |
|---|---|
| **Grad-CAM visualisation** | Confirm model attends to ring distortions and point sources rather than background artifacts — critical for scientific trustworthiness |
| **EfficientNet-B0** | 5.3M params, compound scaling, worth comparing at same image size |
| **Test Time Augmentation (TTA)** | Average predictions over multiple augmented versions of each test image — free performance boost |
| **Ensemble** | ResNet18 + EfficientNet-B0, average softmax outputs → typically +1-2% AUC |
| **Per-epoch AUC tracking** | More informative training signal than accuracy for subtle multi-class problems |

---

## Model Weights

`best_model_task1_v3.pth` — saved at epoch 38 (best val loss: 0.1339)

To load:
```python
from model import DeepLenseClassifier
model = DeepLenseClassifier(num_classes=3, dropout=0.2)
model.load_state_dict(torch.load('best_model_task1_v3.pth', map_location='cpu'))
model.eval()
```

---

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Update `DATA_DIR` in the notebook to point to your local dataset path.

---

## File Structure

```
Task1_CommonTest/
├── deeplense_task_1_multiTask.ipynb   ← main notebook with all outputs
├── deeplense_task_1_multiTask.pdf     ← PDF export with rendered outputs
├── best_model_task1_v3.pth            ← best model weights
├── roc_curves_task1.png               ← saved ROC curve plot
└── README.md                          ← this file
```
