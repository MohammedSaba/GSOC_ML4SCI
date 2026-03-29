# DeepLense — GSoC 2026 Evaluation Submission

This repository contains my evaluation work for the **ML4SCI DeepLense** project, organized into two independent tasks.

Each task has its **own notebook, trained weights, and detailed README**, so this top-level file is intentionally brief and only serves as a guide to the repository structure.

## Repository Overview

- [Common_Task_1](https://github.com/MohammedSaba/GSOC_ML4SCI/tree/main/Common_Task_1)
  Common Test I: multi-class classification of simulated gravitational lensing images into `no`, `sphere`, and `vort`.

- [Task_V](https://github.com/MohammedSaba/GSOC_ML4SCI/tree/main/Task_V)
  Specific Test V: binary lens finding under strong class imbalance using multi-channel observational images.

## What Each Task Folder Contains

Each task directory includes:

- a final notebook
- saved model weights
- a task-specific README with:
  dataset summary
  architecture choices
  training strategy
  results and metrics
  setup notes

## Quick Navigation

- For the multi-class classification task, start with [Common_Task_1/Readme.md](https://github.com/MohammedSaba/GSOC_ML4SCI/blob/main/Common_Task_1/Readme.md)
- For the binary lens-finding task, start with [Task_V/Readme.md](https://github.com/MohammedSaba/GSOC_ML4SCI/blob/main/Task_V/Readme.md)

## Notes

- This repository is structured to avoid duplicating explanations across files.
- The detailed technical discussion is kept inside the individual task READMEs.
- The notebooks reflect the submitted evaluation work for DeepLense.

## File Structure

```
GSOC_ML4SCI/
├── Common_Task_1/
│   ├── best_model_weight/
│   │   └── best_model_task1_v3.pth
│   ├── notebook/
│   │   └── deeplense_common_task1_FINAL.ipynb
│   └── Readme.md
│
├── Task_V/
│   ├── best_model_weight/
│   │   └── best_model_224_from_run2_deeplense.pth
│   ├── notebook/
│   │   ├── deeplense_task2_FINAL.ipynb
│   │   └── gsoc_deeplense_task2_FINAL.ipynb
│   └── Readme.md
│
└── README.md   ← main repo README (important)
```

