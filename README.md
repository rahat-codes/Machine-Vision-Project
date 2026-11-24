# Image Classification with Deep CNNs in Keras

**A VGG-style plain CNN (Sequential API) versus a residual CNN (Functional API), trained from scratch on CIFAR-10.**

Machine Vision group project · Group NN *(replace)* · Members: Member 1, Member 2, Member 3 *(replace)*

---

## Overview

This project builds, trains and compares two deep convolutional neural networks on the CIFAR-10 dataset (60,000 colour images of 32×32 pixels in 10 classes) using Python and Keras 3 on TensorFlow.

| | Model A | Model B |
|---|---|---|
| Style | VGG-style plain CNN | Residual CNN with skip connections |
| Keras API | `Sequential` | Functional API (`Add()` merges) |
| Convolutional layers | 6 (32, 32, 64, 64, 128, 128 kernels) | 10 (1 stem + 6 main-path + 3 shortcut) |
| Downsampling | 2×2 max pooling | Stride-2 convolutions |
| Before the classifier | Flatten (2,048 values) | Global average pooling (256 values) |
| Classifier head | Dense 256 → Dense 10 | Dense 128 → Dense 10 |
| Parameters (total / trainable) | 815,914 / 815,018 | 1,246,154 / 1,243,402 |

Both models use the same stratified split, the same augmentation, the same training recipe and the same random seed (42), so any difference in results comes from the architecture and not from the set-up. No pre-trained weights are used.

## Results

Measured on the held-out 10,000-image test set, evaluated once per model from the saved `.keras` files.

| Metric | Model A (VGG-style) | Model B (Residual) |
|---|---|---|
| **Test accuracy** | **84.25%** | **87.64%** |
| Test loss | 0.4831 | 0.3865 |
| Best validation accuracy (epoch) | 85.74% (43) | 88.04% (53) |
| Epochs run (maximum 60) | 60 | 58 |
| Time per epoch | 51.7 s | 179.0 s |
| Target from the brief | about 85% | about 88–90% |
| Gap to target | −0.75 points | −0.36 points |

- Model B is **3.39 accuracy points** better than Model A, at about 1.5× the parameters and about 3.5× the time per epoch on our hardware (times depend on the machine; update this line with your GPU).
- The most-confused class pairs (both directions summed):
  - Model A: cat ↔ dog (219 errors), cat ↔ frog (111), bird ↔ frog (103)
  - Model B: cat ↔ dog (179 errors), bird ↔ deer (76), automobile ↔ truck (70)
- Both models miss their targets by less than one point; the report discusses the likely reasons (single seed, no hyperparameter search, small 32×32 images).

> Numbers above come from one run per model. If you retrain, update this section from the comparison table in the notebook (section 3.5).

## Repository contents

| File | Description |
|---|---|
| `Machine_Vision_CIFAR10_CNN_Project.ipynb` | The full project notebook, organised by weekly checkpoint |
| `model_a.keras`, `model_b.keras` | Best saved models (by validation accuracy) |
| `history_model_a.json`, `history_model_b.json` | Per-epoch training and validation metrics and epoch times |
| `test_results.json`, `probs_model_a.npy`, `probs_model_b.npy` | Cached one-time test evaluation and class probabilities |
| `model_b_architecture.png` | `plot_model` diagram of Model B (needs Graphviz, see below) |
| `demo_images/` | New photos (not from CIFAR-10) for the live prediction demo |
| `README.md` | This file |

## Setup

**Requirements:** Python 3.10 or later, TensorFlow 2.16+ with Keras 3, and a GPU runtime (Google Colab's free GPU works; each model takes roughly 20–40 minutes there).

```bash
pip install "tensorflow>=2.16" numpy pandas matplotlib scikit-learn pillow pydot jupyter
```

`keras.utils.plot_model` also needs the **Graphviz** program on your PATH (preinstalled on Colab; on Windows install it from graphviz.org or with `winget install Graphviz`, then restart the terminal and the kernel).

## How to run

1. Open `Machine_Vision_CIFAR10_CNN_Project.ipynb` in Jupyter or upload it to Google Colab and select a GPU runtime.
2. Run all cells from top to bottom. CIFAR-10 downloads automatically through `keras.datasets.cifar10`.
3. The switches in cell 1.1 control long-running work:

| Switch | Default | Meaning |
|---|---|---|
| `QUICK_TEST` | `False` | `True` runs a 2-epoch dry run on 2,048 images and writes to `quick_artifacts/` |
| `FORCE_RETRAIN` | `False` | `False` reuses saved models and histories, so Restart and Run All is fast |
| `RUN_EXTENSION` | `False` | `True` runs the optional CIFAR-100 transfer-learning bonus |

4. For the live demo, put your own photos in `demo_images/` (or upload them when the demo cell asks in Colab), then run section 4.2.

## Method in brief

**Data.** The official 50,000 training images are split into 45,000 training and 5,000 validation images (500 per class) with a stratified split and seed 42. The official 10,000 test images are used once for the final evaluation.

**Preprocessing and augmentation.** Pixels are cast to `float32` and divided by 255. Labels stay integers 0–9 (sparse categorical cross-entropy). Augmentation runs inside each model as Keras layers, on training data only: `RandomFlip("horizontal")`, `RandomTranslation(0.1, 0.1)`, `RandomRotation(0.05)`.

**Training recipe (identical for both models).**

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3 |
| Loss | Sparse categorical cross-entropy |
| Batch size / max epochs | 64 / 60 |
| Early stopping | monitor `val_loss`, patience 10, restore best weights |
| LR schedule | `ReduceLROnPlateau`: factor 0.5, patience 4, minimum 1e-5 |
| Checkpoint | best `val_accuracy`, saved in `.keras` format |
| Seed | 42 |

**Model B residual block.** Main path: Conv 3×3 (stride *s*) → BN → ReLU → Conv 3×3 → BN. Shortcut: Conv 1×1 (same stride *s*) → BN. The two paths are merged with `Add()` and followed by ReLU. The shortcut uses the same stride so both branches have identical shapes at the `Add()`.

## Notebook structure (weekly checkpoints)

| Part | Checkpoint weight | Content |
|---|---|---|
| Week 1 | 15% | Environment and seed, stratified split, sample grid, preprocessing, augmentation demo, Model A, hand calculation of shapes and parameters |
| Week 2 | 20% | Training recipe, Model A training and diagnosis, Model B (Functional API), shape checks at every `Add()` |
| Week 3 | 25% | Model B training, one-time test evaluation, per-class metrics, confusion matrices, comparison table, error analysis |
| Week 4 | 40% | Reproducibility check, live prediction demo, report and presentation plan, optional CIFAR-100 extension |
| Bonus | +5% | Transfer learning from CIFAR-10 to CIFAR-100 (Model 2 with transfer versus Model 3 from scratch) |

## Reproducibility and rules followed

- One fixed seed (42), set again before each model is built.
- Identical split, augmentation and training settings for both models.
- Tuning decisions (early stopping, learning-rate schedule, checkpoint) use the validation set only.
- The test set is evaluated once per model; the result is cached in `test_results.json`.
- The saved models reload and reproduce the reported test accuracy (checked in section 4.1).
- GPU kernels can add tiny run-to-run differences, so compare accuracies rather than exact bit patterns.

## Limitations

- One training run per model (single seed), so differences of a few tenths of a point are not conclusive.
- No hyperparameter search; the recipe is fixed by the brief.
- 32×32 images limit the detail available, especially for similar animals (cat, dog, frog, bird, deer).
- The demo classifies photos into only ten classes, so objects outside those classes still get a (meaningless) answer.

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10 dataset).
- Keras documentation: *The Sequential model* and *The Functional API*.
- Add any tutorial or documentation snippet your group adapted.

## Authors

| Member | Role / contribution |
|---|---|
| Member 1 | |
| Member 2 | |
| Member 3 | |