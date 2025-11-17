# Face (vs No-Face) Classification Project

End-to-end pipeline for training, evaluating, and deploying binary face/no-face classifiers. The codebase supports multiple CNN variants (including fine-tuned pretrained models), test-set evaluation, and qualitative detection overlays on full-scene images.

---

## 1. Environment Setup

```bash
python -m venv .venv          # optional but recommended
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: PyTorch, TorchVision, OpenCV, scikit-learn, matplotlib, seaborn, pandas.

---

## 2. Project Layout

```
.
├── artifacts/
│   ├── <model>/
│   │   ├── best_model.pt       # best checkpoint for that model
│   │   └── history.json        # training history
│   ├── detections/             # annotated detection outputs
│   └── evaluation/             # metrics, plots, bootstrap stats
├── train_images/               # ImageFolder dataset (0 = no-face, 1 = face)
├── test_images/                # ImageFolder dataset (same structure)
├── load_data.py                # preprocessing + data augmentation
├── models.py                   # MODEL_REGISTRY with all architectures
├── train.py / train_all.py     # training entry points
├── test.py                     # evaluates a single model on test set
├── detect_and_classify.py      # detection + classification for one model
├── detect_all_models.py        # detection overlays for every model
├── compare_models.py           # log face probabilities for all models
├── evaluate_models.py          # metrics, plots, bootstrap confidence
└── report.ipynb                # notebook for final presentation
```

---

## 3. Dataset Preparation

1. Place cropped or full-scene images into `train_images/` and `test_images/` following `ImageFolder` structure:
   - `train_images/0` → “no-face” examples
   - `train_images/1` → “face” examples
   - Repeat for `test_images/`
2. The dataloaders automatically adapt preprocessing to the selected model (grayscale 36×36 vs. RGB 224×224).

---

## 4. Training Commands

- **Train every model in sequence (recommended for comparisons)**
  ```bash
  python train_all.py
  ```

- **Train a specific model**
  ```bash
  python train.py --model improved --epochs 10
  python train.py --model resnet18 --epochs 10 --batch-size 32  # pretrained needs smaller batch
  ```

Common flags: `--epochs`, `--batch-size`, `--lr`, `--num-workers`.

Checkpoints and histories are written to `artifacts/<model>/`.

---

## 5. Model Zoo (MODEL_REGISTRY)

| Model           | Input        | Description                                                                 |
|-----------------|--------------|-----------------------------------------------------------------------------|
| `tiny`, `small` | 36×36 gray   | Minimal CNNs; extremely fast, lower capacity                                |
| `baseline`      | 36×36 gray   | Original two-conv network                                                   |
| `bn`            | 36×36 gray   | Adds BatchNorm for improved stability                                       |
| `threeconv`     | 36×36 gray   | Three convolutional blocks; higher capacity                                 |
| `residual`      | 36×36 gray   | Residual skip connections for better gradient flow                          |
| `improved`      | 36×36 gray   | Deeper CNN with BatchNorm everywhere and Dropout2D/Dropout1D               |
| `attention`     | 36×36 gray   | Channel attention (SE-style) on top of the improved CNN                     |
| `resnet18`      | 224×224 RGB  | Fine-tuned ResNet18 (ImageNet pretrained)                                   |
| `mobilenetv2`   | 224×224 RGB  | Fine-tuned MobileNetV2 (efficient, good for edge devices)                   |
| `efficientnet`  | 224×224 RGB  | Fine-tuned EfficientNet-B0 (strong accuracy vs. compute trade-off)          |

All models are instantiated through `MODEL_REGISTRY`; training/eval scripts pick the right preprocessing automatically.

---

## 6. Detection & Qualitative Review

- **Single model detection on a full image**
  ```bash
  python detect_and_classify.py path/to/image.jpg --model improved --threshold 0.6 --show
  ```
  (Omit `--show` on headless servers. Output saved to `artifacts/detections_<image>.png`.)

- **Run detection with all trained models**
  ```bash
  python detect_all_models.py path/to/image.jpg --skip-missing
  ```
  Each model’s overlay is stored under `artifacts/detections/<image>/<model>.png`.

- **Compare face probability (no detection, entire image crop)**
  ```bash
  python compare_models.py path/to/image.jpg --save
  ```
  Useful for sanity checks and quick rankings.

---

## 7. Evaluation & Reporting

- **Test-set metrics, confusion matrices, bootstrap CIs**
  ```bash
  python evaluate_models.py
  python evaluate_models.py --bootstrap 1000 --ci 95
  ```
  Outputs (`artifacts/evaluation/`):
  - `summary.csv` with accuracy/precision/recall/F1
  - `accuracy.png`, `f1.png`, `precision.png`, `recall.png`
  - `cm_<model>.png` (confusion matrices for top-3 models)
  - `bootstrap_summary.csv`, `bootstrap_<model>.csv`, and violin plots (`*_violin.png`) if bootstrapping is enabled

- **Evaluate a single trained model with confusion matrix + sample predictions**
  ```bash
  python test.py --model improved
  ```

- **Notebook report**
  Open `report.ipynb` to combine quantitative tables, qualitative detections, and narrative summaries for your submission.

---

## 8. Practical Tips

- Increase `--threshold` (e.g. 0.6–0.7) to reduce false positives; adjust `--scale-factor` and `--min-neighbors` for detector sensitivity.
- Pretrained models should use `--pretrained-batch-size 32` (or lower) and `num_workers=0` on limited shared memory environments (already handled in `train_all.py` and `evaluate_models.py`).
- If `cv2` is missing, install `opencv-python-headless` on servers/Jupyter, or `opencv-python` locally.
- `artifacts/` can grow large; tweak `.gitignore` if you wish to commit selected outputs.

---

## 9. Quick Command Reference

```bash
# Train all models
python train_all.py

# Train one model
python train.py --model improved --epochs 10

# Evaluate one model on the test set
python test.py --model resnet18

# Run detection (single model / all models)
python detect_and_classify.py image.jpg --model efficientnet --threshold 0.6
python detect_all_models.py image.jpg --skip-missing

# Compare face probabilities (no detection)
python compare_models.py image.jpg --save

# Summarise metrics + bootstrap CIs
python evaluate_models.py --bootstrap 1000 --ci 95
```

This README now covers everything needed to install, train, evaluate, and inspect the results across the full model zoo. Suit up the dataset, run the scripts above, and plug the generated metrics/plots into your final report.


