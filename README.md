# Face (vs No-Face) Classification – Training, Detection and Model Zoo

This project trains binary classifiers (face vs no-face), detects multiple faces in full images, and compares them using different architectures, including pretrained networks with fine-tuning.

## Project Structure

```
.
├── artifacts/
│   ├── <model>/
│   │   ├── best_model.pt        # best checkpoint for that model name
│   │   └── history.json         # loss/acc history
│   ├── detections_*.png         # annotated detection images
│   └── compare_*.txt            # comparison outputs
├── load_data.py                 # dataloaders, preprocessing and augmentation
├── models.py                    # all architectures + model registry
├── net.py                       # original (legacy) baseline CNN
├── train.py                     # training/evaluation script
├── detect_and_classify.py       # detector+classifier for full images
├── compare_models.py            # compare every model on a single image
├── train_images/                # training dataset (ImageFolder)
│   ├── 0/ ... (noface)
│   └── 1/ ... (face)
└── test_images/                 # test dataset (ImageFolder)
    ├── 0/ ... (noface)
    └── 1/ ... (face)
```

Data format (ImageFolder):
- `train_images/0` = no-face, `train_images/1` = face
- `test_images/0` = no-face, `test_images/1` = face

## Requirements

- Python 3.9+
- PyTorch, TorchVision
- OpenCV
- scikit-learn (metrics only)

Installation (pip):
```bash
pip install torch torchvision opencv-python scikit-learn
```

## Available Models (MODEL_REGISTRY)

Models are referenced by name with `--model <name>`.

Small models (36x36 grayscale input):
- tiny: minimal CNN (8→16 channels, small FC). Very fast, less capacity.
- small: small CNN (12→24 channels). Balance between speed and capacity.
- baseline: original base CNN (16→32, 2 FC). Reference point.
- bn: CNN with BatchNorm. Typically stabilizes and speeds up training.
- threeconv: CNN with 3 conv blocks + two poolings. Higher capacity.
- residual: CNN with simple residual blocks. Better gradient flow.
- improved: enhanced CNN (BatchNorm everywhere, 3 blocks, Dropout2D/1D). Generally more robust.
- attention: CNN with channel attention (SE-like). Focuses on relevant features.

Pretrained models (224x224 RGB, ImageNet normalization):
- resnet18: ImageNet-pretrained ResNet18 + fine-tuning. Strong accuracy with moderate cost.
- mobilenetv2: Pretrained MobileNetV2. Very efficient on CPU/edge.
- efficientnet: Pretrained EfficientNet-B0. Excellent accuracy/efficiency trade-off.

Key family differences:
- Input size: 36x36 (small CNNs) vs 224x224 (pretrained).
- Preprocessing: Grayscale + normalization (small) vs RGB + ImageNet normalization (pretrained).
- Capacity/regularization: `improved`/`attention` use BatchNorm + Dropout; `residual` uses shortcuts; pretrained models reuse ImageNet features with fine-tuning.

## Preprocessing and Data Augmentation

`load_data.py` automatically adapts preprocessing per model:
- Small CNNs: 36x36, grayscale, normalization mean=0.5, std=0.5.
- Pretrained: 224x224, grayscale→RGB (3 channels), ImageNet normalization.

Data augmentation (training):
- Pretrained: Resize+RandomCrop(224), HorizontalFlip, Rotation, ColorJitter.
- Small: HorizontalFlip, light Rotation.

## Training

Train a model and store the best checkpoint and history in `artifacts/<model>/`:
```bash
# Small CNNs
python3 train.py --model baseline --epochs 10
python3 train.py --model improved --epochs 10
python3 train.py --model attention --epochs 10

# Pretrained (often need smaller batch sizes)
python3 train.py --model resnet18 --epochs 10 --batch-size 32
python3 train.py --model mobilenetv2 --epochs 10 --batch-size 32
python3 train.py --model efficientnet --epochs 10 --batch-size 32
```
Useful parameters:
- `--epochs`, `--batch-size`, `--lr`, `--num-workers`

During training, each epoch prints train/valid loss & accuracy, and the final best checkpoint is evaluated on the test set.

## Detection and Classification on Full Images

Detect faces (Haar cascade), crop, preprocess, and classify each face with the selected model. Produces an annotated output image.
```bash
python3 detect_and_classify.py /path/to/image.jpg --model improved --show
# Loads artifacts/<model>/best_model.pt by default
# Haar detector tuning
python3 detect_and_classify.py /path/image.jpg --model resnet18 \
  --scale-factor 1.1 --min-neighbors 5 --threshold 0.5
```
Output:
- Annotated image at `artifacts/detections_<name>.png` (or `--save` path).
- Log containing bbox, label, and probability per detection.

### Run Detection with All Models

Process an image with every trained model and produce an annotated image per model:

```bash
python3 detect_all_models.py /path/image.jpg --skip-missing
```

- Saves outputs at `artifacts/detections/<image_name>/<model>.png`.
- Use `--models baseline,improved,resnet18` to limit the list.
- Adjust `--threshold`, `--scale-factor`, `--min-neighbors` like `detect_and_classify.py`.
- With `--skip-missing`, models without checkpoints are skipped instead of failing.

## Quickly Compare Models on a Single Image

Compute the “face” probability of every registered model on a single image (no detection, full image). Useful for a sanity check and quick ranking.
```bash
python3 compare_models.py /path/to/image.jpg --save
```
- Prints a console table.
- `--save` writes `artifacts/compare_<name>.txt`.

## Comparative Model Evaluation (metrics + plots)

Evaluate all (or a subset of) models on the test set to get accuracy, precision, recall, and F1 plus comparison plots:

```bash
python3 evaluate_models.py
# or limit it
python3 evaluate_models.py --models baseline,improved,resnet18
```

Outputs in `artifacts/evaluation/`:
- `summary.csv`: table with metrics per model.
- `accuracy.png`, `f1.png`, `precision.png`, `recall.png`: bar plots.
- `cm_<model>.png`: confusion matrices for the top-3 models by accuracy.

### Confidence Intervals via Bootstrapping

Estimate uncertainty with bootstrapping (resampling the test set with replacement):

```bash
# 1000 resamples and 95% CI
python3 evaluate_models.py --bootstrap 1000 --ci 95
```

Extra files:
- `bootstrap_summary.csv`: mean, std, CI [low, high] per model/metric.
- `bootstrap_<model>.csv`: bootstrap distributions per metric.
- `<metric>_violin.png`: violin plots comparing bootstrap distributions.

## Quick Start Guide

1) Install dependencies
```bash
# Headless environments (Jupyter/servers)
pip install torch torchvision opencv-python-headless scikit-learn
# Local environments with GUI
pip install torch torchvision opencv-python scikit-learn
```

2) Prepare data
- Create `train_images/0`, `train_images/1`, `test_images/0`, `test_images/1`.
- Place “no-face” images inside `0/` and “face” images inside `1/`.

3) Train
```bash
# Every model (recommended if you want comparisons)
python3 train_all.py

# Single model (faster)
python3 train.py --model improved --epochs 10
```
Outputs: `artifacts/<model>/best_model.pt` and `history.json`.

4) Test detection on an image
```bash
python3 detect_and_classify.py /path/image.jpg --model improved --show
# On servers/Jupyter omit --show and check artifacts/detections_<name>.png
```

5) Run every model at once
```bash
python3 detect_all_models.py /path/image.jpg --skip-missing
```
Outputs: one annotated image per model in `artifacts/detections/<image_name>/`.

6) Compare “face probability” without detection (full image)
```bash
python3 compare_models.py /path/image.jpg --save
```

## What does the “percentage” in the images mean?
- It is the probability assigned to class “face” for each detected crop.
- If the value is greater than or equal to the threshold (`--threshold`, default 0.5) the label is “face”; otherwise “noface”.
- Increase the threshold (0.6–0.7) for fewer false positives at the cost of missing some true faces (lower recall).

## Model Differences (High Level)

- Small models (36x36 grayscale input):
  - `tiny` and `small`: extremely fast and light; lower capacity.
  - `baseline`: original reference, decent performance.
  - `bn`: adds BatchNorm; training is more stable.
  - `threeconv`: slightly deeper; usually better.
  - `residual`: residual shortcuts; improved gradient flow.
  - `improved`: best practices (BatchNorm+Dropout2D/1D); robust.
  - `attention`: channel attention (SE-style); focuses on useful cues.

- Pretrained (224x224 RGB, ImageNet normalization):
  - `resnet18`: strong accuracy/time balance.
  - `mobilenetv2`: very efficient for CPU/edge.
  - `efficientnet`: high accuracy/efficiency, slightly heavier.

Rules of thumb:
- Small or diverse datasets benefit from pretrained models.
- Need speed and low resource usage? Choose `mobilenetv2` or `tiny/small`.
- Want a reliable non-pretrained option? Try `improved` or `residual`.

## How to Compare Models Fairly

- Use the same test set for every model.
- Keep the same threshold (`--threshold`) when comparing “face probability”.
- Recommended metrics: accuracy, precision, recall, F1.
- Options:
  - Use `test.py` for evaluation + confusion matrix (one model at a time).
  - Extend `compare_models.py` to iterate over `test_images/` and compute global metrics per model (suggested for reports).

Quick idea (not yet implemented):
- Script that loops through `test_images/`, runs each model, and stores accuracy/F1 per model into a CSV table.

## FAQ / Troubleshooting

- “cv2 (OpenCV) not found”
  - Install `pip install opencv-python-headless` (server/Jupyter) or `pip install opencv-python` (local with GUI).

- “Pretrained models crash with shared memory (shm) errors”
  - Train with `num_workers=0` for pretrained models (default in `train_all.py`).
  - Lower `--batch-size` if RAM/GPU is limited.

- “No image window appears”
  - Servers/Jupyter have no GUI. Skip `--show` and inspect the saved image in `artifacts/`.

- “No checkpoint found”
  - Train first: `python3 train.py --model baseline` (or any other).

- “How do I adjust sensitivity?”
  - Raise `--threshold` (0.6 or 0.7) to reduce false positives.
  - Tune Haar detector via `--scale-factor` and `--min-neighbors`.

## Glossary
- Detection: find where a face is (bounding box) in an image.
- Classification: decide whether a crop is “face” or “noface”.
- Pretrained: model learned generic features from millions of images (ImageNet) before fine-tuning on this task.
- Data augmentation: image transformations (rotate, crop, flip) to make the model more robust.

## Cheat Sheet Commands
```bash
# Train everything
python3 train_all.py

# Train a single model
python3 train.py --model improved --epochs 10

# Detection with one model
python3 detect_and_classify.py /path/image.jpg --model resnet18 --threshold 0.6

# Detection with every model
python3 detect_all_models.py /path/image.jpg --skip-missing

# Compare probabilities without detection
python3 compare_models.py /path/image.jpg --save
```

## Suggested Extended Evaluation

- Use `test.py` for metrics and confusion matrix on the test set (one model at a time).
- Extend `compare_models.py` to loop over `test_images/` and generate an aggregated table with accuracy/precision/recall/F1 per model.

## Practical Tips

- Class balance: if the dataset is imbalanced you can:
  - Add a `WeightedRandomSampler` in `load_data.py`.
  - Use class weights in `CrossEntropyLoss`.
- Decision threshold: tweak `--threshold` to prioritize recall vs precision.
- Early stopping & schedulers: consider schedulers (Cosine/OneCycle) and early stopping for pretrained runs.
- Resources: pretrained models consume more memory; reduce `--batch-size` if you hit OOM.

## Dependencies and Notes

- Install the main dependencies:
```bash
pip install torch torchvision opencv-python scikit-learn
```
- scikit-learn: install `scikit-learn` and import via `from sklearn ...`.
- OpenCV Haar cascade is bundled in `cv2.data.haarcascades`.

## FAQ

- The image will not open in `detect_and_classify.py`:
  - Double-check the path. Quote it if it contains spaces.
  - Example: `python3 detect_and_classify.py "~/Pictures/photo.jpg" --model baseline`
- Missing checkpoint:
  - Train first: `python3 train.py --model baseline`
- Can I use video/webcam?
  - The pipeline is built for images. You can extend it by capturing frames and applying the same process (happy to add it if needed).

---
