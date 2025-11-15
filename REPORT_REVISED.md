# Face Detection and Classification System

## Table of Contents

1. Introduction
2. Data and Data Preprocessing
3. Models and Architecture
4. Project Journey
5. Model Training Explanation
6. Error Reduction with Bootstrapping
7. Model Comparison and Results
8. Conclusion

---

## Introduction

This document explains how we built, trained, and tested a system that detects whether there are one or multiple faces in an image. The project addresses two complementary tasks:

**Classification**: Identifies whether a small crop of a photo contains a face or not (binary classification: "face" vs. "no-face").

**Detection**: Scans a full image to locate faces, drawing bounding boxes around detected faces and assigning confidence scores.

We evaluated two types of models:

1. **Small custom CNNs** (9 variants): Models we designed by adding/removing layers, adjusting parameters, and modifying architectures. These operate on 36×36 grayscale inputs for fast inference.

2. **Large pretrained models** (3 variants): Fine-tuned architectures (EfficientNet-B0, MobileNetV2, ResNet18) that leverage features learned from ImageNet. These operate on 224×224 RGB inputs and generally achieve higher accuracy with less training data.

As expected, the pretrained models achieved superior performance compared to our custom CNNs, demonstrating better accuracy with less training effort. However, the custom CNNs offer advantages in terms of inference speed and memory footprint, making them suitable for resource-constrained environments.

An important component of our evaluation is **bootstrapping**, which we used to estimate confidence intervals for our metrics and assess model robustness across different data subsets.

---

## Data and Data Preprocessing

### Folder Setup

The dataset follows PyTorch's `ImageFolder` structure:

- `train_images/0/` → no-face examples
- `train_images/1/` → face examples
- `test_images/0/` → no-face examples (held-out for evaluation)
- `test_images/1/` → face examples (held-out for evaluation)

### Preprocessing for Small Custom CNNs

For our custom CNNs, images undergo the following transformations:
- **Grayscale conversion**: Single-channel input to reduce computational cost
- **Resize to 36×36 pixels**: Standardized input size matching the network architecture
- **Normalization**: Mean=0.5, std=0.5 to stabilize training and improve convergence

### Preprocessing for Pretrained Models

Pretrained models require different preprocessing to match their original training conditions:
- **Grayscale to RGB conversion**: Duplicated channels (since pretrained models expect 3-channel input)
- **Resize and crop to 224×224 pixels**: Standard ImageNet input size
- **ImageNet normalization**: Mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Data Augmentation

During training, we apply augmentation to improve generalization:
- **Pretrained models**: Random crop, horizontal flip, rotation (±10°), color jitter (brightness/contrast)
- **Custom CNNs**: Horizontal flip, light rotation (±5°)

Validation and test sets use no augmentation to ensure fair evaluation.

---

## Models and Architecture

### Tiny, Small, Baseline, and Large CNNs

These models consist of lightweight two-block convolutions with fully connected heads. They are fast and have minimal parameters, making them suitable for constrained environments. The differences between them primarily involve capacity increases:

- **Tiny**: 8→16 channels, minimal fully connected layer (32 units)
- **Small**: 12→24 channels, moderate capacity
- **Baseline**: 16→32 channels, two fully connected layers (128→64 units)
- **Large**: 32→64 channels with deeper architecture, significantly more parameters

All offer fast inference but with slightly lower accuracy compared to deeper architectures.

### BatchNorm (BN) CNN

This model adds batch normalization after convolutional layers to:
- Stabilize training by normalizing activations
- Allow higher learning rates
- Improve generalization

It maintains low runtime while typically improving accuracy over the baseline.

### ThreeConv CNN

We added a third convolutional block to acquire richer feature representations without a dramatic parameter increase. This improves:
- Optimization stability
- Representational capacity
- Accuracy

While still maintaining a lightweight footprint.

### Residual CNN

This model introduces skip connections (residual blocks) to:
- Ease gradient flow during backpropagation
- Enable training of deeper networks
- Maintain high accuracy without excessive parameter growth

Residual connections help prevent vanishing gradients and enable more stable training.

### Improved CNN

This architecture incorporates best practices:
- **Batch normalization** on every layer
- **Dropout2D** (spatial dropout) and **Dropout1D** for regularization
- **Deeper structure** with a 128-channel block
- **Three pooling stages** (36→18→9→4 spatial reduction)

This model balances accuracy and speed, serving as a robust baseline for custom architectures.

### Attention CNN

An extension of the Improved CNN that adds **channel attention mechanisms** (similar to Squeeze-and-Excitation networks). The attention blocks:
- Dynamically reweight feature maps
- Focus the network on relevant facial features
- Suppress less informative channels

This often yields better focus on discriminative facial cues.

### Pretrained Models

These models leverage features learned from ImageNet (1.2M images, 1000 classes) and are fine-tuned for our binary classification task:

- **ResNet18**: 18-layer residual network, good balance of accuracy and computational cost
- **MobileNetV2**: Depthwise separable convolutions, optimized for mobile/edge devices
- **EfficientNet-B0**: Compound scaling of depth, width, and resolution, excellent accuracy-to-efficiency ratio

All pretrained models have their final classification layers replaced with a custom head (128-unit hidden layer → 2-class output) and are fine-tuned on our dataset.

---

## Project Journey

### Initial Setup and Baseline

We began with a simple grayscale CNN (`net.py`) that classified cropped 36×36 images as "face" or "no-face". The dataset was organized via `ImageFolder`, and `train.py`/`test.py` handled training and evaluation. At this stage, the system only worked on single-face crops.

### Full-Image Detection Pipeline

We extended the workflow to detect faces in full images. The detection pipeline works as follows:

1. **Face detection**: Using OpenCV's Haar cascade detector to find candidate face regions (bounding boxes)
2. **Crop extraction**: Each detected region is cropped from the original image
3. **Preprocessing**: Each crop is preprocessed according to the selected model (36×36 grayscale or 224×224 RGB)
4. **Classification**: The model assigns a "face" or "no-face" label with a confidence score
5. **Visualization**: Results are drawn on the original image with bounding boxes and labels

When we run detection, the script saves annotated images in `artifacts/detections/`. The script `detect_and_classify.py` handles single-model inference, while `detect_all_models.py` runs every trained model and saves per-model overlays for comparison.

### Model Zoo Expansion

We replaced the legacy `net.py` with a centralized registry (`models.py`) that exposes all architectures through a `MODEL_REGISTRY`. Small CNN variants keep the 36×36 grayscale pipeline, whereas fine-tuned pretrained models operate on 224×224 RGB inputs. Training scripts (`train.py`, `train_all.py`) were parameterized to:
- Load the appropriate model architecture
- Apply model-specific preprocessing
- Save checkpoints to `artifacts/<model>/best_model.pt`

This unified workflow allows easy comparison across all models.

---

## Model Training Explanation

### Training Process Overview

During training, we show the model batches of images with their labels. The model makes predictions, and we calculate the error (loss) between predictions and true labels. The model then adjusts its parameters (weights and biases) through backpropagation to minimize this error. We repeat this process for multiple epochs until the model converges.

### Batch Size Considerations

- **Custom CNNs**: Can handle larger batches (typically 64-128) due to smaller input size and fewer parameters
- **Pretrained models**: Require smaller batches (typically 32) due to:
  - Larger input size (224×224 vs 36×36)
  - More parameters
  - Higher memory requirements

### Operation Per Epoch

1. **Data loading**: `load_data.py` builds the appropriate transforms automatically based on the selected model
2. **Forward pass**: The chosen model from `models.py` processes the batch
3. **Loss calculation**: Cross-entropy loss compares predictions vs. labels
4. **Backward pass**: Gradients are computed via backpropagation
5. **Parameter update**: Optimizer (Adam) updates weights and biases
6. **Validation**: After each epoch, we evaluate on a held-out validation set
7. **Checkpointing**: If the model achieves the best validation accuracy so far, we save a copy

### Saving Information

We save critical information to track and reproduce results:

- **Weights**: Model parameters are saved in `artifacts/<model>/best_model.pt`
- **Training history**: Per-epoch metrics (train/validation loss and accuracy) are saved in `artifacts/<model>/history.json`

Later, `evaluate_models.py` reads `best_model.pt` and creates tables and plots of accuracy, precision, recall, and F1 on the test set.

---

## Error Reduction with Bootstrapping

### What is Bootstrapping?

Bootstrapping is a statistical resampling technique used to estimate the uncertainty of our metrics. Instead of evaluating on the test set once, we:

1. **Resample with replacement**: Create multiple "bootstrap samples" by randomly sampling from the test set (same size, but some examples may appear multiple times, others not at all)
2. **Evaluate on each sample**: Run the model on each bootstrap sample and compute metrics
3. **Compute confidence intervals**: From the distribution of bootstrap metrics, we calculate confidence intervals (e.g., 95% CI) to understand the range of plausible values

### Why Bootstrapping Matters

Bootstrapping helps us:
- **Assess robustness**: Understand how stable our metrics are across different data subsets
- **Compare models fairly**: If two models have similar mean accuracy but different confidence intervals, we can see which is more reliable
- **Quantify uncertainty**: Instead of a single accuracy number, we get a range (e.g., "accuracy = 0.95 ± 0.02 with 95% confidence")

### Implementation

We implemented bootstrapping in `evaluate_models.py` with the `--bootstrap` flag. For each model, we:
- Generate N bootstrap samples (typically 1000)
- Evaluate metrics on each sample
- Compute mean, standard deviation, and confidence intervals
- Generate violin plots showing the distribution of metrics across bootstrap samples

This helps identify models that are not only accurate on average but also consistently reliable.

---

## Model Comparison and Results

### Evaluation Metrics

We used four key metrics to measure model performance:

1. **Accuracy**: Percentage of correct predictions overall
   - Formula: (True Positives + True Negatives) / Total Samples
   - Measures overall correctness but can be misleading with imbalanced datasets

2. **Precision**: Control of false positives
   - Formula: True Positives / (True Positives + False Positives)
   - Answers: "Of all predictions labeled 'face', how many were actually faces?"
   - High precision means fewer false alarms

3. **Recall**: Control of false negatives (misses)
   - Formula: True Positives / (True Positives + False Negatives)
   - Answers: "Of all actual faces, how many did we find?"
   - High recall means we miss fewer faces

4. **F1 Score**: Harmonic mean of precision and recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Balances both precision and recall in a single metric
   - Useful when you need to consider both false positives and false negatives

### Evaluation Process

To compare all models, we run `evaluate_models.py`, which:
1. Loads each trained model from `artifacts/<model>/best_model.pt`
2. Evaluates on the held-out test set
3. Computes accuracy, precision, recall, and F1
4. Generates comparison plots (bar charts for each metric)
5. Creates confusion matrices for the top-performing models
6. Optionally performs bootstrapping to compute confidence intervals

All results are stored in `artifacts/evaluation/`:
- `summary.csv`: Table with all metrics per model
- `accuracy.png`, `f1.png`, `precision.png`, `recall.png`: Bar charts comparing models
- `cm_<model>.png`: Confusion matrices for top-3 models
- `bootstrap_summary.csv`: Mean, std, and confidence intervals (if bootstrapping enabled)
- `*_violin.png`: Violin plots showing metric distributions (if bootstrapping enabled)

### Results Summary

[**Note**: Insert your actual results here. Example format:]

Our best-performing model, **EfficientNet-B0**, achieved:
- **Accuracy**: 0.XX
- **Precision**: 0.XX
- **Recall**: 0.XX
- **F1 Score**: 0.XX

on the test set and processes an image in approximately XX ms/image.

### Key Observations

From the comparison graphs and confusion matrices, we observe:

1. **Pretrained models outperform custom CNNs**: As expected, models fine-tuned from ImageNet achieve higher accuracy due to their rich feature representations learned from millions of images.

2. **Trade-offs among pretrained models**:
   - **EfficientNet**: Catches more real faces (higher recall) but has more false alarms (lower precision)
   - **ResNet18**: Has fewer false alarms (higher precision) but misses more faces (lower recall)
   - **MobileNetV2**: Sits between the other two, offering a balanced trade-off

3. **Custom CNNs show similar accuracy but vary in precision/recall**: While custom CNNs achieve comparable overall accuracy, they differ significantly in precision and recall, suggesting different failure modes.

4. **Model selection depends on application**:
   - **High precision needed** (e.g., security systems where false alarms are costly): Choose ResNet18
   - **High recall needed** (e.g., surveillance where missing faces is critical): Choose EfficientNet
   - **Balanced performance** (e.g., general-purpose applications): Choose MobileNetV2
   - **Resource-constrained environments**: Choose custom CNNs (Improved or Attention variants)

### Confusion Matrix Analysis

The confusion matrices for pretrained models reveal:
- **True Positives (TP)**: Correctly identified faces
- **True Negatives (TN)**: Correctly identified non-faces
- **False Positives (FP)**: Non-faces incorrectly labeled as faces
- **False Negatives (FN)**: Faces that were missed

These matrices help us understand each model's specific failure patterns and guide model selection based on application requirements.

---

## Conclusion

We successfully developed a complete face detection and classification system that scales from simple cropped images to full-scene face detection. The system:

1. **Detects faces** in full images using OpenCV's Haar cascade
2. **Classifies each detected region** using trained models (custom CNNs or pretrained)
3. **Visualizes results** with bounding boxes and confidence scores
4. **Compares multiple models** using comprehensive metrics (accuracy, precision, recall, F1)
5. **Assesses robustness** through bootstrapping confidence intervals

### Key Achievements

- **Model diversity**: Evaluated 12 different architectures, from lightweight custom CNNs to state-of-the-art pretrained models
- **Comprehensive evaluation**: Used multiple metrics and bootstrapping to understand model performance beyond simple accuracy
- **Practical pipeline**: Built an end-to-end system from training to deployment-ready detection

### Lessons Learned

This project helped solidify our understanding of:
- **Deep learning fundamentals**: Architecture design, training procedures, and hyperparameter tuning
- **Transfer learning**: How pretrained models can be adapted to new tasks with limited data
- **Evaluation methodology**: The importance of using multiple metrics and statistical techniques (bootstrapping) to assess model reliability
- **Practical considerations**: Trade-offs between accuracy, speed, and resource requirements

### Future Work

Potential improvements and extensions:
1. **Better detection**: Replace Haar cascade with modern detectors (MTCNN, RetinaFace, or YOLO) for improved face localization
2. **Data augmentation**: Explore more aggressive augmentation strategies (MixUp, CutMix) to improve generalization
3. **Class imbalance**: Implement weighted loss or sampling strategies if the dataset is imbalanced
4. **Real-time inference**: Optimize models for deployment (quantization, pruning) for edge devices
5. **Multi-face scenarios**: Enhance the detection pipeline to handle occlusions and overlapping faces
6. **Video processing**: Extend the system to process video streams in real-time

### Final Thoughts

This project demonstrates the practical application of machine learning concepts learned throughout the course. By building, training, and evaluating multiple models, we gained hands-on experience with the entire deep learning pipeline—from data preprocessing to model deployment. The comparison between custom CNNs and pretrained models highlights the power of transfer learning while also showing that simpler architectures can be valuable in resource-constrained scenarios.

The use of bootstrapping and multiple evaluation metrics ensures that our model selection is based on robust statistical evidence rather than single-point estimates, which is crucial for real-world applications where reliability matters as much as raw performance.

---

**Note**: All code, trained models, and evaluation results are available in the repository. The `README.md` file provides detailed instructions for reproducing our experiments and using the detection pipeline.

