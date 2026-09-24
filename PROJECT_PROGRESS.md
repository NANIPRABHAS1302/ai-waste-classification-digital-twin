# PROJECT PROGRESS

## Project Title

**Efficient Lightweight CNN-Based Automated Waste Classification for Circular Economy and Sustainable Urban Waste Management**

---

## Conference

South Asian Conference on Sustainability & Climate Justice (SACSCJ) 2026

---

## Project Goal

Develop an intelligent deep learning framework capable of automatically classifying municipal solid waste into six categories while maintaining high classification accuracy and computational efficiency suitable for sustainable urban waste management and edge deployment.

---

## Current Status

**Overall Completion:** 99%

```text
███████████████████████████████████████████████████░ 99%
```

---

## Technologies Used

### Programming Language

- Python 3.11

### IDE

- Visual Studio Code

### Frameworks

- TensorFlow
- Keras

### Libraries

- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-Learn

### Dataset

- TrashNet Dataset

### Models

- Baseline CNN
- MobileNetV2
- EfficientNet-B0

---
# Development Timeline

## Phase 1 — Dataset Preparation

Status: ✅ Completed

Tasks:

- Dataset organization
- Train / Validation / Test split
- Folder generation
- Class balancing
- Image resizing
- Data preprocessing

Output:

- dataset/train/
- dataset/val/
- dataset/test/

---

## Phase 2 — Model Training

Status: ✅ Completed

Implemented:

- Baseline CNN
- MobileNetV2
- EfficientNet-B0

Features:

- Early Stopping
- Class Weights
- Transfer Learning
- Data Augmentation

Outputs:

- Saved trained models
- Training history
- JSON logs

---

## Phase 2B — EfficientNet Improvement

Status: ✅ Completed

Improvements:

- Official EfficientNet preprocessing
- Fine-tuning
- Learning-rate scheduling

Result:

Accuracy improved significantly after correcting preprocessing.

---

## Phase 3 — Model Evaluation

Status: ✅ Completed

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Weighted F1
- Confusion Matrix
- Inference Time
- Model Size

Generated:

- Evaluation reports
- CSV comparison
- JSON summary

---
# Publication Assets

## Phase 4 — Figure Generation

Status: ✅ Completed

Generated:

- Training Curves
- Accuracy Comparison
- Weighted F1 Comparison
- Model Size Comparison
- Inference Time Comparison
- Confusion Matrices

Total Figures:

10

---

## Phase 5 — Grad-CAM

Status: ✅ Completed

Generated:

- Cardboard Grad-CAM
- Glass Grad-CAM
- Metal Grad-CAM
- Paper Grad-CAM
- Plastic Grad-CAM
- Trash Grad-CAM

Output:

results/gradcam/

---

## Phase 6 — Publication Tables

Status: ✅ Completed

Generated:

Table 1

Dataset Statistics

Table 2

Model Comparison

Table 3

Per-Class Performance

Table 4

Training Configuration

Table 5

Model Complexity

---

## Phase 7 — Paper Assets

Status: ✅ Completed

Generated:

- Paper figures
- Paper tables
- References folder
- Assets summary

---
# Problems Encountered

## Problem 1

Incorrect dataset paths.

Solution:

Dynamic BASE_DIR implementation.

---

## Problem 2

Initial EfficientNet accuracy was extremely poor (~5%).

Cause:

Incorrect preprocessing.

Solution:

Official EfficientNet preprocessing and fine-tuning.

Result:

Test Accuracy ≈ 79.90%

---

## Problem 3

Grad-CAM graph connection error.

Error:

ValueError: Output with path 0 is not connected to inputs.

Cause:

Nested Functional Keras model.

Solution:

Rebuilt Grad-CAM implementation compatible with the nested EfficientNet model.

---

# Final Model Performance

| Model | Accuracy | Weighted F1 |
|--------|---------:|------------:|
| Baseline CNN | 50.65% | 49.69% |
| MobileNetV2 | 74.15% | 74.15% |
| EfficientNet-B0 | 79.90% | 79.73% |

Best Model:

EfficientNet-B0

---
# Remaining Work

## Repository

- README Review
- GitHub Upload

---

## Conference

- Complete manuscript
- Prepare presentation
- Final submission

---

# Future Enhancements

- TensorFlow Lite conversion
- Edge deployment
- Mobile application
- IoT smart-bin integration
- Larger datasets
- ONNX export
- Docker support
- Real-time webcam classification

---

# Repository Status

```text
Dataset                 ✅
Training                ✅
Evaluation              ✅
Figures                 ✅
Grad-CAM                ✅
Tables                  ✅
Paper Assets            ✅
README                  ✅
requirements.txt        ✅
.gitignore              ✅
LICENSE                 ✅

Conference Paper        ⏳
Presentation            ⏳
GitHub Release          ⏳
```

---

## Author

Nani Prabhas

B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)

---

**Last Updated:** August 2026
