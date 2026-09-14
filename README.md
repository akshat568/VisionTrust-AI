# VisionTrust AI — Reliability-Aware Computer Vision for Detecting Model Failure Under Distribution Shift

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Tailwind CSS v4](https://img.shields.io/badge/Tailwind-v4-38B2AC.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-50%20Passed-emerald.svg)]()

---

## Overview

**VisionTrust AI** is a reliability-aware computer vision framework designed to evaluate, quantify, and detect vision model failure under distribution shift (e.g., image corruptions, domain shifts, and out-of-distribution inputs).

While standard computer vision models produce overconfident softmax probabilities even when severely wrong, **VisionTrust AI** extracts a multi-dimensional reliability signal vector and passes it through a separate **Random Forest Trust Model** to estimate $P(\text{original vision prediction is correct})$.

---

## Problem Statement

Deep neural network vision classifiers are routinely deployed in safety-critical domains (autonomous driving, medical imaging, quality inspection). However, standard deep learning models exhibit a fundamental flaw:
- **Overconfidence under Shift**: Softmax probabilities do not represent true predictive probability under distribution shift.
- **Silent Failures**: Models frequently assign $>90\%$ softmax confidence to wildly incorrect predictions when processing noisy, blurred, or corrupted images.
- **Lack of Guardrails**: Downstream automated systems have no native mechanism to distinguish between a trustworthy high-confidence prediction and a dangerous overconfident misclassification.

---

## Why Confidence Is Not Reliability

Softmax probability $\max_c p_c$ is calculated by normalizing uncalibrated class logits $z$:
$$p_c = \frac{\exp(z_c)}{\sum_{k=1}^C \exp(z_k)}$$

Because the exponential function amplifies the largest logit regardless of feature representation quality, a model forced to classify an out-of-distribution or heavily corrupted image will output a high softmax score simply because one logit happens to be slightly larger than the others.

**Key Insight of VisionTrust AI**:
Evaluating prediction reliability requires looking beyond the top softmax output. By combining **prediction entropy**, **penultimate feature-space centroid distance**, **logit free-energy OOD scores**, **test-time augmentation consistency**, and **image quality metrics**, we can train a separate meta-classifier that reliably predicts whether the vision model's output should be trusted.

---

## System Architecture

```text
                                [ User Image Upload ]
                                         │
                                         ▼
                            [ React 19 Web Dashboard ]
                                (http://localhost:5173)
                                         │
                                         ▼  (HTTP POST /predict - multipart/form-data)
                                [ FastAPI Backend API ]
                                (http://127.0.0.1:8000)
                                         │
                                         ▼
                             [ VisionTrustPipeline ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [ Raw Image Tensor [0,1] ]                      [ Normalized Tensor μ/σ ]
                 │                                               │
     [ Image Quality Extractor ]                      [ Frozen ResNet-18 Model ]
        ├─ Sharpness                                             │
        ├─ Brightness                        ┌───────────────────┼───────────────────┐
        ├─ Contrast                          ▼                   ▼                   ▼
        └─ Composite Quality          [ Logits & Probs ]  [ 512-D Features ]  [ K=3 Augmentations ]
                                             │                   │                   │
                                      Softmax Conf,       Centroid Distance   Aug Consistency
                                      Entropy, OOD Energy        │                   │
                                             │                   │                   │
                                             └───────────────────┼───────────────────┘
                                                                 │
                                                                 ▼
                                                  [ 9 Reliability Features ]
                                                                 │
                                                                 ▼
                                                   [ Random Forest Trust Model ]
                                                                 │
                                                                 ▼
                                                    P(correct) Trust Score
                                                                 │
                                                                 ▼
                                                   [ Decision Threshold 0.50 ]
                                                                 │
                                                 ┌───────────────┴───────────────┐
                                                 ▼                               ▼
                                         [ "TRUSTED" ]                   [ "UNTRUSTED" ]
```

---

## Methodology

### 1. Baseline Vision Predictor
- **Architecture**: CIFAR-10 adapted ResNet-18 ($3 \times 3$ stride-1 stem, 512-D bottleneck, no maxpool).
- **Status**: Strictly frozen (`outputs/models/baseline_resnet18_best.pth`).
- **Clean Test Accuracy**: **81.25%** on 10,000 clean CIFAR-10 test images.

### 2. Controlled Distribution Shifts
25 evaluation datasets generated across 5 corruption families at Severities 1 to 5 ($250,000$ test samples):
- **Blur**: Gaussian Blur
- **Noise**: Gaussian Noise
- **Brightness**: Brightness Scaling
- **Contrast**: Contrast Reduction
- **Rotation**: Random Image Rotation

### 3. Six Reliability Signal Categories (9 Feature Columns)
1. **Softmax Confidence**: Maximum probability $\max_c p_c \in [0, 1]$ (Higher = More Reliable).
2. **Prediction Entropy**: Shannon entropy $H(p) = -\sum p_c \log p_c \ge 0$ (Lower = More Reliable).
3. **Feature-Space Centroid Distance**: Euclidean distance in 512-D feature space $\|\mathbf{f} - \boldsymbol{\mu}_{\hat{y}}\|_2$ to the pre-computed training centroid of predicted class $\hat{y}$ (Lower = More Reliable). Centroids are pre-computed strictly from 45,000 training samples.
4. **OOD / Logit Energy Score**: Free energy $E(\mathbf{x}; T) = T \cdot \log \sum_c \exp(z_c / T)$ at $T=1.0$ (Higher = In-Distribution).
5. **Augmentation Consistency**: Agreement fraction under $K=3$ deterministic test-time augmentations (flip, center-crop/resize, and 5° rotation) in $[0, 1]$ (Higher = More Reliable).
6. **Image Quality Diagnostics**: Sharpness (Laplacian variance), Brightness, Contrast, and Composite Quality.

### 4. Separate Trust / Reliability Model
- **Model**: Random Forest (100 trees, max_depth=5) (`outputs/models/trust_model.pkl`).
- **Training Data**: Fitted strictly on 5,000 clean validation signal samples with 5-fold cross-validation. Zero test-set leakage.
- **Target**: Binary prediction correctness ($1 = \text{Correct}, 0 = \text{Incorrect}$).
- **Output**: Estimated $P(\text{original vision prediction is correct}) \in [0, 1]$.

---

## Experimental Results

### 1. Baseline & Trust Model Failure AUROC / AUPRC

All metrics evaluate the model's ability to discriminate correct vs. incorrect predictions on clean test data ($10,000$ samples):

| Reliability Mechanism | Input Features | Failure AUROC (%) | Failure AUPRC | ECE (%) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Raw Softmax Confidence** | Confidence only | 85.61% | 0.5351 | 1.58% | 0.1131 |
| **Production Trust Model (RF)** | All 9 Signal Features | **86.69%** | **0.5708** | **1.21%** | **0.1077** |
| **Top Ablation Finding (H)** | Without Image Quality | **86.74%** | **0.5747** | **1.22%** | **0.1075** |
| **Core 4 Signals (K)** | Conf + Ent + Dist + AugCons | **86.60%** | **0.5724** | 1.27% | 0.1079 |

*Statistical Significance*: Non-parametric bootstrap resampling ($1,000$ iterations) confirms the Trust Model achieves a statistically significant AUROC gain over raw confidence (**+0.57%**, 95% CI: `[0.25%, 0.88%]`, $p < 0.0001$).

### 2. Individual Signal Category Performance

| Rank | Reliability Signal | Clean AUROC (%) | AUPRC | Primary Direction |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Softmax Confidence** | **85.61%** | 0.9633 | Higher = More Reliable |
| 2 | **Prediction Entropy** | **85.21%** | 0.9625 | Lower = More Reliable |
| 3 | **Augmentation Consistency** | **74.13%** | 0.8955 | Higher = More Reliable |
| 4 | **Feature Centroid Distance** | **55.62%** | 0.8513 | Lower = More Reliable |
| 5 | **Composite Quality** | **53.71%** | 0.8286 | Diagnostic Metric |
| 6 | **OOD Logit Energy** | **50.18%** | 0.8054 | Higher = In-Distribution |

### 3. Selective Prediction & Risk-Coverage (80% Coverage)

When deferred decision-making is enabled by withholding the 20% lowest-reliability predictions:
- **Raw Confidence Selection**: Retained accuracy = **89.85%**
- **Trust Model Selection**: Retained accuracy = **90.33%** (**+0.48% gain**)

---

## Application Demo

The project includes an interactive React 19 web dashboard connected to a live FastAPI backend:

- **Drag-and-Drop Image Upload**: Supports PNG, JPG, WEBP, BMP up to 10MB.
- **Hero Result Summary Card**: Predicted class name, confidence %, trust probability %, failure probability %, and prominent `TRUSTED` / `UNTRUSTED` status badge.
- **Confidence vs Trust Contrast**: Visual comparison highlighting overconfident misclassifications under distribution shift.
- **6 Reliability Signal Cards**: Individual progress bars, directional indicators, and tooltips for all signal categories.
- **Dynamic Trust Interpretation**: Conservative, non-causal observations ("Why this prediction is rated this way").
- **10-Class Probability Bar Chart**: Complete softmax probability breakdown sorted descending.

---

## Tech Stack

- **Deep Learning**: PyTorch 2.0+, Torchvision
- **Machine Learning**: Scikit-Learn, NumPy, Pandas, SciPy, Joblib
- **Backend API**: FastAPI, Uvicorn, Pydantic V2, Python-Multipart, HTTPX
- **Frontend Dashboard**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide React
- **Testing**: Pytest, Vitest, React Testing Library, JSDOM

---

## Project Structure

```text
VisionTrust-AI/
│
├── frontend/                  # React 19 + TypeScript + Tailwind Web Application (Phase 11)
│   ├── src/
│   │   ├── components/        # Header, ImageUploader, ResultSummaryCard, ConfidenceVsTrustCard, etc.
│   │   ├── services/          # api.ts (Centralized HTTP client)
│   │   ├── types/             # api.ts (Pydantic-matched TypeScript interfaces)
│   │   ├── __tests__/         # Vitest component unit tests
│   │   └── App.tsx            # Main dashboard container
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
│
├── configs/                   # Experiment configuration YAMLs
├── data/                      # CIFAR-10 raw, processed, and shifted datasets
├── outputs/                   # Models (`baseline_resnet18_best.pth`, `trust_model.pkl`), centroids, plots
├── scripts/                   # CLI predict script, train scripts, shift evaluation scripts
│
├── src/                       # Core Python Package
│   ├── api/                   # FastAPI Backend Application (Phase 10)
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/                  # Loaders, stratified splitters, transforms, corruptions
│   ├── models/                # ResNet18CIFAR architecture & trainer
│   ├── evaluation/            # Evaluation metrics
│   ├── reliability/           # Signals, Trust Model, Ablation & Robustness modules
│   ├── inference/             # VisionTrustPipeline & InferenceResult (Phase 9)
│   └── utils/                 # Seed setting, logging
│
├── tests/                     # 50 Python unit and integration tests (100% passing)
├── requirements.txt           # Python dependencies
├── PHASE_12_COMPLETION_REPORT.md # Final QA & Audit Report
└── README.md                  # Project documentation
```

---

## Installation & Environment Setup

### 1. Clone Repository & Setup Python Environment
```powershell
git clone https://github.com/user/VisionTrust-AI.git
cd VisionTrust-AI

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Setup Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

---

## How to Run

### 1. Run All Backend Unit & Integration Tests (50 Tests)
```powershell
pytest tests/
```

### 2. Run All Frontend Unit Tests (3 Tests)
```powershell
cd frontend
npm test
cd ..
```

### 3. Build Frontend Production Bundle
```powershell
cd frontend
npm run build
cd ..
```

### 4. Start Full Stack (Backend + Frontend)

**Terminal 1 — Start FastAPI Backend:**
```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

**Terminal 2 — Start React Frontend Dashboard:**
```powershell
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.

### 5. CLI Inference Prediction Utility
```powershell
python scripts/predict.py --image scratch/test_images/sample1.png
```

---

## API Reference

### `GET /health`
Returns backend liveness status and model initialization state. Does **NOT** run model inference.
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Accepts single image upload (`multipart/form-data`). Runs real ML inference and returns prediction, 6 reliability signal categories, and Trust Model outputs.

### `POST /predict/batch`
Accepts multiple image uploads and returns a batch of `PredictResponse` objects.

---

## Reproducibility & Code Health

- **100% Deterministic Seed**: All stratified data splits, corruption generation, and model evaluations use seed `42`.
- **Zero Mock Policy**: The running web dashboard and backend API contain zero mock predictions or hard-coded confidence metrics.
- **Clean Test Health**: 50/50 Python backend tests passing, 3/3 Frontend Vitest tests passing, 0 TypeScript build errors.

---

## Limitations

- **Benchmark Dataset Scope**: VisionTrust AI is evaluated on CIFAR-10 ($32 \times 32$ pixels). Real-world high-resolution images may present different corruption dynamics.
- **Controlled Shift Conditions**: Distribution shifts evaluated consist of synthetic image corruptions (blur, noise, brightness, contrast, rotation).
- **Feature Distance Metric**: Euclidean distance in raw 512-D bottleneck feature space exhibits moderate standalone discriminative power; learned distance metrics (e.g. Mahalanobis distance) offer room for future improvement.
- **Probabilistic Estimate**: The Trust Model outputs an *estimated probability* $P(\text{correct})$, not a deterministic proof of correctness.

---

## Future Work

- **Learned Feature Metrics**: Replace Euclidean centroid distance with class-conditional Mahalanobis distance.
- **Deep Ensembles & Conformal Prediction**: Incorporate ensemble variance and conformal risk control for distribution-free coverage guarantees.
- **High-Resolution Medical/Industrial Benchmarks**: Extend reliability signal extraction to high-resolution domain shift datasets (e.g., WILDS, ImageNet-C).

---

## Interview & Technical Summary

When presenting VisionTrust AI in technical interviews or portfolio reviews, key talking points include:
1. **Core Problem Solved**: "Neural networks are notoriously overconfident under distribution shift. VisionTrust AI builds a secondary guardrail to flag overconfident errors before downstream decisions."
2. **Multi-Signal Fusion**: "Rather than relying on softmax confidence alone, we extract 9 features across 6 categories (entropy, feature distance, OOD energy, test-time augmentation consistency, and quality) to train a Random Forest Trust Model."
3. **Strict Zero-Leakage Protocol**: "The baseline ResNet-18 remains 100% frozen. The Trust Model is trained strictly on 5,000 clean validation signals with 5-fold cross-validation, achieving an 86.69% AUROC on clean test data."
4. **Full-Stack Implementation**: "Packaged into a single entry-point inference pipeline, served via a FastAPI backend, and visualized through an interactive React + Tailwind CSS dashboard."

---

## Project Highlights

- **81.25%** Clean CIFAR-10 Baseline Accuracy
- **25** Controlled Distribution-Shift Datasets ($250,000$ samples)
- **6** Reliability Signal Categories (9 total features)
- **86.69%** Clean Test Failure AUROC (Production Trust Model)
- **86.74%** Top Signal Ablation AUROC
- **90.33%** Retained Accuracy at 80% Coverage (Selective Prediction)
- **50** Passing Python Backend Tests
- **3** Passing React Vitest Tests
