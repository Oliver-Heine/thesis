# QR Quishing Detector: AI-Powered Malicious QR Code Scanner

![CI - Model Training](https://img.shields.io/badge/CI-model__training-blue)
![CI - Backend](https://img.shields.io/badge/CI-backend__server-green)
![CI - Android](https://img.shields.io/badge/CI-android__app-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

**Quishing** (QR phishing) is the practice of embedding malicious URLs inside QR codes to trick users into visiting harmful websites. Because QR codes are opaque to the human eye, users cannot inspect the encoded URL before scanning it—making quishing a rapidly growing attack vector.

This project provides a three-component system for detecting malicious QR codes in real time:

1. **ML Model Training** (`model_training/`) — Fine-tunes lightweight Transformer models (DistilBERT, TinyBERT, ALBERT, MobileBERT) on a ~651k-row URL dataset to perform binary malicious/benign classification.
2. **Android App** (`android_app/`) — Scans QR codes via the device camera, runs on-device TFLite inference, and warns the user before following any URL.
3. **Java Backend** (`backend_server/`) — Spring Boot REST API that augments on-device inference with Playwright-based feature extraction and an LLM-assisted analysis stage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Android App                          │
│  ┌─────────────┐   QR scan   ┌──────────────────────────┐  │
│  │  ZXing Cam  │────────────▶│  TFLite Inference Engine  │  │
│  └─────────────┘             │  (model.tflite in assets) │  │
│                              └──────────┬───────────────┘  │
│                                         │ confidence < 0.80 │
│                                         ▼                   │
│                              ┌──────────────────────────┐  │
│                              │  OkHttp → POST /validate  │  │
│                              └──────────┬───────────────┘  │
└─────────────────────────────────────────┼───────────────────┘
                                          │ HTTPS
                              ┌───────────▼───────────────────┐
                              │     Backend Server (Java)      │
                              │  Spring Boot :8080             │
                              │  ┌─────────────────────────┐  │
                              │  │ UrlValidatorController  │  │
                              │  │  + Rate Limiter          │  │
                              │  └──────────┬──────────────┘  │
                              │             │                  │
                              │  ┌──────────▼──────────────┐  │
                              │  │  UrlFeatureExtractor     │  │
                              │  │  (Playwright Chromium)   │  │
                              │  └──────────┬──────────────┘  │
                              │             │                  │
                              │  ┌──────────▼──────────────┐  │
                              │  │  LlmAnalyzer             │  │
                              │  │  (OpenAI API)            │  │
                              │  └─────────────────────────┘  │
                              └───────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Model Training (Python)                   │
│  malicious-urls.csv → data.py → train.py → evaluate.py     │
│  → model.onnx / model.tflite (exported to android assets)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Model Training (`model_training/`)

Fine-tunes four lightweight Transformer models for URL classification.

**Dataset:** `malicious-urls.csv` from Kaggle  
- ~651,000 rows  
- Columns: `url` (string), `label` (string category), `result` (int, 0 = benign / 1 = malicious)

**Setup:**

```bash
cd model_training
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place malicious-urls.csv in model_training/ (not tracked by git)
python src/train.py                # trains all models, exports ONNX + TFLite
python src/evaluate.py             # prints comparison table
```

**Outputs:**
- `checkpoints/<model>/` — best PyTorch checkpoint per model
- `exports/onnx/<model>.onnx` — ONNX export
- `exports/tflite/<model>.tflite` — TFLite export (copy to `android_app/app/src/main/assets/`)
- `results/metrics.csv` — per-model metrics table

---

### 2. Android App (`android_app/`)

Real-time QR scanner with on-device TFLite inference.

**Requirements:** Android Studio Hedgehog+, JDK 17+, Android SDK 34.

**Setup:**

```bash
cd android_app

# 1. Copy the best TFLite model:
cp ../model_training/exports/tflite/<best_model>.tflite \
   app/src/main/assets/model.tflite

# 2. Copy the tokenizer vocab (from HuggingFace cache or training output):
cp <vocab_file> app/src/main/assets/vocab.txt

# 3. Open in Android Studio → Run on emulator/device (API 26+)
```

See [`android_app/README.md`](android_app/README.md) for AVD setup and testing instructions.

---

### 3. Backend Server (`backend_server/`)

Spring Boot service providing deeper URL analysis for uncertain cases.

**Requirements:** JDK 21, Maven 3.9+, Docker (optional).

**Setup (local):**

```bash
cd backend_server

# Set API keys (never commit these):
export OPENAI_API_KEY=sk-...
export VT_API_KEY=<virustotal_key>

mvn spring-boot:run
# → http://localhost:8080/validate
```

**Setup (Docker):**

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
```

**API:**

```
POST /validate
Content-Type: application/json

{"url": "https://example.com"}

→ {"verdict": "benign", "confidence": 0.97, "features": {...}}
```

See [`backend_server/README.md`](backend_server/README.md) for full API docs.

---

## Repository Structure

```
thesis/
├── model_training/
│   ├── src/
│   │   ├── data.py          # dataset loading & preprocessing
│   │   ├── train.py         # fine-tuning loop + ONNX/TFLite export
│   │   └── evaluate.py      # metrics & model comparison
│   ├── tests/
│   │   ├── test_data.py
│   │   └── test_evaluate.py
│   ├── notebooks/           # Jupyter experiments (M1 Mac)
│   ├── config.yaml
│   └── requirements.txt
├── android_app/
│   ├── app/src/main/
│   │   ├── kotlin/com/thesis/qrquishing/
│   │   │   └── MainActivity.kt
│   │   ├── res/layout/
│   │   │   ├── activity_main.xml
│   │   │   └── dialog_warning.xml
│   │   └── AndroidManifest.xml
│   ├── build.gradle.kts
│   ├── app/build.gradle.kts
│   ├── settings.gradle.kts
│   └── gradle/libs.versions.toml
├── backend_server/
│   ├── src/main/java/com/thesis/qrquishing/
│   │   ├── UrlValidatorController.java
│   │   ├── UrlFeatureExtractor.java
│   │   ├── LlmAnalyzer.java
│   │   ├── UrlFeatures.java
│   │   ├── ValidationRequest.java
│   │   └── ValidationResponse.java
│   ├── src/main/resources/application.properties
│   ├── pom.xml
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    └── README.md
```

---

## Security Notes

- API keys are loaded from environment variables only — never committed to source.
- The backend blocks SSRF by rejecting private/loopback IP targets.
- Docker containers run as non-root user (UID 1000) with `no-new-privileges` and all capabilities dropped.
- The Android app enforces a confidence threshold (0.80) before trusting an inference result.

---

## Thesis Documentation

Full thesis write-up, experimental results, and figures are maintained separately in [`docs/`](docs/).

---

## License

[MIT](LICENSE) © Oliver Heine, 2025
