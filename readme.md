# Virtual Diabetes Clinic Triage (ML Service)

This project simulates a **virtual diabetes clinic triage** system.  
A small FastAPI service predicts a diabetes progression score to help nurses prioritize patient follow-ups.

---

## Overview
- Dataset: `sklearn.datasets.load_diabetes()`
- Target: "progression index" (higher = worse)
- v0.1: StandardScaler + LinearRegression
- v0.2: StandardScaler + Ridge Regression (+ high-risk flag)
- Pipeline: CI/CD via GitHub Actions → build → push Docker image to GHCR

---

## How to Run Locally

### 1. Install and train
```bash
pip install -r requirements.txt
python train.py
```

### 2. Start API
```bash
uvicorn diabetes_service.app:app --app-dir src --port 8000
```

Then visit:

GET `http://127.0.0.1:8000/health`

POST `http://127.0.0.1:8000/predict`

### 3. Example request

#### Health check
```bash
curl http://127.0.0.1:8000/health
```
→ `{"status":"ok","model_version":"v0.2"}`

#### Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,
     "s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```
→ `{"prediction": 145.2}`

#### Error handling

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"age":"abc"}'
```
→ Returns JSON error with details

### 4. Docker

#### Build and run

```bash
docker build -t diabetes:v0.2 .
docker run -p 8000:8000 diabetes:v0.2
```

### GHCR

#### Pull the prebuilt image:

```bash
docker pull ghcr.io/0x00a0/diabetes-triage-ml:v0.2
docker run -p 8000:8000 ghcr.io/0x00a0/diabetes-triage-ml:v0.2
```

---

Author: Xinrui Wan

Stockholm University — MSc AI for Health



