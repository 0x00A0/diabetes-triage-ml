# Diabetes Progression Service

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn diabetes_service.app:app --app-dir src --port 8000
# GET http://127.0.0.1:8000/health
# POST http://127.0.0.1:8000/predict
