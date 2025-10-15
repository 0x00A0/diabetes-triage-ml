from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .schemas import Features, Prediction
from .predict import predict_one
from .config import MODEL_VERSION


app = FastAPI(title="Diabetes Progression Service")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    y = predict_one(payload.dict())
    return {"prediction": y}


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})
