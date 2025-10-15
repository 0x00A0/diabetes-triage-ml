import numpy as np
from .model import get_model

FEATURE_ORDER = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

def predict_one(d: dict) -> float:
    x = np.array([[d[k] for k in FEATURE_ORDER]], dtype=float)
    model = get_model()
    y = float(model.predict(x)[0])
    return y
