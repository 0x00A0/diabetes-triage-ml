import json, os, time, joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = (Path("VERSION").read_text().strip()
           if Path("VERSION").exists() else "v0.1")
MODEL_PATH = OUT_DIR / f"model_{VERSION}.joblib"
METRICS_PATH = OUT_DIR / f"metrics_{VERSION}.json"

def main():
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    rmse = mean_squared_error(yte, pred)

    joblib.dump(pipe, MODEL_PATH)

    metrics = {
        "version": VERSION,
        "seed": SEED,
        "rmse": rmse,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "timestamp": int(time.time())
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))

if __name__ == "__main__":
    main()
