import json, os, time, joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERSION = (Path("VERSION").read_text().strip()
           if Path("VERSION").exists() else "v0.2")
MODEL_PATH = OUT_DIR / f"model_{VERSION}.joblib"
METRICS_PATH = OUT_DIR / f"metrics_{VERSION}.json"

def main():
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # 模型 pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=SEED)),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    rmse = mean_squared_error(yte, pred)

    threshold = ytr.mean()  # 均值以上视为高风险
    y_true_flag = (yte > threshold).astype(int)
    y_pred_flag = (pred > threshold).astype(int)

    precision = precision_score(y_true_flag, y_pred_flag)
    recall = recall_score(y_true_flag, y_pred_flag)

    # 保存模型
    joblib.dump(pipe, MODEL_PATH)

    metrics = {
        "version": VERSION,
        "seed": SEED,
        "rmse": rmse,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "timestamp": int(time.time())
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
