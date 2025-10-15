from pathlib import Path
import json
import subprocess


def test_train_produces_artifacts():
    subprocess.check_call(["python", "train.py"])
    v = Path("VERSION").read_text().strip()
    model = Path(f"models/model_{v}.joblib")
    metrics_file = Path(f"models/metrics_{v}.json")

    assert model.exists()
    m = json.loads(metrics_file.read_text())
    assert "rmse" in m and m["rmse"] > 0
