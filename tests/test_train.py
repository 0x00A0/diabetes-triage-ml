from pathlib import Path
import json, subprocess

def test_train_produces_artifacts():
    subprocess.check_call(["python", "train.py"])
    v = Path("VERSION").read_text().strip()
    assert Path(f"models/model_{v}.joblib").exists()
    m = json.loads(Path(f"models/metrics_{v}.json").read_text())
    assert "rmse" in m and m["rmse"] > 0
