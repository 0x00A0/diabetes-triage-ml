from pathlib import Path
import os


def read_version_file() -> str:
    p = Path("VERSION")
    return p.read_text().strip() if p.exists() else "v0.1"


MODEL_VERSION = os.getenv("MODEL_VERSION", read_version_file())
MODEL_PATH = os.getenv("MODEL_PATH", f"models/model_{MODEL_VERSION}.joblib")
