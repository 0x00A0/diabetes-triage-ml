import subprocess
import time

import requests


def test_api_smoke():
    # start a local server
    proc = subprocess.Popen(
        [
            "uvicorn",
            "diabetes_service.app:app",
            "--app-dir",
            "src",
            "--port",
            "8001",
        ]
    )
    try:
        time.sleep(1.5)

        resp = requests.get("http://127.0.0.1:8001/health", timeout=5)
        assert resp.status_code == 200

        sample = {
            "age": 0.02,
            "sex": -0.044,
            "bmi": 0.06,
            "bp": -0.03,
            "s1": -0.02,
            "s2": 0.03,
            "s3": -0.02,
            "s4": 0.02,
            "s5": 0.02,
            "s6": -0.001,
        }

        pr = requests.post(
            "http://127.0.0.1:8001/predict",
            json=sample,
            timeout=5,
        )
        assert pr.status_code == 200
        body = pr.json()
        assert "prediction" in body and isinstance(body["prediction"], (int, float))
    finally:
        proc.terminate()
