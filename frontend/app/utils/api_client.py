import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def predict(scenario: str, payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict/{scenario}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_history(limit: int = 20) -> list[dict]:
    r = requests.get(f"{API_URL}/history?limit={limit}", timeout=5)
    r.raise_for_status()
    return r.json().get("data", [])
