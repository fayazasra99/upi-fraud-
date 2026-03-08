from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app.main import app


def run() -> None:
    client = TestClient(app)

    login_res = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert login_res.status_code == 200, login_res.text
    token = login_res.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}

    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200, me_res.text

    csv_content = """merchant_id,payer_id,amount,currency,device_id,lat,lon,payer_account_age_days,timestamp
MRT-ELECTRO-01,PAYER-CSV-1,1099,INR,DEV-CSV-01,19.0760,72.8777,120,2026-03-07T10:00:00Z
MRT-ELECTRO-01,PAYER-CSV-2,7800,INR,NEWDEV-CSV-99,22.1000,78.3000,3,2026-03-07T11:00:00Z
"""

    upload_res = client.post(
        "/api/transactions/upload-csv",
        headers=headers,
        files={"file": ("batch.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert upload_res.status_code == 200, upload_res.text
    assert upload_res.json()["inserted"] >= 2, upload_res.text

    fraud_txn_res = client.post(
        "/api/transactions/score",
        json={
            "merchant_id": "MRT-ELECTRO-01",
            "payer_id": "PAYER-SMOKE-FRAUD",
            "amount": 9200,
            "currency": "INR",
            "device_id": "NEWDEV-SMOKE-1",
            "lat": 23.2,
            "lon": 77.1,
            "payer_account_age_days": 2,
        },
        headers=headers,
    )
    assert fraud_txn_res.status_code == 200, fraud_txn_res.text
    fraud_txn_id = fraud_txn_res.json()["id"]

    legit_txn_res = client.post(
        "/api/transactions/score",
        json={
            "merchant_id": "MRT-ELECTRO-01",
            "payer_id": "PAYER-SMOKE-LEGIT",
            "amount": 499,
            "currency": "INR",
            "device_id": "DEV-SMOKE-LEGIT",
            "lat": 19.08,
            "lon": 72.88,
            "payer_account_age_days": 450,
        },
        headers=headers,
    )
    assert legit_txn_res.status_code == 200, legit_txn_res.text
    legit_txn_id = legit_txn_res.json()["id"]

    mark_fraud = client.patch(
        f"/api/transactions/{fraud_txn_id}/status",
        json={"status": "confirmed_fraud", "note": "Smoke fraud label"},
        headers=headers,
    )
    assert mark_fraud.status_code == 200, mark_fraud.text

    mark_legit = client.patch(
        f"/api/transactions/{legit_txn_id}/status",
        json={"status": "resolved_legit", "note": "Smoke legit label"},
        headers=headers,
    )
    assert mark_legit.status_code == 200, mark_legit.text

    train_res = client.post(
        "/api/model/train-from-cases",
        json={
            "merchant_id": "MRT-ELECTRO-01",
            "min_samples": 60,
            "max_cases": 1000,
            "allow_synthetic": True,
        },
        headers=headers,
    )
    assert train_res.status_code == 200, train_res.text
    assert train_res.json()["trained_samples"] >= 60, train_res.text

    status_res = client.get("/api/model/status", headers=headers)
    assert status_res.status_code == 200, status_res.text
    assert status_res.json()["loaded"] is True, status_res.text

    print("Smoke test passed")


if __name__ == "__main__":
    run()
