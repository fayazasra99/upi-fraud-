from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", str(BASE_DIR / "fraud_shield.db")))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "70"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "45"))

AUTH_SECRET = os.getenv("AUTH_SECRET", "change-this-demo-secret")
TOKEN_TTL_MINUTES = int(os.getenv("TOKEN_TTL_MINUTES", "480"))

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", str(ARTIFACTS_DIR / "fraud_model.pkl")))
MODEL_WEIGHT = float(os.getenv("MODEL_WEIGHT", "0.30"))
