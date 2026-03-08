from __future__ import annotations

import csv
import io
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .auth import (
    AuthUser,
    authenticate_user,
    create_access_token,
    ensure_merchant_scope,
    get_current_user,
    require_analyst,
    user_profile_dict,
)
from .config import BASE_DIR, BLOCK_THRESHOLD, DB_PATH, MODEL_PATH, MODEL_WEIGHT, REVIEW_THRESHOLD
from .db import Repository
from .ml_model import FraudModelService
from .model_training import train_model_artifact_from_cases
from .risk_engine import build_features, decision_to_initial_status, evaluate_transaction
from .schemas import (
    CsvUploadResult,
    LoginRequest,
    LoginResponse,
    MerchantSummary,
    Metrics,
    ModelStatus,
    ModelTrainRequest,
    ModelTrainResult,
    NoteUpdate,
    RuleCreate,
    RuleRecord,
    StatusUpdate,
    ToggleRuleRequest,
    TransactionIn,
    TransactionRecord,
    UserProfile,
)

app = FastAPI(
    title="UPI Merchant Fraud Shield",
    version="0.3.0",
    description="Real-time transaction scoring + case workflow + model retraining.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = Repository(DB_PATH)
model_service = FraudModelService(MODEL_PATH)

FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


MERCHANT_PROFILES = [
    {"id": "MRT-ELECTRO-01", "lat": 19.0760, "lon": 72.8777, "mean": 980},
    {"id": "MRT-GROCERY-07", "lat": 12.9716, "lon": 77.5946, "mean": 610},
    {"id": "MRT-APPAREL-03", "lat": 28.6139, "lon": 77.2090, "mean": 1350},
    {"id": "MRT-FOOD-11", "lat": 17.3850, "lon": 78.4867, "mean": 420},
]

VALID_DECISIONS = {"allow", "review", "block"}
VALID_STATUSES = {
    "allow",
    "review",
    "block",
    "investigating",
    "resolved_legit",
    "confirmed_fraud",
}

MAX_CSV_ERRORS = 50
MAX_CSV_ROWS = 5000


@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> dict[str, Any]:
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token, expires_in = create_access_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_seconds": expires_in,
        "user": user_profile_dict(user),
    }


@app.get("/api/auth/me", response_model=UserProfile)
def me(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    return user_profile_dict(user)


@app.get("/api/merchants", response_model=list[str])
def list_merchants(user: AuthUser = Depends(get_current_user)) -> list[str]:
    if user.role == "merchant_admin":
        if not user.merchant_id:
            return []
        return [user.merchant_id]

    known = {profile["id"] for profile in MERCHANT_PROFILES}
    observed = set(repo.list_merchants())
    return sorted(known | observed)


@app.get("/api/merchants/summary", response_model=list[MerchantSummary])
def merchant_summary(user: AuthUser = Depends(get_current_user)) -> list[dict[str, Any]]:
    summary = repo.merchant_summary_24h()

    if user.role != "merchant_admin":
        return summary

    if not user.merchant_id:
        return []

    return [row for row in summary if row["merchant_id"] == user.merchant_id]


@app.get("/api/model/status", response_model=ModelStatus)
def model_status(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    _ = user
    return model_service.status()


@app.post("/api/model/reload", response_model=ModelStatus)
def model_reload(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    require_analyst(user)
    model_service.load()
    return model_service.status()


@app.post("/api/model/train-from-cases", response_model=ModelTrainResult)
def model_train_from_cases(
    payload: ModelTrainRequest,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    require_analyst(user)

    merchant_id = _normalize_optional(payload.merchant_id)

    cases = repo.list_labeled_cases(
        merchant_id=merchant_id,
        limit=payload.max_cases,
    )

    try:
        trained = train_model_artifact_from_cases(
            cases=cases,
            model_path=MODEL_PATH,
            min_samples=payload.min_samples,
            allow_synthetic=payload.allow_synthetic,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    model_service.load()

    return {
        "merchant_id": merchant_id,
        **trained,
    }


@app.post("/api/transactions/score", response_model=TransactionRecord)
def score_transaction(
    payload: TransactionIn,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    merchant_id = ensure_merchant_scope(user, payload.merchant_id)
    transaction = payload.model_copy(update={"merchant_id": merchant_id or payload.merchant_id})
    return _score_and_store(transaction)


@app.post("/api/transactions/seed")
def seed_transactions(
    count: int = Query(default=40, ge=1, le=300),
    suspicious_ratio: float = Query(default=0.25, ge=0.0, le=1.0),
    merchant_id: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    scoped_merchant = ensure_merchant_scope(user, _normalize_optional(merchant_id))

    inserted = 0
    for _ in range(count):
        payload = _generate_synthetic_payload(suspicious_ratio, scoped_merchant)
        _score_and_store(payload)
        inserted += 1

    return {"inserted": inserted}


@app.post("/api/transactions/upload-csv", response_model=CsvUploadResult)
async def upload_transactions_csv(
    file: UploadFile = File(...),
    merchant_id: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    scoped_merchant = ensure_merchant_scope(user, _normalize_optional(merchant_id))

    raw_content = await file.read()
    if not raw_content:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    try:
        csv_text = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded") from error

    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV header row is missing")

    total_rows = 0
    inserted = 0
    failed = 0
    errors: list[dict[str, Any]] = []

    for row_number, row in enumerate(reader, start=2):
        total_rows += 1

        if total_rows > MAX_CSV_ROWS:
            failed += 1
            if len(errors) < MAX_CSV_ERRORS:
                errors.append({"row": row_number, "message": f"Row limit exceeded ({MAX_CSV_ROWS})"})
            continue

        try:
            payload = _transaction_from_csv_row(row, scoped_merchant)
            resolved_merchant = ensure_merchant_scope(user, payload.merchant_id)
            payload = payload.model_copy(update={"merchant_id": resolved_merchant or payload.merchant_id})
            _score_and_store(payload)
            inserted += 1
        except Exception as error:  # noqa: BLE001
            failed += 1
            if len(errors) < MAX_CSV_ERRORS:
                errors.append({"row": row_number, "message": str(error)})

    return {
        "total_rows": total_rows,
        "inserted": inserted,
        "failed": failed,
        "errors": errors,
    }


@app.get("/api/transactions", response_model=list[TransactionRecord])
def list_transactions(
    limit: int = Query(default=30, ge=1, le=200),
    decision: str | None = Query(default=None),
    status: str | None = Query(default=None),
    merchant_id: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
) -> list[dict[str, Any]]:
    if decision is not None and decision not in VALID_DECISIONS:
        raise HTTPException(status_code=400, detail="Decision must be allow, review, or block.")

    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status filter.")

    scoped_merchant = ensure_merchant_scope(user, _normalize_optional(merchant_id))

    return repo.list_transactions(
        limit=limit,
        decision=decision,
        status=status,
        merchant_id=scoped_merchant,
    )


@app.get("/api/transactions/review-queue", response_model=list[TransactionRecord])
def review_queue(
    limit: int = Query(default=40, ge=1, le=200),
    merchant_id: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
) -> list[dict[str, Any]]:
    scoped_merchant = ensure_merchant_scope(user, _normalize_optional(merchant_id))
    return repo.list_review_queue(limit=limit, merchant_id=scoped_merchant)


@app.patch("/api/transactions/{transaction_id}/status", response_model=TransactionRecord)
def update_transaction_status(
    transaction_id: str,
    payload: StatusUpdate,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    transaction = repo.get_transaction(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    ensure_merchant_scope(user, transaction["merchant_id"])

    updated = repo.set_transaction_status(transaction_id, payload.status, payload.note)
    if not updated:
        raise HTTPException(status_code=404, detail="Transaction not found")

    record = repo.get_transaction(transaction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return record


@app.post("/api/transactions/{transaction_id}/note", response_model=TransactionRecord)
def update_case_note(
    transaction_id: str,
    payload: NoteUpdate,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    transaction = repo.get_transaction(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    ensure_merchant_scope(user, transaction["merchant_id"])

    if not repo.add_case_note(transaction_id, payload.note):
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated = repo.get_transaction(transaction_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return updated


@app.get("/api/metrics", response_model=Metrics)
def metrics(
    merchant_id: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    scoped_merchant = ensure_merchant_scope(user, _normalize_optional(merchant_id))
    return repo.metrics_24h(scoped_merchant)


@app.post("/api/rules", response_model=RuleRecord, status_code=201)
def create_rule(
    payload: RuleCreate,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    merchant_id = payload.merchant_id

    if user.role == "merchant_admin":
        merchant_id = ensure_merchant_scope(user, merchant_id)
    elif merchant_id is not None:
        merchant_id = merchant_id.strip() or None

    rule_payload = payload.model_copy(update={"merchant_id": merchant_id})
    return repo.create_rule(_dump_model(rule_payload))


@app.get("/api/rules", response_model=list[RuleRecord])
def list_rules(
    merchant_id: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
) -> list[dict[str, Any]]:
    scoped_merchant = ensure_merchant_scope(user, _normalize_optional(merchant_id))
    return repo.list_rules(merchant_id=scoped_merchant)


@app.patch("/api/rules/{rule_id}/enabled", response_model=RuleRecord)
def toggle_rule(
    rule_id: str,
    payload: ToggleRuleRequest,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    current = repo.get_rule(rule_id)
    if not current:
        raise HTTPException(status_code=404, detail="Rule not found")

    ensure_merchant_scope(user, current["merchant_id"])

    updated = repo.set_rule_enabled(rule_id, payload.enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="Rule not found")

    return updated


def _score_and_store(payload: TransactionIn) -> dict[str, Any]:
    transaction = _dump_model(payload)
    transaction["currency"] = transaction["currency"].upper()
    transaction["timestamp"] = _normalize_timestamp(transaction["timestamp"])

    current_epoch = int(transaction["timestamp"].timestamp())

    velocity_10m = repo.get_velocity_10m(transaction["merchant_id"], current_epoch)
    merchant_avg_amount = repo.get_merchant_average_amount(transaction["merchant_id"], current_epoch)
    payer_last_transaction = repo.get_payer_last_transaction(transaction["payer_id"])
    rules = repo.list_applicable_rules(transaction["merchant_id"])

    features = build_features(
        payload=transaction,
        velocity_10m=velocity_10m,
        merchant_avg_amount=merchant_avg_amount,
        payer_last_transaction=payer_last_transaction,
    )

    model_probability = model_service.predict_probability(features)

    scoring = evaluate_transaction(
        payload=transaction,
        features=features,
        rules=rules,
        review_threshold=REVIEW_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD,
        model_probability=model_probability,
        model_weight=MODEL_WEIGHT,
    )

    record = {
        "id": str(uuid.uuid4()),
        **transaction,
        "txn_epoch": current_epoch,
        "decision": scoring["decision"],
        "status": decision_to_initial_status(scoring["decision"]),
        "risk_score": scoring["risk_score"],
        "heuristic_score": scoring["heuristic_score"],
        "model_probability": scoring["model_probability"],
        "reasons": scoring["reasons"],
        "features": scoring["features"],
        "case_note": None,
    }

    repo.insert_transaction(record)
    return record


def _transaction_from_csv_row(row: dict[str, str], fallback_merchant: str | None) -> TransactionIn:
    normalized = {
        (key or "").strip().lower(): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }

    merchant_id = normalized.get("merchant_id") or fallback_merchant
    if not merchant_id:
        raise ValueError("merchant_id missing (or pass merchant_id query parameter)")

    payer_id = normalized.get("payer_id")
    if not payer_id:
        raise ValueError("payer_id is required")

    device_id = normalized.get("device_id")
    if not device_id:
        raise ValueError("device_id is required")

    amount_text = normalized.get("amount")
    if not amount_text:
        raise ValueError("amount is required")

    lat_text = normalized.get("lat")
    if not lat_text:
        raise ValueError("lat is required")

    lon_text = normalized.get("lon")
    if not lon_text:
        raise ValueError("lon is required")

    account_age_text = normalized.get("payer_account_age_days") or normalized.get("account_age_days")
    if not account_age_text:
        raise ValueError("payer_account_age_days is required")

    currency = normalized.get("currency") or "INR"
    timestamp_value = normalized.get("timestamp")

    timestamp = datetime.now(timezone.utc)
    if timestamp_value:
        timestamp = _normalize_timestamp(timestamp_value)

    return TransactionIn(
        merchant_id=merchant_id,
        payer_id=payer_id,
        amount=float(amount_text),
        currency=currency,
        device_id=device_id,
        lat=float(lat_text),
        lon=float(lon_text),
        payer_account_age_days=int(float(account_age_text)),
        timestamp=timestamp,
    )


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _normalize_timestamp(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _generate_synthetic_payload(suspicious_ratio: float, merchant_id: str | None = None) -> TransactionIn:
    merchant = _pick_merchant_profile(merchant_id)
    suspicious = random.random() < suspicious_ratio

    payer_id = f"PAYER-{random.randint(1, 180):04d}"
    payer_account_age_days = random.randint(2, 20) if suspicious else random.randint(45, 1200)

    base_amount = max(40, random.gauss(merchant["mean"], merchant["mean"] * 0.35))
    amount = base_amount
    if suspicious and random.random() < 0.65:
        amount *= random.uniform(2.8, 7.2)

    timestamp = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 24 * 60))

    device_prefix = "DEV"
    if suspicious and random.random() < 0.4:
        device_prefix = "NEWDEV"

    lat_jitter = random.uniform(-0.03, 0.03)
    lon_jitter = random.uniform(-0.03, 0.03)

    if suspicious and random.random() < 0.35:
        lat_jitter += random.choice([-1.8, 2.3, 3.1, -2.5])
        lon_jitter += random.choice([1.6, -2.7, 3.4, -1.9])

    return TransactionIn(
        merchant_id=merchant["id"],
        payer_id=payer_id,
        amount=round(amount, 2),
        currency="INR",
        device_id=f"{device_prefix}-{random.randint(1, 900):04d}",
        lat=merchant["lat"] + lat_jitter,
        lon=merchant["lon"] + lon_jitter,
        payer_account_age_days=payer_account_age_days,
        timestamp=timestamp,
    )


def _pick_merchant_profile(merchant_id: str | None) -> dict[str, Any]:
    if merchant_id:
        for profile in MERCHANT_PROFILES:
            if profile["id"] == merchant_id:
                return profile

    return random.choice(MERCHANT_PROFILES)


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None

    text = value.strip()
    return text or None
