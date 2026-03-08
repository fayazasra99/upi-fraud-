# UPI Merchant Fraud Shield

A full MVP for merchant fraud prevention with:

- token-based login and role-aware access control
- real-time transaction scoring (`allow` / `review` / `block`)
- case workflow statuses (`investigating`, `resolved_legit`, `confirmed_fraud`)
- merchant-scoped dashboards and analytics
- custom rule engine (global or merchant-specific)
- batch CSV upload for bulk scoring
- ML model blending with heuristic risk score
- retraining from confirmed case outcomes

## Demo Credentials

- `admin / admin123` (analyst)
- `ops_lead / ops123` (analyst)
- `electro_mgr / electro123` (merchant admin, `MRT-ELECTRO-01`)
- `grocery_mgr / grocery123` (merchant admin, `MRT-GROCERY-07`)

## Project Structure

- `backend/app/main.py` - APIs, auth usage, scope enforcement, CSV ingest
- `backend/app/auth.py` - login + token creation/verification + role checks
- `backend/app/db.py` - SQLite schema, queries, metrics, case labels
- `backend/app/risk_engine.py` - explainable feature extraction + hybrid scoring
- `backend/app/ml_model.py` - model loading and probability inference
- `backend/app/model_training.py` - retraining from confirmed cases
- `frontend/index.html` - dashboard UI
- `frontend/styles.css` - responsive styling
- `frontend/app.js` - login flow + dashboard interactions
- `scripts/train_model.py` - synthetic dataset generation + baseline model training
- `scripts/smoke_test.py` - end-to-end API smoke test

## Local Run

1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Optional: train baseline model artifact

```powershell
python scripts\train_model.py --samples 6000
```

4. Start API server

```powershell
uvicorn backend.app.main:app --reload
```

5. Open:

- Dashboard: `http://127.0.0.1:8000/`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## One-Command Docker Deployment

```powershell
docker compose up --build
```

Open `http://127.0.0.1:8000/`.

Persistent data paths:

- SQLite DB: `./data/fraud_shield.db`
- Model artifacts: `./artifacts/`

## CSV Upload Format

Use headers (in any order):

`merchant_id,payer_id,amount,currency,device_id,lat,lon,payer_account_age_days,timestamp`

Notes:

- `currency` and `timestamp` are optional.
- If `merchant_id` is omitted in file rows, pass merchant via dashboard filter or API query.
- Timestamp format: ISO-8601 (`2026-03-07T11:00:00Z`).

## Retraining From Confirmed Cases

Dashboard analyst action: `Train From Cases` button.

API:

`POST /api/model/train-from-cases`

Example body:

```json
{
  "merchant_id": "MRT-ELECTRO-01",
  "min_samples": 300,
  "max_cases": 20000,
  "allow_synthetic": true
}
```

Labels used:

- `confirmed_fraud` => fraud
- `resolved_legit` => legit

If labeled cases are insufficient and `allow_synthetic=true`, synthetic samples are added to reach `min_samples`.

## Verification

Run smoke test:

```powershell
python scripts\smoke_test.py
```
