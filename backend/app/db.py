from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL,
                    payer_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    payer_account_age_days INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    txn_epoch INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    heuristic_score REAL,
                    model_probability REAL,
                    reasons_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    case_note TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_transactions_merchant_epoch
                    ON transactions (merchant_id, txn_epoch);

                CREATE INDEX IF NOT EXISTS idx_transactions_payer_epoch
                    ON transactions (payer_id, txn_epoch);

                CREATE INDEX IF NOT EXISTS idx_transactions_decision
                    ON transactions (decision);

                CREATE INDEX IF NOT EXISTS idx_transactions_status
                    ON transactions (status);

                CREATE TABLE IF NOT EXISTS rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    merchant_id TEXT,
                    action TEXT NOT NULL,
                    condition_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rules_merchant_enabled
                    ON rules (merchant_id, enabled);
                """
            )

            self._ensure_column(conn, "transactions", "heuristic_score", "REAL")
            self._ensure_column(conn, "transactions", "model_probability", "REAL")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        column_names = {row["name"] for row in columns}

        if column_name in column_names:
            return

        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def insert_transaction(self, record: dict[str, Any]) -> None:
        timestamp = record["timestamp"]
        if isinstance(timestamp, datetime):
            timestamp = timestamp.astimezone(timezone.utc).isoformat()

        reasons_json = json.dumps(record.get("reasons", []), separators=(",", ":"))
        features_json = json.dumps(record.get("features", {}), separators=(",", ":"))

        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transactions (
                    id, merchant_id, payer_id, amount, currency, device_id,
                    lat, lon, payer_account_age_days, timestamp, txn_epoch,
                    decision, status, risk_score, heuristic_score, model_probability,
                    reasons_json, features_json, case_note, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["merchant_id"],
                    record["payer_id"],
                    float(record["amount"]),
                    record["currency"],
                    record["device_id"],
                    float(record["lat"]),
                    float(record["lon"]),
                    int(record["payer_account_age_days"]),
                    timestamp,
                    int(record["txn_epoch"]),
                    record["decision"],
                    record["status"],
                    float(record["risk_score"]),
                    _to_optional_float(record.get("heuristic_score")),
                    _to_optional_float(record.get("model_probability")),
                    reasons_json,
                    features_json,
                    record.get("case_note"),
                    created_at,
                ),
            )

    def list_transactions(
        self,
        *,
        limit: int = 50,
        decision: str | None = None,
        status: str | None = None,
        merchant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []

        if decision:
            where_clauses.append("decision = ?")
            params.append(decision)

        if status:
            where_clauses.append("status = ?")
            params.append(status)

        if merchant_id:
            where_clauses.append("merchant_id = ?")
            params.append(merchant_id)

        query = "SELECT * FROM transactions"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY txn_epoch DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_transaction(row) for row in rows]

    def list_review_queue(
        self,
        *,
        limit: int = 50,
        merchant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        query = (
            "SELECT * FROM transactions "
            "WHERE status IN ('review', 'block', 'investigating')"
        )

        if merchant_id:
            query += " AND merchant_id = ?"
            params.append(merchant_id)

        query += " ORDER BY txn_epoch DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_transaction(row) for row in rows]

    def list_labeled_cases(
        self,
        *,
        merchant_id: str | None = None,
        limit: int = 20000,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        query = (
            "SELECT id, merchant_id, status, features_json "
            "FROM transactions "
            "WHERE status IN ('confirmed_fraud', 'resolved_legit')"
        )

        if merchant_id:
            query += " AND merchant_id = ?"
            params.append(merchant_id)

        query += " ORDER BY txn_epoch DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        cases: list[dict[str, Any]] = []
        for row in rows:
            status = str(row["status"])
            label = 1 if status == "confirmed_fraud" else 0
            features = json.loads(row["features_json"] or "{}")
            if not isinstance(features, dict):
                continue

            cases.append(
                {
                    "id": row["id"],
                    "merchant_id": row["merchant_id"],
                    "status": status,
                    "label": label,
                    "features": features,
                }
            )

        return cases

    def get_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM transactions WHERE id = ? LIMIT 1", (transaction_id,)
            ).fetchone()

        if not row:
            return None

        return self._row_to_transaction(row)

    def add_case_note(self, transaction_id: str, note: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE transactions SET case_note = ? WHERE id = ?",
                (note, transaction_id),
            )

        return cursor.rowcount > 0

    def set_transaction_status(self, transaction_id: str, status: str, note: str | None = None) -> bool:
        with self._connect() as conn:
            if note:
                cursor = conn.execute(
                    "UPDATE transactions SET status = ?, case_note = ? WHERE id = ?",
                    (status, note, transaction_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE transactions SET status = ? WHERE id = ?",
                    (status, transaction_id),
                )

        return cursor.rowcount > 0

    def get_velocity_10m(self, merchant_id: str, current_epoch: int) -> int:
        lower_bound = current_epoch - 600
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM transactions
                WHERE merchant_id = ?
                  AND txn_epoch >= ?
                  AND txn_epoch <= ?
                """,
                (merchant_id, lower_bound, current_epoch),
            ).fetchone()

        return int(row["total"] if row else 0)

    def get_merchant_average_amount(self, merchant_id: str, current_epoch: int) -> float | None:
        lower_bound = current_epoch - (30 * 24 * 60 * 60)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT AVG(amount) AS avg_amount
                FROM transactions
                WHERE merchant_id = ?
                  AND txn_epoch >= ?
                  AND txn_epoch <= ?
                """,
                (merchant_id, lower_bound, current_epoch),
            ).fetchone()

        if not row or row["avg_amount"] is None:
            return None

        return float(row["avg_amount"])

    def get_payer_last_transaction(self, payer_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM transactions
                WHERE payer_id = ?
                ORDER BY txn_epoch DESC
                LIMIT 1
                """,
                (payer_id,),
            ).fetchone()

        if not row:
            return None

        return self._row_to_transaction(row)

    def create_rule(self, payload: dict[str, Any]) -> dict[str, Any]:
        rule_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        condition_json = json.dumps(payload["condition"], separators=(",", ":"))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rules (id, name, merchant_id, action, condition_json, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule_id,
                    payload["name"],
                    payload.get("merchant_id") or None,
                    payload["action"],
                    condition_json,
                    1 if payload.get("enabled", True) else 0,
                    created_at,
                ),
            )

        return {
            "id": rule_id,
            "name": payload["name"],
            "merchant_id": payload.get("merchant_id") or None,
            "action": payload["action"],
            "condition": payload["condition"],
            "enabled": bool(payload.get("enabled", True)),
            "created_at": created_at,
        }

    def list_rules(self, merchant_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM rules"
        params: list[Any] = []

        if merchant_id:
            query += " WHERE merchant_id = ? OR merchant_id IS NULL"
            params.append(merchant_id)

        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_rule(row) for row in rows]

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM rules WHERE id = ? LIMIT 1", (rule_id,)).fetchone()

        if not row:
            return None

        return self._row_to_rule(row)

    def list_applicable_rules(self, merchant_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM rules
                WHERE enabled = 1
                  AND (merchant_id = ? OR merchant_id IS NULL)
                ORDER BY created_at DESC
                """,
                (merchant_id,),
            ).fetchall()

        return [self._row_to_rule(row) for row in rows]

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute("UPDATE rules SET enabled = ? WHERE id = ?", (1 if enabled else 0, rule_id))
            row = conn.execute("SELECT * FROM rules WHERE id = ? LIMIT 1", (rule_id,)).fetchone()

        if not row:
            return None

        return self._row_to_rule(row)

    def metrics_24h(self, merchant_id: str | None = None) -> dict[str, Any]:
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        lower_bound = now_epoch - (24 * 60 * 60)
        params: list[Any] = [lower_bound]

        query = (
            "SELECT "
            "COUNT(*) AS total_24h, "
            "SUM(CASE WHEN decision = 'allow' THEN 1 ELSE 0 END) AS allow_24h, "
            "SUM(CASE WHEN decision = 'review' THEN 1 ELSE 0 END) AS review_24h, "
            "SUM(CASE WHEN decision = 'block' THEN 1 ELSE 0 END) AS block_24h, "
            "AVG(risk_score) AS avg_risk_score_24h "
            "FROM transactions WHERE txn_epoch >= ?"
        )

        if merchant_id:
            query += " AND merchant_id = ?"
            params.append(merchant_id)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        return {
            "merchant_id": merchant_id,
            "total_24h": int(row["total_24h"] or 0),
            "allow_24h": int(row["allow_24h"] or 0),
            "review_24h": int(row["review_24h"] or 0),
            "block_24h": int(row["block_24h"] or 0),
            "avg_risk_score_24h": round(float(row["avg_risk_score_24h"] or 0.0), 2),
        }

    def merchant_summary_24h(self) -> list[dict[str, Any]]:
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        lower_bound = now_epoch - (24 * 60 * 60)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    merchant_id,
                    COUNT(*) AS total_24h,
                    SUM(CASE WHEN decision = 'allow' THEN 1 ELSE 0 END) AS allow_24h,
                    SUM(CASE WHEN decision = 'review' THEN 1 ELSE 0 END) AS review_24h,
                    SUM(CASE WHEN decision = 'block' THEN 1 ELSE 0 END) AS block_24h,
                    AVG(risk_score) AS avg_risk_score_24h
                FROM transactions
                WHERE txn_epoch >= ?
                GROUP BY merchant_id
                ORDER BY total_24h DESC, block_24h DESC
                """,
                (lower_bound,),
            ).fetchall()

        summary: list[dict[str, Any]] = []
        for row in rows:
            total = int(row["total_24h"] or 0)
            review = int(row["review_24h"] or 0)
            block = int(row["block_24h"] or 0)

            summary.append(
                {
                    "merchant_id": row["merchant_id"],
                    "total_24h": total,
                    "allow_24h": int(row["allow_24h"] or 0),
                    "review_24h": review,
                    "block_24h": block,
                    "avg_risk_score_24h": round(float(row["avg_risk_score_24h"] or 0.0), 2),
                    "block_rate_pct": round((block / total) * 100, 2) if total else 0.0,
                    "review_rate_pct": round((review / total) * 100, 2) if total else 0.0,
                }
            )

        return summary

    def list_merchants(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT merchant_id FROM transactions ORDER BY merchant_id ASC"
            ).fetchall()

        return [row["merchant_id"] for row in rows if row["merchant_id"]]

    def _row_to_transaction(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "merchant_id": row["merchant_id"],
            "payer_id": row["payer_id"],
            "amount": float(row["amount"]),
            "currency": row["currency"],
            "device_id": row["device_id"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "payer_account_age_days": int(row["payer_account_age_days"]),
            "timestamp": row["timestamp"],
            "txn_epoch": int(row["txn_epoch"]),
            "decision": row["decision"],
            "status": row["status"],
            "risk_score": round(float(row["risk_score"]), 2),
            "heuristic_score": _to_optional_float(row["heuristic_score"]),
            "model_probability": _to_optional_float(row["model_probability"]),
            "reasons": json.loads(row["reasons_json"]),
            "features": json.loads(row["features_json"]),
            "case_note": row["case_note"],
        }

    def _row_to_rule(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "merchant_id": row["merchant_id"],
            "action": row["action"],
            "condition": json.loads(row["condition_json"]),
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
        }


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return round(float(value), 4)
