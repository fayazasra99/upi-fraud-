from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

FEATURE_ORDER = [
    "velocity_10m",
    "amount_spike_ratio",
    "new_device",
    "geo_distance_km",
    "geo_mismatch",
    "account_age_days",
    "night_hours",
    "amount",
]


def build_features(
    payload: dict[str, Any],
    velocity_10m: int,
    merchant_avg_amount: float | None,
    payer_last_transaction: dict[str, Any] | None,
) -> dict[str, Any]:
    amount = float(payload["amount"])
    account_age_days = int(payload["payer_account_age_days"])
    timestamp = _normalize_timestamp(payload["timestamp"])
    hour = timestamp.hour

    amount_spike = 1.0
    if merchant_avg_amount and merchant_avg_amount > 0:
        amount_spike = amount / merchant_avg_amount

    new_device = False
    geo_distance_km = 0.0
    if payer_last_transaction:
        new_device = payer_last_transaction["device_id"] != payload["device_id"]
        geo_distance_km = _haversine_km(
            float(payer_last_transaction["lat"]),
            float(payer_last_transaction["lon"]),
            float(payload["lat"]),
            float(payload["lon"]),
        )

    geo_mismatch = geo_distance_km > 25.0
    night_hours = hour in {0, 1, 2, 3, 4, 5, 23}

    return {
        "velocity_10m": int(velocity_10m),
        "merchant_avg_amount": round(float(merchant_avg_amount or 0.0), 2),
        "amount_spike_ratio": round(float(amount_spike), 4),
        "new_device": bool(new_device),
        "geo_distance_km": round(float(geo_distance_km), 4),
        "geo_mismatch": bool(geo_mismatch),
        "account_age_days": int(account_age_days),
        "night_hours": bool(night_hours),
        "hour_of_day": int(hour),
        "amount": round(amount, 2),
    }


def feature_vector(features: dict[str, Any], feature_order: list[str] | None = None) -> list[float]:
    order = feature_order or FEATURE_ORDER
    vector: list[float] = []

    for name in order:
        value = features.get(name, 0)
        if isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
            continue

        vector.append(float(value or 0.0))

    return vector


def evaluate_transaction(
    payload: dict[str, Any],
    *,
    features: dict[str, Any],
    rules: list[dict[str, Any]],
    review_threshold: float,
    block_threshold: float,
    model_probability: float | None = None,
    model_weight: float = 0.30,
) -> dict[str, Any]:
    heuristic_score, reasons = _heuristic_score(features)

    score = heuristic_score
    if model_probability is not None:
        bounded_model_prob = max(0.0, min(float(model_probability), 1.0))
        model_score = bounded_model_prob * 100.0
        weight = max(0.0, min(model_weight, 1.0))
        score = (1.0 - weight) * heuristic_score + weight * model_score
        reasons.append(f"ML model fraud probability is {bounded_model_prob:.2%}.")

    score, rule_reasons = _apply_rules(score, payload, features, rules)
    reasons.extend(rule_reasons)

    score = round(max(0.0, min(score, 100.0)), 2)
    heuristic_score = round(max(0.0, min(heuristic_score, 100.0)), 2)

    decision = _decision_from_score(score, review_threshold, block_threshold)

    if not reasons:
        reasons.append("No strong fraud signals detected.")

    return {
        "risk_score": score,
        "heuristic_score": heuristic_score,
        "model_probability": None if model_probability is None else round(float(model_probability), 4),
        "decision": decision,
        "reasons": reasons,
        "features": features,
    }


def _heuristic_score(features: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    velocity_10m = int(features["velocity_10m"])
    velocity_component = min(max(velocity_10m - 2, 0) / 8, 1) * 22
    score += velocity_component
    if velocity_10m >= 6:
        reasons.append(f"High transaction velocity detected ({velocity_10m} in 10 minutes).")

    amount_spike = float(features["amount_spike_ratio"])
    if amount_spike > 1.3:
        amount_component = min((amount_spike - 1.3) / 2.7, 1) * 24
        score += amount_component
        reasons.append(f"Amount spike is {amount_spike:.2f}x above merchant baseline.")

    if bool(features["new_device"]):
        score += 14
        reasons.append("Payer switched to an unseen device.")

    if bool(features["geo_mismatch"]):
        geo_distance_km = float(features["geo_distance_km"])
        geo_component = min((geo_distance_km - 25) / 100, 1) * 14
        score += max(8, geo_component)
        reasons.append(f"Geo mismatch detected ({geo_distance_km:.1f} km from last location).")

    account_age_days = int(features["account_age_days"])
    if account_age_days < 7:
        score += 18
        reasons.append("Very new payer account.")
    elif account_age_days < 30:
        score += 10
        reasons.append("Recently created payer account.")

    if bool(features["night_hours"]):
        score += 10
        reasons.append("Transaction placed during high-risk night window.")

    return score, reasons


def _apply_rules(
    score: float,
    payload: dict[str, Any],
    features: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    reasons: list[str] = []

    for rule in rules:
        condition = rule["condition"]
        if not _rule_matches(condition, payload, features):
            continue

        reasons.append(f"Custom rule matched: {rule['name']} ({rule['action']}).")
        action = rule["action"]

        if action == "block":
            score = max(score, 85)
        elif action == "review":
            score = max(score, 60)
        elif action == "allow":
            score = min(score, 35)

    return score, reasons


def _rule_matches(condition: dict[str, Any], payload: dict[str, Any], features: dict[str, Any]) -> bool:
    amount = float(payload["amount"])

    min_amount = condition.get("min_amount")
    if min_amount is not None and amount < float(min_amount):
        return False

    max_amount = condition.get("max_amount")
    if max_amount is not None and amount > float(max_amount):
        return False

    velocity_10m_gt = condition.get("velocity_10m_gt")
    if velocity_10m_gt is not None and float(features["velocity_10m"]) <= float(velocity_10m_gt):
        return False

    amount_spike_gt = condition.get("amount_spike_gt")
    if amount_spike_gt is not None and float(features["amount_spike_ratio"]) <= float(amount_spike_gt):
        return False

    if condition.get("require_new_device") and not bool(features["new_device"]):
        return False

    geo_distance_km_gt = condition.get("geo_distance_km_gt")
    if geo_distance_km_gt is not None and float(features["geo_distance_km"]) <= float(geo_distance_km_gt):
        return False

    account_age_days_lt = condition.get("account_age_days_lt")
    if account_age_days_lt is not None and int(features["account_age_days"]) >= int(account_age_days_lt):
        return False

    if condition.get("night_hours_only") and not bool(features["night_hours"]):
        return False

    return True


def decision_to_initial_status(decision: str) -> str:
    if decision in {"allow", "review", "block"}:
        return decision
    return "review"


def _decision_from_score(score: float, review_threshold: float, block_threshold: float) -> str:
    if score >= block_threshold:
        return "block"
    if score >= review_threshold:
        return "review"
    return "allow"


def _normalize_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_km * c
