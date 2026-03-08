from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.risk_engine import FEATURE_ORDER


@dataclass
class Sample:
    features: dict[str, float]
    label: int


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def make_sample() -> Sample:
    velocity_10m = random.randint(0, 16)
    amount_spike_ratio = round(random.uniform(0.7, 8.5), 3)
    new_device = 1 if random.random() < 0.22 else 0
    geo_distance_km = round(random.uniform(0, 550), 2)
    geo_mismatch = 1 if geo_distance_km > 25 else 0
    account_age_days = random.randint(1, 2000)
    night_hours = 1 if random.random() < 0.28 else 0
    amount = round(random.uniform(30, 15000), 2)

    z = (
        -4.9
        + 0.26 * velocity_10m
        + 1.35 * max(amount_spike_ratio - 1.2, 0)
        + 1.4 * new_device
        + 0.017 * geo_distance_km
        + 0.75 * geo_mismatch
        + 0.85 * night_hours
        + 0.00011 * amount
        - 0.0028 * account_age_days
    )

    probability = max(0.001, min(sigmoid(z), 0.995))
    label = 1 if random.random() < probability else 0

    return Sample(
        features={
            "velocity_10m": float(velocity_10m),
            "amount_spike_ratio": float(amount_spike_ratio),
            "new_device": float(new_device),
            "geo_distance_km": float(geo_distance_km),
            "geo_mismatch": float(geo_mismatch),
            "account_age_days": float(account_age_days),
            "night_hours": float(night_hours),
            "amount": float(amount),
        },
        label=label,
    )


def build_dataset(n_samples: int) -> tuple[list[list[float]], list[int]]:
    vectors: list[list[float]] = []
    labels: list[int] = []

    for _ in range(n_samples):
        sample = make_sample()
        vectors.append([sample.features[name] for name in FEATURE_ORDER])
        labels.append(sample.label)

    return vectors, labels


def train_model(n_samples: int) -> dict:
    x, y = build_dataset(n_samples)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 1]
    pred = [1 if p >= 0.5 else 0 for p in prob]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "positive_rate": round(float(sum(y) / len(y)), 4),
        "samples": n_samples,
    }

    return {
        "model": model,
        "feature_order": FEATURE_ORDER,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fraud Shield demo model")
    parser.add_argument("--samples", type=int, default=6000, help="Number of synthetic training samples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "fraud_model.pkl",
        help="Path to output model file",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts") / "model_metrics.json",
        help="Path to output metrics JSON",
    )

    args = parser.parse_args()

    artifact = train_model(args.samples)
    trained_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "model": artifact["model"],
        "feature_order": artifact["feature_order"],
        "metrics": artifact["metrics"],
        "trained_at": trained_at,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("wb") as model_file:
        pickle.dump(payload, model_file)

    with args.metrics.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload["metrics"], metrics_file, indent=2)

    print("Model trained successfully")
    print(f"Model file: {args.output}")
    print(f"Metrics file: {args.metrics}")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
