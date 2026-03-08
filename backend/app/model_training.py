from __future__ import annotations

import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from .risk_engine import FEATURE_ORDER


def train_model_artifact_from_cases(
    *,
    cases: list[dict[str, Any]],
    model_path: Path,
    min_samples: int = 300,
    allow_synthetic: bool = True,
) -> dict[str, Any]:
    vectors: list[list[float]] = []
    labels: list[int] = []

    for case in cases:
        features = case.get("features")
        label = case.get("label")

        if not isinstance(features, dict):
            continue

        if label not in {0, 1}:
            continue

        vectors.append(_vectorize(features))
        labels.append(int(label))

    confirmed_cases_used = len(vectors)
    synthetic_added = 0

    if (len(vectors) < min_samples or len(set(labels)) < 2) and allow_synthetic:
        target_total = max(min_samples, len(vectors))
        missing = target_total - len(vectors)

        if len(set(labels)) < 2:
            missing = max(missing, 80)

        synth_vectors, synth_labels = _generate_synthetic_dataset(missing, labels)
        vectors.extend(synth_vectors)
        labels.extend(synth_labels)
        synthetic_added = len(synth_vectors)

    if len(vectors) < 40:
        raise ValueError("Not enough labeled data to train. Add more confirmed cases.")

    if len(set(labels)) < 2:
        raise ValueError("Training requires both fraud and legit labeled cases.")

    test_size = 0.2 if len(vectors) >= 200 else 0.25

    x_train, x_test, y_train, y_test = train_test_split(
        vectors,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = [1 if value >= 0.5 else 0 for value in probabilities]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "positive_rate": round(float(sum(labels) / len(labels)), 4),
        "samples": len(labels),
    }

    trained_at = datetime.now(timezone.utc)

    artifact_payload = {
        "model": model,
        "feature_order": FEATURE_ORDER,
        "metrics": metrics,
        "trained_at": trained_at.isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as model_file:
        pickle.dump(artifact_payload, model_file)

    metrics_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return {
        "model_path": str(model_path),
        "trained_at": trained_at,
        "confirmed_cases_used": confirmed_cases_used,
        "synthetic_added": synthetic_added,
        "trained_samples": len(labels),
        "metrics": metrics,
    }


def _vectorize(features: dict[str, Any]) -> list[float]:
    vector: list[float] = []

    for name in FEATURE_ORDER:
        value = features.get(name, 0)
        if isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
        else:
            vector.append(float(value or 0.0))

    return vector


def _generate_synthetic_dataset(count: int, existing_labels: list[int]) -> tuple[list[list[float]], list[int]]:
    if count <= 0:
        return [], []

    positives = sum(existing_labels)
    total = len(existing_labels)
    positive_rate = (positives / total) if total else 0.5

    positive_rate = min(max(positive_rate, 0.25), 0.75)

    vectors: list[list[float]] = []
    labels: list[int] = []

    for _ in range(count):
        label = 1 if random.random() < positive_rate else 0
        labels.append(label)
        vectors.append(_synthetic_vector_for_label(label))

    combined = existing_labels + labels
    if len(set(combined)) < 2:
        if len(labels) >= 2:
            labels[0] = 0
            labels[1] = 1
            vectors[0] = _synthetic_vector_for_label(0)
            vectors[1] = _synthetic_vector_for_label(1)
        elif len(labels) == 1:
            labels[0] = 1 - labels[0]
            vectors[0] = _synthetic_vector_for_label(labels[0])

    return vectors, labels


def _synthetic_vector_for_label(label: int) -> list[float]:
    if label == 1:
        feature_map = {
            "velocity_10m": random.uniform(5, 16),
            "amount_spike_ratio": random.uniform(1.5, 8.0),
            "new_device": random.choice([0.0, 1.0, 1.0]),
            "geo_distance_km": random.uniform(30, 600),
            "geo_mismatch": random.choice([0.0, 1.0, 1.0]),
            "account_age_days": random.uniform(1, 120),
            "night_hours": random.choice([0.0, 1.0, 1.0]),
            "amount": random.uniform(1500, 25000),
        }
    else:
        feature_map = {
            "velocity_10m": random.uniform(0, 5),
            "amount_spike_ratio": random.uniform(0.6, 1.8),
            "new_device": random.choice([0.0, 0.0, 1.0]),
            "geo_distance_km": random.uniform(0, 45),
            "geo_mismatch": random.choice([0.0, 0.0, 1.0]),
            "account_age_days": random.uniform(60, 2500),
            "night_hours": random.choice([0.0, 0.0, 1.0]),
            "amount": random.uniform(30, 6000),
        }

    return [float(feature_map[name]) for name in FEATURE_ORDER]

