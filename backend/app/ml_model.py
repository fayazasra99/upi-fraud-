from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .risk_engine import FEATURE_ORDER, feature_vector


class FraudModelService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.feature_order: list[str] = FEATURE_ORDER.copy()
        self.metrics: dict[str, Any] | None = None
        self.trained_at: datetime | None = None
        self.last_loaded_at: datetime | None = None
        self.load()

    def load(self) -> bool:
        if not self.model_path.exists():
            self.model = None
            self.last_loaded_at = datetime.now(timezone.utc)
            return False

        with self.model_path.open("rb") as file:
            artifact = pickle.load(file)

        if not isinstance(artifact, dict):
            raise ValueError("Model artifact must be a dictionary")

        self.model = artifact.get("model")
        if self.model is None:
            raise ValueError("Model artifact missing 'model'")

        order = artifact.get("feature_order")
        if isinstance(order, list) and order:
            self.feature_order = [str(name) for name in order]

        metrics = artifact.get("metrics")
        if isinstance(metrics, dict):
            self.metrics = metrics
        else:
            self.metrics = None

        trained_at = artifact.get("trained_at")
        if isinstance(trained_at, str):
            try:
                parsed = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                self.trained_at = parsed.astimezone(timezone.utc)
            except ValueError:
                self.trained_at = None
        else:
            self.trained_at = None

        self.last_loaded_at = datetime.now(timezone.utc)
        return True

    def predict_probability(self, features: dict[str, Any]) -> float | None:
        if self.model is None:
            return None

        vector = feature_vector(features, self.feature_order)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([vector])
            value = float(probabilities[0][1])
            return max(0.0, min(value, 1.0))

        if hasattr(self.model, "predict"):
            prediction = self.model.predict([vector])
            value = float(prediction[0])
            if value > 1.0:
                value = value / 100.0
            return max(0.0, min(value, 1.0))

        return None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "path": str(self.model_path),
            "feature_order": self.feature_order,
            "last_loaded_at": self.last_loaded_at,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }
