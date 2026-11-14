"""Small trainable demonstration, not the historical competition ensemble."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .probabilities import choice_probabilities


def features_array(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError("features must be a finite nonempty (events, candidates, features) array")
    return x


@dataclass
class LinearChoiceModel:
    """Shared linear utility trained by event cross-entropy with an outside option."""
    weight: np.ndarray

    @classmethod
    def fit(cls, features, targets, steps=400, learning_rate=0.1, l2=0.01):
        x = features_array(features)
        target = np.asarray(targets)
        n, k, dim = x.shape
        if target.shape != (n,) or target.dtype.kind not in "iu" or (target < 0).any() or (target > k).any():
            raise ValueError("targets must be integer candidate indices; K denotes no click")
        if not isinstance(steps, int) or steps <= 0 or not np.isfinite([learning_rate, l2]).all() or learning_rate <= 0 or l2 < 0:
            raise ValueError("steps and learning_rate must be positive; l2 must be nonnegative")
        weight = np.zeros(dim)
        observed = np.eye(k + 1)[target, :k]
        for _ in range(steps):
            p = choice_probabilities(x @ weight)[:, :k]
            gradient = np.einsum("nk,nkd->d", p - observed, x) / n + l2 * weight
            weight -= learning_rate * gradient
        return cls(weight)

    def predict(self, features):
        x = features_array(features)
        weight = np.asarray(self.weight, dtype=np.float64)
        if weight.shape != (x.shape[2],) or not np.isfinite(weight).all():
            raise ValueError("model weights must be finite and match feature count")
        return choice_probabilities(x @ weight)

    def save(self, path):
        weight = np.asarray(self.weight, dtype=np.float64)
        if weight.ndim != 1 or not len(weight) or not np.isfinite(weight).all():
            raise ValueError("invalid model weights")
        Path(path).write_text(json.dumps({"format": "linear-choice-v1", "weight": weight.tolist()}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("format") != "linear-choice-v1":
            raise ValueError("unsupported model format")
        weight = np.asarray(data["weight"], dtype=np.float64)
        if weight.ndim != 1 or not len(weight) or not np.isfinite(weight).all():
            raise ValueError("invalid model weights")
        return cls(weight)
