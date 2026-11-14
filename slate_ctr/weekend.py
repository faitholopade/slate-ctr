"""Portable extraction of V142's bounded weekend choice correction."""
import numpy as np
from .probabilities import matrix, correct_choices


def weekend_correction(base, candidate_features, weekend, mean, std, weight, strength=0.5):
    """Apply the V142 correction to caller-supplied features and parameters.

    candidate_features: (events, candidates, features), before standardization.
    weekend: one binary indicator per event. Training-only mean/std and
    learned weights must be supplied by the caller; no weights are bundled.
    Returns (corrected candidate probabilities, bounded utility corrections).
    """
    base = matrix(base, "base")
    features = np.asarray(candidate_features, dtype=np.float64)
    if features.ndim != 3 or features.shape[:2] != base.shape or not features.shape[2]:
        raise ValueError("features must have shape (events, candidates, features)")
    dim = features.shape[2]
    mean, std, weight = [np.asarray(v, dtype=np.float64) for v in (mean, std, weight)]
    if any(v.shape != (dim,) for v in (mean, std, weight)):
        raise ValueError("mean, std and weight must match feature count")
    if not all(np.isfinite(v).all() for v in (features, mean, std, weight)) or (std <= 0).any():
        raise ValueError("features/parameters must be finite and std strictly positive")
    weekend = np.asarray(weekend)
    if weekend.shape != (len(base),) or not np.isin(weekend, [0, 1]).all():
        raise ValueError("weekend must contain one binary indicator per event")
    x = np.clip((features - mean) / std, -6, 6)
    x -= x.mean(axis=1, keepdims=True)
    linear = np.einsum("nkd,d->nk", x, weight) * (weekend - 2 / 7)[:, None]
    delta = 0.2 * np.tanh(linear / 0.2)
    return correct_choices(base, delta, strength), delta
