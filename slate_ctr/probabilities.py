"""Separate event click propensity from conditional candidate choice."""
import numpy as np


def matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must be a nonempty event-by-candidate matrix")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def probabilities(value):
    value = matrix(value, "probabilities")
    if (value < 0).any() or (value > 1).any() or (value.sum(1) > 1 + 1e-12).any():
        raise ValueError("candidate probabilities must be nonnegative and sum to at most one per event")
    return value


def choice_probabilities(utilities):
    """Softmax over K candidate utilities and an outside/no-click utility of zero.

    Returns all K+1 outcomes; the final column is no click. Utilities are
    relative to the outside option, so adding a common candidate offset
    changes the probability of any click.
    """
    utilities = matrix(utilities, "utilities")
    logits = np.column_stack((utilities, np.zeros(len(utilities))))
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def correct_event_mass(base, logit_delta):
    """Shift log-odds of any click while preserving conditional item choice.

    Zero and unit event masses stay at their boundary values. Zero-mass
    events cannot acquire an item distribution from this operation alone.
    """
    base = probabilities(base)
    delta = np.asarray(logit_delta, dtype=np.float64)
    if delta.shape != (len(base),) or not np.isfinite(delta).all():
        raise ValueError("logit_delta must contain one finite value per event")
    mass = np.minimum(base.sum(1), 1.0)
    interior = (mass > 0) & (mass < 1)
    updated = mass.copy()
    logits = np.log(mass[interior]) - np.log1p(-mass[interior]) + delta[interior]
    updated[interior] = np.exp(-np.logaddexp(0, -logits))
    scale = np.divide(updated, base.sum(1), out=np.zeros_like(mass), where=base.sum(1) > 0)
    return base * scale[:, None]


def correct_choices(base, delta, strength=0.5):
    """Reweight conditional choice, retaining each event's original click mass.

    Structural zero probabilities remain zero. No arbitrary epsilon is added.
    This is the probability transfer used by V142, generalized to K candidates.
    """
    base = probabilities(base)
    delta = matrix(delta, "delta")
    if delta.shape != base.shape or not np.isfinite(strength):
        raise ValueError("delta must match base shape and strength must be finite")
    if strength == 0:
        return base.copy()
    mass = base.sum(axis=1, keepdims=True)
    active = mass[:, 0] > 0
    result = np.zeros_like(base)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        logits = np.log(base[active]) + strength * delta[active]
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("scaled delta exceeds the supported numeric range")
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    result[active] = weights / weights.sum(axis=1, keepdims=True) * mass[active]
    return result
