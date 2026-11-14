"""Original synthetic events; no organizer inputs or hidden-label reconstruction."""
import numpy as np
from .model import LinearChoiceModel
from .probabilities import choice_probabilities, correct_event_mass
from .weekend import weekend_correction


def generate(events, seed):
    rng = np.random.default_rng(seed)
    # Each event has independent toy preference/history summaries. There are
    # no repeated users or dates, so this cannot test temporal leakage.
    preference = rng.normal(size=(events, 1, 4))
    items = rng.normal(size=(events, 10, 4))
    history = preference + rng.normal(scale=0.8, size=preference.shape)
    affinity = (preference * items).sum(2) / 2
    history_affinity = (history * items).sum(2) / 2
    popularity = rng.normal(size=(events, 10))
    propensity = np.broadcast_to(rng.normal(size=(events, 1)), (events, 10))
    x = np.stack([np.ones_like(affinity), affinity, history_affinity, popularity, propensity], axis=2)
    truth = choice_probabilities(x @ np.array([-2.2, 0.9, 0.3, 0.35, 0.55]))
    target = (rng.random((events, 1)) > np.cumsum(truth, axis=1)).sum(1)
    return x, np.minimum(target, 10)


def auc(labels, scores):
    """Global binary ROC AUC using average ranks for ties."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AUC expects aligned finite scores and binary labels")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both label classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def run():
    train, target = generate(2500, 142)
    test, truth = generate(1000, 143)
    model = LinearChoiceModel.fit(train, target)
    p = model.predict(test)
    labels = np.eye(11, dtype=int)[truth, :10].ravel()
    baseline = np.full_like(p, 1 / 11)
    weekend = np.arange(len(p)) % 2
    corrected, _ = weekend_correction(p[:, :10], test[:, :, 1:3], weekend, np.zeros(2), np.ones(2), np.array([0.05, -0.03]))
    event = correct_event_mass(p[:, :10], np.full(len(p), 0.1))
    return {
        "scope": "Original synthetic demonstration, not competition performance",
        "training_events": len(train), "evaluation_events": len(test),
        "uniform_event_log_loss": float(-np.log(baseline[np.arange(len(truth)), truth]).mean()),
        "trained_event_log_loss": float(-np.log(p[np.arange(len(truth)), truth]).mean()),
        "trained_global_candidate_auc": auc(labels, p[:, :10].ravel()),
        "choice_correction_max_event_mass_error": float(abs(corrected.sum(1) - p[:, :10].sum(1)).max()),
        "event_shift_mean_click_probability_change": float((event.sum(1) - p[:, :10].sum(1)).mean()),
    }
