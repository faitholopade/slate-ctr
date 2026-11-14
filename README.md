# Slate CTR

A public companion to **P3 Labs' entry in Huawei Ireland Tech Arena 2026**, maintained by Faith Olopade. Our team qualified for the Top 15 on-site final; we did not attend Phase 2.

The challenge was to predict click-through probability for each item in a ten-candidate slate. An event could produce one click or no click, and each user came with up to 20 recent clicked items. Submissions were ranked by global ROC AUC across all candidate rows. That makes both personalization and the probability of any click important: choosing the best item within each slate is only part of the problem.

The historical solution combined latent preference and exposure representations, history features, calibration heads, and small corrections to event propensity and item choice. The best reported Phase 1 submission was **V142, with ROC AUC 0.832748** on 2 November 2026. This is a recorded portal result, not a score produced by this repository. The last supplied snapshot placed us 12th; it is not a final-rank claim.

This repository extracts the probability operations into a small NumPy package and supplies an original synthetic training example. The V142 weekend correction standardizes candidate features, centers them within each slate, applies a bounded context-dependent utility adjustment, and reweights candidates while preserving the probability of any click. An event-mass correction performs the complementary operation: it changes click propensity while preserving conditional item choice. The package also includes a linear choice model with an explicit no-click outcome, JSON model serialization, and tests.

From a checkout, use Python 3.12 or newer:

```bash
python -m pip install -e .
python -m slate_ctr demo
python -m unittest discover -s tests -v
```

The demo trains on 2,500 original synthetic events and evaluates on 1,000 independently generated events. It runs without downloads, private checkpoints, or organizer data. It demonstrates the interface and probability constraints; it is **not the V142 ensemble**, a reconstruction of the organizer's generator, or a predictor of competition performance. NumPy is the only runtime dependency.

```python
from slate_ctr import choice_probabilities, correct_event_mass, correct_choices

# Last column is the probability of no click.
outcomes = choice_probabilities([[0.2, -0.3, 0.8]])
base = outcomes[:, :-1]

# Change whether a click happens; keep the relative item probabilities.
event_adjusted = correct_event_mass(base, [0.1])

# Change which item is clicked; keep total click probability unchanged.
choice_adjusted = correct_choices(base, [[0.02, -0.03, 0.01]])
```

The main lesson was that small local gains did not reliably transfer to the leaderboard. V136 to V142 improved the reported score by only 0.000004. Reused validation cohorts, proxy baselines, and a growing chain of saved components made both selection and reproduction harder. A later recovery reproduced V65 saved-model inference within a fixed numerical tolerance and replayed V142's original CSV exactly using saved baseline predictions. **Neither result establishes full training reproduction**, and a larger historical benchmark discrepancy remains unresolved.

Read [the model notes](docs/model.md) for the probability decomposition, [evaluation](docs/evaluation.md) for measured results and their limits, and [reproduction](docs/reproduction.md) for what can and cannot be reproduced. [Sources and rights](docs/sources-and-rights.md) explains provenance and exclusions. Code and documentation were developed with AI assistance and checked against archived source and evidence.

This is a clean public companion with a fresh history. The private research archive, competition data, submission predictions, model weights, team discussions, and account details are not included.
