# Reproduction scope

A fresh public checkout can install the NumPy package, generate original
synthetic data, train and serialize the demonstration model, and run the tests.
It cannot regenerate the scored competition submissions.

## Public commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m slate_ctr demo
```

A model can be saved without executable pickle objects:

```python
from slate_ctr.demo import generate
from slate_ctr.model import LinearChoiceModel

features, targets = generate(1000, seed=7)
model = LinearChoiceModel.fit(features, targets)
model.save("model.json")
restored = LinearChoiceModel.load("model.json")
probabilities = restored.predict(features[:2])  # K candidates plus no click
```

Use an independent evaluation set to measure quality; this snippet only
illustrates serialization. The model does not store feature names or perform
normalization. Callers must preserve feature order and preprocessing themselves.

## Historical recovery

On 7 November, a separate private recovery restored the original archive inputs
and recomputed V65 on the fixed Phase 1 evaluation set. The original source used
exact intermediate CSV hashes, which failed in that environment. Separate
compatibility copies retained the model math and source/input checks, recorded
intermediate hashes, and required a final full-array comparison at a fixed 1e-7
absolute tolerance. Reference probabilities were read after inference, not used
to manufacture its output. Both recomputations produced identical new bytes.

The maximum absolute difference from the scored V65 output was
7.280000002829468e-08. The original CSV SHA-256 was
`d541041a61c559d5d3e36f4a9c9121b70239d83861ca9734bb3eb64a5d3f8590`;
the recomputed CSV SHA-256 was
`c40257faeafc39939280d8c694fbcbfa61c96bbe16a6958ba35a8196bffaeda4`.
All 529,390 IDs matched in order. No new leaderboard score was measured.

V142 replay used saved V136 baseline probabilities and a saved correction,
matching the original CSV SHA-256
`23897eb33f1d288349838f34493608e52578551d697a17d9addc731af433a2dc`.
Because it depends on predictions tied to fixed evaluation rows, that replay
cannot predict an unseen evaluation set by itself.

Neither route retrained the full historical ensemble. The held-user benchmark
reported a much larger discrepancy (about 0.001444 in probability), which remains
unresolved. A suggested Sobol/PyTorch explanation was not confirmed. Probability
error and AUC change are different quantities and should not be compared as if
they had the same scale.

The private recovery used Python 3.12, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
scikit-learn 1.8.0, CPU PyTorch 2.5.1, LightGBM 4.6.0 and Numba 0.62.1 on Linux.
This list records that run, not the original training environment and not the
requirements of this public package.
