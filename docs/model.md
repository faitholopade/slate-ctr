# Model notes

## Event choice and the outside option

For candidate utilities u_1,...,u_K and no-click utility zero, the candidate
probability is exp(u_j) / (1 + sum_l exp(u_l)). The remaining probability
belongs to no click. `choice_probabilities` computes this distribution using
a maximum-subtracted softmax and returns K+1 columns.

Write each candidate probability as p_j = m q_j, where m is the probability
of any click and q is the conditional item-choice distribution. This separates
two different errors: an event can have the right item ranking but an incorrect
click propensity, or the right propensity but the wrong choice distribution.
Global candidate ROC AUC compares candidates across events as well as within
them, so preserving within-event ranking does not preserve global AUC.

`correct_event_mass` changes logit(m) by a caller-supplied delta and scales all
candidate probabilities together. It preserves q. Boundary masses zero and one
remain fixed; a zero-mass event provides no conditional distribution to recover.
The historical V65 system learned a bounded event-logit correction from event
features and transferred it onto a frozen ensemble. This public function is the
transfer operation, not its learned network or its complete feature pipeline.

`correct_choices` computes q'_j proportional to p_j exp(strength * delta_j),
then returns m q'_j. It preserves m. Zero probabilities retain zero support.
The API checks nonnegative probabilities and total event mass at most one,
with 1e-12 allowance for floating-point summation.

## The V142 correction

`weekend_correction` implements the archived V142 inference calculation:

1. Standardize candidate catalog features with training-only means and standard
   deviations and clip to [-6, 6].
2. Subtract each feature's mean across the current slate.
3. Project onto a shared learned weight vector and multiply by
   (is_weekend - 2/7).
4. Bound the adjustment with 0.2 * tanh(linear / 0.2).
5. Apply conditional-choice transfer with strength 0.5.

The historical code obtained the weekend flag through an integer context
encoding. This API accepts an explicit binary indicator to avoid embedding
that dataset-specific encoding. It accepts already gathered candidate features
instead of catalog IDs. It also supports arbitrary candidate counts, validates
inputs, and handles zero-probability support. On positive ten-item synthetic
inputs, its result was compared directly with the archived inference function;
see the [parity record](evidence/v142_synthetic_parity.json).

The constants describe the archived recipe, not a general recommendation for
new datasets. No catalog, normalization statistics, learned weights, or baseline
predictions are included. Callers must fit their own parameters using training
data and validate the transfer before use.

## Historical modeling context

Earlier components represented user preference and item geometry with latent
factors, modeled exposure effects separately, and used history similarity,
uncertainty and activity features. Calibration heads used an outside option
and, in some experiments, a candidate-ranking loss. Later variants corrected
first-query behavior, event mass and context-conditioned item choice.

The public `LinearChoiceModel` is a new, deliberately small teaching baseline:
shared linear utilities fitted by event cross-entropy and L2 regularization.
Its synthetic features stand in for preference similarity, history similarity,
item popularity and event propensity. It is not a replacement implementation
of the multi-stage competition ensemble.

For real longitudinal data, keep whole events together, build history and
statistics from information available before each query, fit preprocessing only
on training rows, and evaluate on a later time window. The toy generator has
independent events and does not test temporal leakage, user generalization,
exposure selection bias, or production throughput.
