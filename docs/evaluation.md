# Evaluation and evidence

## Reported competition results

These values come from the participant's supplied portal snapshots. They have
not been independently re-scored by this public package.

| Submission | Reported global ROC AUC | Context |
| --- | ---: | --- |
| V38 | 0.832634 | Re-submission reported the same score |
| V65 | 0.832700 | Saved-model inference later recovered numerically |
| V79 | 0.832715 | Improved V65 |
| V108 | 0.832723 | Prior incumbent |
| V129 | 0.832679 | Regression |
| V136 | 0.832744 | Improved V108 |
| V142 | 0.832748 | Best reported result available in these records |

The 2 November V142 snapshot showed rank 12 and third place at 0.834362.
Those are dated observations, not final standings. The team received a Top 15
finalist invitation on 6 November. The team subsequently did not attend Phase 2,
as reported by Faith. No on-site result is claimed.

## What the recovery measured

| Check on 7 November | Result | Meaning |
| --- | --- | --- |
| V65 recomputation, 529,390 rows | Maximum absolute difference 7.28e-8; fixed limit 1e-7 | Numerical saved-model inference agreement |
| V65 runtime | 381.99 seconds on the recovery CPU environment | One local observation, not a latency guarantee |
| V142 replay, 529,390 rows | Original CSV SHA-256 matched | Saved V136 predictions plus correction, not retraining |
| Historical held-user benchmark | Reported max difference 0.0014441895435540197 | Unresolved; not covered by the smaller replay error |

The V65 output was not byte-identical to the scored CSV. The numerical agreement
does not establish identical AUC: close values can still reverse near-ties.
The [reproduction notes](reproduction.md) preserve the exact distinctions.

## Public synthetic checks

Run `python -m slate_ctr demo`. The example uses seeds 142 and 143 for independent
training and evaluation events. It fits one fixed configuration without a search.
It reports event cross-entropy including no click and global binary AUC across
candidate rows. These are different objectives. The recorded local output is
in [synthetic_demo.json](evidence/synthetic_demo.json); minor float differences
across numerical environments are possible.

Ten automated tests cover outside-option probabilities, event/choice invariants,
zero support, extreme utilities, candidate permutation, the V142 transfer
formula, invalid input rejection, model training and JSON round-trip, and tied
AUC ranks. They check implementation behavior, not competition quality.

The workflow runs these tests and the demo on Python 3.12 and 3.13. See GitHub
Actions for the actual status; a configured workflow is not itself proof of a
passing remote run.

## Lessons and limits

Many experiments used the same development cohorts. Conditional bootstrap
intervals after candidate selection do not account for that repeated reuse.
Some later checks used proxies rather than exact held-out predictions from the
full incumbent ensemble. Tiny local gains therefore deserved less confidence
than their point estimates suggested. Increasing version numbers did not imply
progress: several leaderboard regressions are visible above.

Future work should establish one reproducible training baseline, a frozen
chronological holdout, and a timed new-data inference contract before expanding
the ensemble. This public release documents the useful calculations and the
reproduction failures rather than claiming that the historical system is solved.
