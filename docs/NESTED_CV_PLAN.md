# Nested Cross-Validation Plan

Date: 2026-11-11

Tuning counts as beneficial for a model only if (a) the mean outer-fold recall difference (tuned minus default) is positive, (b) the bootstrap 95% CI of that difference excludes zero, and (c) mean F1 does not drop by more than 0.01.
If no model qualifies, default hyperparameters are retained and tuning is reported as a negative result.
If a model qualifies, final parameters come from a single flat search on the full training set in a separate, explicitly approved step, with the holdout evaluated once for that single candidate.
The nested results never change this rule.
