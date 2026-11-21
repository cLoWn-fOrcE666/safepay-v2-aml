# ADR 004: TreeSHAP Explainability in Raw Log-Odds Space

## Status
Accepted

## Context
SafePay 2.0 flags potential money-laundering transactions for human investigators.
Two hard requirements follow:

1. **Regulatory auditability.** Every automated alert must be accompanied by a
   defensible, reproducible, per-transaction justification of *why* the model
   raised it. Auditors must be able to reconstruct the exact contribution of each
   feature to the decision.
2. **Investigator triage.** Analysts need a compact, ranked list of the handful
   of features that drove an alert (default Top-3), with direction
   (risk-increasing vs. risk-decreasing) and magnitude, so they can quickly
   form a hypothesis. Alerts are ~0.13% of volume; the explanation must be cheap
   enough to compute for thousands of alerts without blocking the dashboard.

The model (Phase 3) is an XGBoost gradient-boosted tree ensemble trained under
extreme class imbalance (~0.08% positive rate in train, ~0.13% in the scoring
population) with a validation-locked decision threshold
(`metrics.json: threshold = 0.9893`).

## Decision
Use **exact, path-dependent TreeSHAP** computed in **raw log-odds (margin)
space** — specifically via XGBoost's native C++ engine,
`Booster.predict(dmatrix, pred_contribs=True)`, which implements Lundberg's exact
$O(TLD^2)$ TreeSHAP directly and returns an `(n_samples, n_features + 1)` array
(per-feature $\phi_i$ plus the trailing intercept $\phi_0$).

### Why TreeSHAP over KernelSHAP
KernelSHAP is model-agnostic but is a sampling-based approximation: it fits a
weighted local linear surrogate with Monte Carlo estimates, so results carry
**sampling variance** and are not guaranteed to satisfy the SHAP axioms exactly.
TreeSHAP computes the *exact* Shapley values for tree ensembles by exploiting
tree structure — no sampling, no randomness, fully deterministic and reproducible
across runs (critical for audit trails). It is also orders of magnitude faster,
which is essential for per-alert explanation at dashboard scale.

### Why raw log-odds margin space over probability space
Explaining in probability space (`model_output="probability"`) maps contribution
through the sigmoid, which **saturates** at the extremes. Under a 0.13% fraud
rate with `scale_pos_weight ≈ 1221`, flagged rows sit at raw margins of many logit
units, where the sigmoid is nearly flat (p ≈ 0.99+). Probability-space
attributions then compress genuinely large differences into nearly identical
values, and the additivity identity no longer holds cleanly. In **margin space**
contributions are additive *exactly*:

$$z(x) = \phi_0 + \sum_{i} \phi_i(x)$$

where $z(x)$ is the raw log-odds, $\phi_0$ the base log-odds (expected margin), and
$\phi_i$ the per-feature SHAP value. This linearity in log-odds is the natural
currency of a logistic model: contributions compose without sigmoid distortion, so
"how many logits did feature $i$ add toward flagging?" is directly meaningful and
threshold-comparable. The predicted probability is recovered only at the final
step via $1/(1+e^{-z(x)})$ for the human-facing number.

### Mathematical justification
- **Complexity.** Exact Shapley values are exponential, $O(2^M)$ in the number of
  features $M$. TreeSHAP reduces this to **$O(TLD^2)$** ($T$ trees, $L$ leaves, $D$
  depth) by sharing computations across the tree structure — polynomial and
  practical.
- **Zero Monte Carlo variance.** Because `tree_path_dependent` uses the tree cover
  directly (no background/reference dataset and no sampling), attributions are
  deterministic. The same transaction always yields the same explanation — a
  prerequisite for reproducible regulatory reporting.
- **Exact additivity / local accuracy.** Guaranteed by construction:
  `base_value + sum(shap) == raw_margin`. `src/explainer.py` enforces this
  invariant on every evaluated row
  (`abs(base_values + shap_values.sum(axis=1) - raw_margin) < 1e-4`) and fails
  loudly if violated, so no corrupted attribution can reach the reports. The
  `1e-4` tolerance reflects XGBoost's internal `float32` accumulation precision
  between the `pred_contribs` and `output_margin` code paths.

## Implementation Notes
- **Native C++ TreeSHAP, not the `shap` Python wrapper.** Attributions are read
  directly from XGBoost's booster via
  `booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)`.
  Column slicing splits the result into `shap_values = contribs[:, :-1]`
  (per-feature log-odds $\phi_i$) and `base_values = contribs[:, -1]` (exact
  log-odds intercept $\phi_0$), so the intercept and contributions are produced by
  the *same* kernel that produced the model.
- **Why this over `shap.TreeExplainer`.** On newer Python versions (e.g. 3.14)
  `shap.TreeExplainer` can misparse the `base_score` intercept from XGBoost's JSON
  model header, so `base + sum(contribs)` drifts from the true margin (observed
  error ≈ 0.19, far above tolerance). The native `pred_contribs` path has no
  separate JSON-header parsing step and is the ground-truth implementation of the
  same exact algorithm, eliminating that mismatch.
- **Categorical fidelity.** The `DMatrix` is built with `enable_categorical=True`
  and the booster's own `feature_names`, preserving the exact feature order and
  the `type` category codes the model learned.

## Consequences
- **Sub-millisecond per-alert inference.** TreeSHAP on this ensemble is fast
  enough to explain individual alerts interactively; batch/global importance over
  a 2,000-row stratified sample completes quickly on CPU.
- **Precomputed JSON artifacts.** `reports/global_shap_importance.json` (mean-|SHAP|
  ranking) and `reports/sample_explanations.json` (Top-3 drivers for the highest-risk
  alerts) are generated offline by the CLI. Phase 6 (Streamlit) reads these files
  directly, so the UI renders explanations **non-blocking** without recomputing
  SHAP on the request path.
- **Structured payloads ready for Phase 5.** Each explanation is a clean,
  JSON-serializable dict (`predicted_probability`, `raw_log_odds`, `base_log_odds`,
  `threshold`, `is_flagged`, `top_drivers[]`), giving the LLM investigator exactly
  the ranked driver payload it needs to synthesise a 2-sentence summary.
- **Trade-off.** Path-dependent TreeSHAP attributes against the *training* tree
  cover rather than a user-supplied background distribution; this is deliberate —
  it is the exact, variance-free method and the correct choice for auditability.
  If a background-conditional interpretation is ever required, it can be added as
  a separate explainer mode without changing the payload contract.
