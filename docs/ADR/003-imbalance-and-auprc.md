# ADR 003: Imbalance Handling & AUPRC Optimisation

## Status
Accepted

## Context
The SafePay 2.0 fraud label (`isFraud`) is extremely rare. On the Phase 2
training split the positive rate is roughly **0.08%** (≈3,643 positives in ≈4.45M
transactions). At such ratios the choice of metric, imbalance strategy, and
decision threshold dominate whether the model is useful in production, where the
real cost is the analyst time spent triaging flagged alerts.

`src/model.py` therefore makes three coupled decisions: (1) optimise and report
**AUPRC**, not ROC-AUC; (2) reject synthetic oversampling (SMOTE) in favour of
**cost-sensitive weighting + validation-tuned threshold**; (3) keep a
**tabular-only baseline** on the identical split to prove the graph features add
real value.

## Decision

### 1. Optimise AUPRC, not ROC-AUC
ROC-AUC is computed over the full range of the false-positive-rate axis. Under
extreme class imbalance the negative class is enormous, so a model can inflate
ROC-AUC by ranking easy negatives correctly while the precision in the operating
region that actually matters — the top of the score distribution — stays poor.

AUPRC focuses entirely on the positive (fraud) class and is dominated by the
**precision/recall trade-off at high scores**, which is exactly the region a
triage system operates in. It is far more sensitive to the rare positives and
does not reward trivially easy true negatives. We consequently:

* set `eval_metric="aucpr"` so early stopping selects the iteration that
  maximises PR-AUC on the validation set;
* report AUPRC as the headline test metric (alongside ROC-AUC for completeness).

### 2. Reject SMOTE; use cost-sensitive weighting + threshold tuning
Synthetic Minority Over-sampling (SMOTE) interpolates new minority points in
feature space. For a **graph-derived, mostly binary/zero-inflated feature set**
(degree, PageRank, clustering that are 0.0 for the ~90% pruned leaf accounts),
linear interpolation between neighbours produces feature vectors that do not
correspond to any real transaction topology — it manufactures structurally
impossible accounts. SMOTE also inflates training time and memory on a 4.45M-row
table, and its synthetic points can leak local structure across the temporal
boundary.

Instead we use two complementary, leak-free levers:

* **Cost-sensitive learning** via `scale_pos_weight = negatives / positives`
  (~1,221×). This reweights the gradient so the rare positives contribute
  proportionally, without fabricating data. We keep an **unweighted
  (`scale_pos_weight = 1.0`) ablation** in the metrics to quantitatively confirm
  the weighting helps AUPRC on the identical split.
* **Validation-tuned decision threshold.** The classifier's raw probability is
  converted to an alert by a threshold chosen on `val_graph.parquet` to maximise
  **F1**, and we additionally report **Precision at 80% Recall** — the operating
  point analysts usually care about (catch 80% of fraud, minimise false alarms).
  The threshold is then **locked** and applied unchanged to the test set, so the
  reported test numbers reflect a genuinely out-of-sample operating point.

### 3. Prove architectural value with a tabular-vs-graph ablation
Architecture claims ("the graph features help") are worthless without a
controlled comparison. `src/model.py` trains a **tabular-only baseline** that
drops all eight Phase 2 graph features (`orig_*`/`dest_*` degree, PageRank,
clustering) and keeps everything else — same split, same hyper-parameters, same
thresholding procedure. The only difference is the presence of graph topology.

The **`graph_feature_auprc_lift`** (`full_weighted` test AUPRC − `tabular_only`
test AUPRC) recorded in `models/metrics.json` is therefore a clean, quantitative
measure of what the Phase 2 graph engineering contributes. If that lift were near
zero, the graph pipeline would be unjustified complexity; the ablation makes the
trade-off explicit and auditable.

## Consequences
- **Pros:** Headline metric aligns with the imbalanced, triage-oriented objective;
  no fabricated samples; threshold is selected out-of-sample and locked before
  touching test; the graph-vs-tabular ablation yields a defensible number for the
  value of Phase 2; all decisions are reproducible via fixed `random_state=42`,
  `tree_method="hist"`, and early stopping on `aucpr`.
- **Cons:** F1-optimal thresholds can favour precision/recall balance that differs
  from a pure precision-at-fixed-recall policy; `scale_pos_weight` can over-boost
  positives (mitigated by the unweighted ablation, reported side-by-side).
- **Follow-ups:** Validate the locked threshold's stability across additional
  temporal folds; consider explicit cost-matrix thresholding if alert-review cost
  ratios become known.
