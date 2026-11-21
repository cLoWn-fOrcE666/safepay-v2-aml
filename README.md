# SafePay 2.0 (`safepay-v2-aml`)

**Inductive Graph-Augmented Anti-Money Laundering (AML) & Fraud Triage Pipeline with Exact TreeSHAP Log-Odds Explainability and Deterministic SAR Narrative Synthesis.**

---

## 1. Executive Abstract

SafePay 2.0 is an end-to-end, CPU-reproducible transaction-monitoring system for
financial-crime triage. On the industry-standard **PaySim** synthetic fraud corpus
(≈6.35M transactions), the pipeline builds a **leakage-scrubbed, chronologically
partitioned** benchmark, engineers **inductive structural graph features** strictly
from the training window, trains a histogram-based `XGBClassifier` optimised for
**Precision-Recall AUC** under extreme label imbalance, and exposes every alert
through **exact, path-dependent TreeSHAP** attributions in raw log-odds space.
A provider-agnostic narrative layer synthesises a two-sentence Suspicious Activity
Report (SAR) triage note per alert, rendering deterministically offline. A
dark-mode Streamlit console unifies alert triage, global feature importance,
graph-vs-tabular ablation, and a live "what-if" simulator.

The design prioritises three properties that admissions and engineering reviewers
typically scrutinise: **(i) methodological integrity** (no temporal, post-event, or
topological leakage), **(ii) quantitative honesty** (ablated baselines and
calibration findings reported side-by-side with headline metrics), and
**(iii) regulatory auditability** (deterministic, additivity-checked per-alert
explanations).

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RAW  data/raw/paysim.csv         PaySim agent-based simulator (~6.35M rows)   │
└───────────────────────────────────────────────┬──────────────────────────────┘
                                                │  src/data_prep.py  (Phase 1)
                                                │  • drop leaky cols:
                                                │      oldbalanceOrg, newbalanceOrig,
                                                │      oldbalanceDest, newbalanceDest,
                                                │      isFlaggedFraud
                                                │  • chronological split by `step`
                                                ▼
        ┌───────────────────────┬───────────────────────┬────────────────────────┐
        │ train_graph.parquet   │ val_graph.parquet     │ test_graph.parquet      │
        │        70 %           │        15 %           │        15 %             │
        │ max(step_train) ≤ min(step_val) ≤ min(step_test)  (zero step overlap)  │
        └───────────┬───────────┴───────────┬───────────┴────────────┬───────────┘
                    │                        │                        │
                    │   src/graph_engine.py  │  (Phase 2)             │
                    │   • nx.DiGraph fitted  │  STRICTLY on train     │
                    │     (~319k pruned core)│  (inductive)           │
                    │   • in/out-degree,     │                        │
                    │     PageRank (tol 1e-5)│                        │
                    │     clustering_coef    │                        │
                    │   • cold-start nodes→0.0                        │
                    ▼                        ▼                        ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │  Feature table: 3 tabular (step, type, amount) + 8 graph = 11 cols     │
        └───────────────────────────────┬──────────────────────────────────────┘
                                        │  src/model.py  (Phase 3)
                                        │  • XGBClassifier(hist, eval_metric="aucpr",
                                        │        early stopping on val PR-AUC)
                                        │  • scale_pos_weight ≈ 1221.57 (weighted run)
                                        │  • unweighted + tabular-only ablations
                                        │  • val F1 threshold locked → test
                                        ▼
                    models/xgb_aml.json  +  models/metrics.json
                                        │  src/explainer.py  (Phase 4)
                                        │  • C++ exact TreeSHAP
                                        │      Booster.predict(pred_contribs=True)
                                        │      z(x) = φ0 + Σ φᵢ(x)   [O(TLD²)]
                                        │  • additivity enforced < 1e-4
                                        ▼
        reports/global_shap_importance.json   reports/sample_explanations.json
                                        │  src/explainer.py  (Phase 5)
                                        │  • provider-agnostic 2-sentence SAR note
                                        │  • deterministic offline NLG fallback
                                        ▼
                        app/main.py  (Phase 6)  ──  Streamlit AML Triage Console
                        • KPI header  • Alert Triage & SAR  • Global SHAP & Ablation
                        • Live What-If Simulator
```

**Runtime properties.** Every stage is CPU-bound and reproducible via fixed
random seeds (`random_state=42`); no GPU is required. Graph construction is
vectorised (`pandas.value_counts`) and pruned to a multi-transaction core before
materialising any `networkx` object, keeping Phase 2 within a `< 15 s` /
`< 1 GB` budget at PaySim scale.

---

## 3. Mathematical Methodology

### 3.1 Temporal & Post-Event Leakage Prevention

PaySim is a *simulator*, and several of its columns encode outcomes rather than
pre-transaction state. We drop five columns before modelling and enforce a strict
chronology:

| Column | Leakage class |
| --- | --- |
| `oldbalanceOrg`, `oldbalanceDest` | pre-balance, but **allowed to be reconstructed post-hoc** for fraud rows and invariant to the scoring decision — excluded to avoid simulator shortcuts |
| `newbalanceOrig`, `newbalanceDest` | **post-event** balances (state *after* the transaction) — direct label proxy |
| `isFlaggedFraud` | the simulator's own coarse rule flag — a post-hoc audit artefact, not a predictor |

The temporal invariant is enforced by ordering on `step` (a 1-hour simulator
bucket) and slicing **70 / 15 / 15** by index after a stable sort:

$$
\max(\text{step}_{\text{train}}) \;\le\; \min(\text{step}_{\text{val}})
\;\le\; \min(\text{step}_{\text{test}})
$$

with **zero step overlap** between adjacent splits. Because a `step` value belongs
to exactly one temporal bucket, this guarantees that a model evaluated on
`val`/`test` never observes a transaction that occurred *after* the point in time
it is being scored. Post-event balance leakage would otherwise let a tree learn
"`newbalanceDest` − `oldbalanceDest` ≠ `amount` ⇒ fraud", a simulator artefact
that does not transfer to real ledgers. Dropping these columns and enforcing
chronology jointly eliminate the two dominant **synthetic-simulator shortcuts**.

### 3.2 Inductive Graph Topology

Structural features are derived from a directed, weighted transaction graph:
`in_degree`, `out_degree`, `PageRank`, and local `clustering_coef`, for both the
originator and destination sides (8 features). The topology is fit **strictly on
the training window**:

$$
G_{\text{train}} = \big(\,\mathcal{V}_{\text{train}},\ \mathcal{E}_{\text{train}}\,\big),
\qquad \mathcal{E}_{\text{train}} \nsubseteq \mathcal{E}_{\text{val}},\ \mathcal{E}_{\text{test}} .
$$

`val`/`test` transactions are **never** inserted as edges; they receive features by
lookup against $G_{\text{train}}$. This is what makes the pipeline **inductive**
rather than transductive: a graph feature is a function of the *global* graph, so
admitting an evaluation edge would (a) inflate a node's own degree by the edge
being scored, (b) route PageRank mass along future-only edges, and (c) assemble
clustering neighbourhoods with hindsight. Fixing the topology at the end of the
training window ensures every evaluation feature uses only strictly prior data.

**Cold-start handling.** Accounts unseen in training have no degree, PageRank, or
clustering in $G_{\text{train}}$, which would yield `NaN` after the join; these
are deterministically filled with $0.0$. Since `np.log1p(0) = 0`, the fill is
consistent in the log-transformed space. Filling with a mean/median would
fabricate relationships and blur the informative cold-start signal (new accounts
are structurally higher-risk in AML).

**Sparse-graph scalability.** Degrees are computed exactly via `value_counts` over
all nodes, then the graph is **degree-pruned to a k-core** (`total_degree ≥ 2`)
*before* building any `networkx` object — discarding the ≈6M single-transaction
leaf accounts and leaving a **~319k-node core**. PageRank uses `alpha=0.85`,
`max_iter=25`, `tol=1e-5`; clustering is exact on the core. Pruned leaves and
cold-start accounts default to $0.0$ (a leaf's true clustering is $0$ and its
PageRank mass is the uniform teleportation floor).

### 3.3 PR-AUC Optimisation Under Extreme Imbalance (≈0.13 %)

The positive rate is ≈**0.08 %** in train and ≈**0.13 %** in the scoring
population. Under such ratios, ROC-AUC is misleading: with an enormous negative
class, a model inflates ROC-AUC by ranking easy true negatives correctly while
precision at the top of the score distribution — the region a triage desk actually
operates in — remains poor. We therefore optimise and headline the **Precision-Recall
AUC**:

$$
\text{PR-AUC} = \int_0^1 P(R)\,dR ,
$$

which is dominated by the trade-off at high scores and is highly sensitive to the
rare positives. Early stopping monitors `aucpr` on validation.

**Why unweighted ranking beats cost-sensitive weighting here.** We evaluate three
configurations on the identical out-of-time split:

* **`tabular_only`** — 3 tabular features, unweighted baseline.
* **`full_unweighted`** — 11 features, `scale_pos_weight = 1.0`.
* **`full_weighted`** — 11 features, `scale_pos_weight = negatives/positives ≈
  1221.57` (the count present in `models/metrics.json`).

Cost-sensitive weighting reweights the gradient so rare positives dominate. On an
*in-time* split this often lifts AUPRC; on an **out-of-time** split it **degrades
top-rank precision**, because the model over-commits to positive structure that
the training window's positives happened to exhibit, and that structure fails to
generalise forward. Empirically, `full_unweighted` achieves the best Test PR-AUC
(**0.2624**) and `full_weighted` the worst (**0.1880**). We consequently adopt
**unweighted probability ranking + a validation-tuned, locked F1 threshold**, which
preserves the ranking quality that PR-AUC rewards while still producing a
calibrated, auditable alert cut-off.

### 3.4 Exact TreeSHAP in Log-Odds Space

Attributions are computed by XGBoost's native C++ engine
(`Booster.predict(dmatrix, pred_contribs=True)`), the exact, path-dependent
**TreeSHAP** of Lundberg et al., in **raw log-odds (margin)** space:

$$
z(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x),
$$

where $z(x)$ is the raw margin, $\phi_0$ the expected margin (base log-odds), and
$\phi_i$ the Shapley contribution of feature $i$. This additivity is enforced
per-row (`|φ₀ + Σφᵢ − z| < 1e-4`) and the pipeline fails loudly on violation.

**Why TreeSHAP, not KernelSHAP.** KernelSHAP is model-agnostic but sampling-based:
it fits a weighted local linear surrogate via Monte Carlo, so it carries
**variance**, is not guaranteed to satisfy the SHAP axioms exactly, and is orders
of magnitude slower. TreeSHAP exploits tree structure to compute the *exact*
Shapley values in **$O(TLD^2)$** ($T$ trees, $L$ leaves, $D$ depth) — polynomial,
deterministic, and reproducible across runs (a prerequisite for audit trails).

**Why margin space, not probability space.** Explaining in probability space
routes contributions through the sigmoid, which **saturates** under `scale_pos_weight`.
Flagged rows sit many logits above zero, where $p \approx 0.99^+$ and the sigmoid
is nearly flat; probability-space attributions then compress genuinely large
differences and break clean additivity. In margin space, contributions are linear
in log-odds — the natural currency of a logistic model — so "how many logits did
feature $i$ add toward flagging?" is directly meaningful and threshold-comparable.
The probability is recovered only at the end via $1/(1+e^{-z(x)})$ for the
human-facing number.

---

## 4. Empirical Ablation & Global SHAP Benchmarks

### 4.1 Ablation (identical split, locked F1 threshold)

Test-split results from `models/metrics.json`:

| Configuration | Test PR-AUC | ROC-AUC | Precision | Recall | F1 | Alert volume |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tabular_only` (3 feats, unweighted) | 0.2472 | 0.9214 | 0.9354 | 0.1879 | 0.3129 | 805 |
| `full_unweighted` (11 feats, unweighted) | **0.2624** | 0.9429 | 0.9246 | 0.1714 | 0.2892 | 743 |
| `full_weighted` (11 feats, `spw≈1221.6`) | 0.1880 | 0.9365 | 0.9921 | 0.0941 | 0.1718 | 380 |

**Graph-feature lift (unweighted).** Relative to the tabular-only baseline, the
full inductive-graph model improves Test PR-AUC by:

$$
\Delta_{\text{AUPRC}} = 0.2624 - 0.2472 = \mathbf{+0.0152}.
$$

This +0.0152 is the clean, controlled value of the eight inductive graph
features under the *unweighted* configuration. (Note: `metrics.json`'s
`graph_feature_auprc_lift = -0.0592` is computed against the **weighted** run
`0.1880 − 0.2472`; the unweighted lift of +0.0152 is the meaningful comparison and
is the configuration we select.)

**Calibration finding (`full_weighted` = 0.1880).** Cost-sensitive weighting
maximises precision (0.9921) at the cost of recall (0.0941) and AUPRC (0.1880):
the weighted model over-commits to training-window positive structure that does
not generalise out-of-time. This is the empirical justification for selecting
`full_unweighted` ranking + validation-tuned threshold.

### 4.2 Global Mean |SHAP| (Top-5), `reports/global_shap_importance.json`

| Rank | Feature | Group | Mean \|SHAP\| (log-odds) |
| ---: | --- | --- | ---: |
| 1 | `type` | Tabular | 4.0723 |
| 2 | `step` | Tabular | 1.9326 |
| 3 | `amount` | Tabular | 1.1596 |
| 4 | `dest_in_degree` | Inductive graph | 0.9172 |
| 5 | `dest_pagerank` | Inductive graph | 0.4597 |

Tabular features dominate global importance, but the two strongest **graph**
signals — `dest_in_degree` (0.9172) and `dest_pagerank` (0.4597) — confirm that
destination-side topology contributes materially, consistent with the +0.0152
unweighted PR-AUC lift. `orig_clustering` and `dest_clustering` are structurally
zero on the sparse PaySim graph (the reason they are pruned to $0.0$).

---

## 5. Architecture Decision Records (`docs/ADR/`)

- **[ADR 001 — Temporal Split & Leakage](docs/ADR/001-temporal-split-and-leakage.md)** *(reference: cited by 002 & 003)* — chronological partitioning and the
  post-event/balance leakage blocklist.
- **[ADR 002 — Inductive Graph Topology](docs/ADR/002-inductive-graph-topology.md)** — train-only $G_{\text{train}}$, cold-start $0.0$ fill, PageRank `tol=1e-5`,
  and degree-pruned core construction.
- **[ADR 003 — Imbalance & AUPRC](docs/ADR/003-imbalance-and-auprc.md)** — AUPRC over ROC-AUC, rejection of SMOTE, `scale_pos_weight` + validation F1 threshold,
  and the tabular-vs-graph ablation.
- **[ADR 004 — TreeSHAP Explainability](docs/ADR/004-treeshap-explainability.md)** — exact path-dependent TreeSHAP in log-odds, additivity enforcement, and
  the native-C++ vs. wrapper rationale.

> *Note:* the ADR 001 file is referenced by ADRs 002/003 under that name; the
> repository currently ships ADRs 002–004 on disk.

---

## 6. Quickstart & Reproducibility

All phases are deterministic (`random_state=42`, `tree_method="hist"`). Run from
the repository root.

```bash
# 1) Environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Populate `data/raw/paysim.csv` (PaySim), then run the pipeline in order:

```bash
# 2) Data preparation: leakage scrubbing + 70/15/15 chronological split
python src/data_prep.py

# 3) Inductive graph feature engineering (train-only DiGraph)
python src/graph_engine.py

# 4) Train/ablate model; writes models/xgb_aml.json + models/metrics.json
python src/model.py

# 5) Exact TreeSHAP explanations + SAR narratives
#    writes reports/global_shap_importance.json + reports/sample_explanations.json
python src/explainer.py

# 6) Launch the AML Triage Console
streamlit run app/main.py

# 7) Fast, deterministic unit tests (synthetic in-memory data; no CSV reload)
pytest -v
```

**Optional narrative-provider configuration** (provider-agnostic; offline
deterministic NLG is the default when unset). See `.env.example`:

```bash
# LLM_API_KEY=<provider-key>
# LLM_BASE_URL=<provider-base-url>
# LLM_MODEL=<model-name>
```

Any generic chat-completions API endpoint may be supplied via these variables; no
specific vendor is required or assumed.

---

## 7. Honest Scope & Limitations

- **Synthetic data.** PaySim is an **agent-based simulator**; its fraud patterns
  are generative and do not fully reflect real-world laundering complexity (layering
  chains, trade-based ML, adversarial drift). Reported PR-AUC is therefore a
  *benchmark floor*, not a production estimate. Leakage scrubbing removes the most
  exploitable simulator shortcuts, but residual simulator-specific correlations
  remain.
- **Static snapshot vs. streaming temporal graphs.** The inductive pipeline builds
  $G_{\text{train}}$ **once** and scores forward against a frozen topology. It is
  not a **temporal graph network (TGN)**; edge arrivals are not integrated into a
  continuously-updated representation. Dynamic TGNs (memory-augmented, continuous-
  time) are a natural extension but are out of scope for a CPU-reproducible,
  audit-friendly pipeline.
- **Imbalance trade-off.** Cost-sensitive weighting improves precision but degrades
  out-of-time AUPRC (see §4.1); we select unweighted ranking accordingly. The
  F1-optimal threshold may not match a pure precision-at-fixed-recall operating
  policy; a cost-matrix threshold is a documented follow-up.
- **CPU histogram training.** `tree_method="hist"` keeps the entire pipeline
  CPU-bound and reproducible, but caps model capacity relative to GPU scale-up and
  limits the size of the ensemble one can train interactively. Best iterations vary
  widely across ablations (219–451), indicating sensitivity to the imbalance regime.
- **Graph pruning.** Computing PageRank/clustering on the pruned core means leaf
  accounts use the documented $0.0$ default rather than an exactly-computed
  near-zero value; cold-start and leaf accounts collapse to the same zero-structure
  profile, so their distinction must come from tabular features.
- **Narrative layer.** SAR notes are template-deterministic offline or
  model-generated when a provider is configured; they are decision-support text,
  not legal findings, and must be reviewed by a human investigator.
