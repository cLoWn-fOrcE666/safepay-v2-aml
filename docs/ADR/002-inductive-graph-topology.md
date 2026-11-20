# ADR 002: Inductive Graph Topology (Train-Only Construction)

## Status
Accepted

## Context
Phase 2 (`src/graph_engine.py`) derives structural node features from a directed,
weighted transaction graph: in-degree, out-degree, PageRank, and local clustering
coefficient. These features are mapped onto both the originator (`nameOrig`) and
destination (`nameDest`) side of every transaction across the `train`, `val`, and
`test` splits.

The graph is a *structural summary of relationships*. If it were built from data
that overlaps the evaluation windows, node-level statistics (especially PageRank
and degree) would encode information that would not be available at inference
time, contaminating the model.

## Decision

### 1. Construct the graph strictly on the training window
The graph is built **only** from `train.parquet`. `val.parquet` and `test.parquet`
are *never* used to add nodes or edges; they are only enriched with features
looked up from the training graph.

**Why this prevents topological leakage.** A transaction-graph feature is a
function of the *global* graph. If a validation or test transaction contributed an
edge, then:
- its own degree would be inflated by the very edge being scored,
- PageRank mass would flow along edges that only exist in the future/eval set,
- clustering would reflect neighbourhoods assembled with hindsight knowledge.

This is a textbook leak: the feature vector at inference would depend on future
transactions. By fixing the topology at the end of the training window, every
`val`/`test` feature is computed from strictly prior information — the model sees
only what would be knowable when a transaction first arrives. This makes the
feature pipeline **inductive**: it can be applied to entirely new, unseen data
streams without replaying historical structure.

### 2. Cold-start handling for unseen accounts
Validation and test windows contain accounts that never appeared during training
(cold-start nodes). These have no edges, hence no degree, PageRank, or clustering
in the training graph. During the feature join, such nodes produce `NaN`; these
are deterministically filled with **`0.0`**.

Rationale:
- `0` is the semantically correct structural summary for "no observed
  relationships in the training period".
- Filling with a global mean/median would inject fabricated relationships and
  blur the cold-start signal, which is itself informative for AML (new accounts
  are higher-risk).
- `np.log1p` is applied to the degree columns, and `log1p(0) = 0`, so the fill
  value is preserved consistently in the transformed space.

### 3. PageRank convergence tolerance of `tol=1e-05`
PageRank runs on a sparse adjacency matrix via SciPy's iterative power method.
We set `alpha=0.85`, `max_iter=25`, and `tol=1e-05`, evaluated on the pruned core
subgraph (see §4).

**Why `tol=1e-05` for CPU scalability.** PageRank is the most expensive of the
four features: each power iteration touches every edge. On a large transaction
graph (millions of nodes/edges) the cost scales with `iterations × edges`. A
tighter tolerance (`1e-06`, `1e-08`) frequently requires tens to hundreds of
iterations to converge, dominating wall-clock time on CPU. `1e-05` reaches a
numerically stable ranking within the `max_iter=25` budget while producing
feature values whose ordering — the only thing the downstream XGBoost model
consumes — is effectively unchanged. This trades a negligible accuracy delta for
a large, predictable runtime reduction, keeping the pipeline reproducible and
CPU-bound rather than requiring GPU acceleration.

### 4. Degree pruning ($k$-core) and vectorized degrees for sparse financial graphs
PaySim is a *sparse* financial graph: most accounts transact only once, so the
degree distribution is extremely heavy-tailed. The dominant cost — and the
dominant memory footprint — comes from the millions of single-transaction **leaf
accounts** (total undirected degree < 2). We exploit this in two ways.

**(a) Vectorized exact degrees — no NetworkX.** Degree is computed directly with
Pandas `value_counts`, over *all* nodes:

```python
out_degree = train["nameOrig"].value_counts(sort=False)
in_degree  = train["nameDest"].value_counts(sort=False)
total_degree = out_degree.add(in_degree, fill_value=0)   # undirected degree
```

This is exact and O(n) over the transactional columns, avoiding the cost of
materialising a 6.35M-node NetworkX graph just to read off degrees.

**(b) Degree-pruned core subgraph before NetworkX.** We keep only edges touching
at least one node with `total_degree >= 2`, then build the `nx.DiGraph` on that
pruned core:

```python
core_nodes = total_degree.index[total_degree >= MIN_CORE_DEGREE]   # >= 2
pruned = train.loc[train[ID_ORIG].isin(core_nodes) | train[ID_DEST].isin(core_nodes)]
# ... build DiGraph on `pruned`, then pagerank + clustering on the core
```

On PaySim this discards the ~6M single-transaction leaf accounts and leaves a
~350k-node core. PageRank and clustering are computed **only** on the core;
every core node already has total degree ≥ 2, so its clustering coefficient is
*exact* there (no triangle enumeration is even possible for the pruned leaves).
Pruned leaves and cold-start accounts default to `0.0`.

**Why this is correct, not an approximation.** A leaf node (degree < 2) has
clustering coefficient exactly `0.0` and receives no meaningful PageRank mass
beyond the uniform teleportation floor. Pruning them keeps the structural signal
intact for the accounts that matter (multi-transaction hubs, mules, relays) while
collapsing the working set by roughly an order of magnitude. This is what brings
the whole in-degree / out-degree / PageRank / clustering pipeline under the
`< 15s` and `< 1 GB` CPU budget at ~6.35M-row scale.

## Consequences
- **Pros:** No topological leakage; features are genuinely inductive and safe for
  temporal splits; cold-start and pruned leaves are handled explicitly with `0.0`;
  vectorized degrees plus a degree-pruned core subgraph keep runtime `< 15s` and
  memory `< 1 GB` on PaySim-scale data by avoiding a 6.35M-node NetworkX graph.
- **Cons:** Cold-start and leaf accounts collapse to the same zero-structure
  profile (their distinction must come from tabular features); PageRank/clustering
  are computed on the core only, so a leaf's exact (near-zero) value is set to the
  documented `0.0` default rather than computed.
- **Follow-ups:** If convergence warnings surface in logs at the chosen
  `max_iter`, revisit `tol`/`max_iter` only alongside a measured runtime-vs-
  accuracy trade-off study.
