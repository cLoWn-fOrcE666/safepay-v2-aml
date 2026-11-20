"""Phase 2: Inductive graph feature engineering for SafePay 2.0.

Builds structural node features from a *training-window-only* transaction graph,
then maps them onto both the originator (``nameOrig``) and destination
(``nameDest``) side of every transaction across train / val / test.

Performance design (``< 15s``, ``< 1 GB`` on ~6.35M PaySim rows):

* Exact in/out degrees are computed with vectorized :meth:`Series.value_counts`
  over ALL train nodes — no NetworkX needed for degree.
* The graph is *degree-pruned*: edges touching only single-transaction leaf
  accounts (total undirected degree < 2) are dropped before NetworkX is
  instantiated. On PaySim this removes ~6M leaf accounts, leaving a ~350k-node
  core subgraph on which ``pagerank`` and ``clustering`` are computed.
* Pruned and cold-start nodes (absent from the training graph) default to 0.0.

The graph is fit strictly on train to avoid topological leakage; ``log1p`` is
applied to degree columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR: Final[Path] = Path("data/processed")

ID_ORIG: Final[str] = "nameOrig"
ID_DEST: Final[str] = "nameDest"
AMOUNT_COL: Final[str] = "amount"

# PageRank hyper-parameters tuned for fast CPU / SciPy sparse convergence on the
# pruned core subgraph.
PAGERANK_ALPHA: Final[float] = 0.85
PAGERANK_MAX_ITER: Final[int] = 25
PAGERANK_TOL: Final[float] = 1e-05

# Total undirected degree threshold below which a node is pruned as a leaf.
MIN_CORE_DEGREE: Final[int] = 2

# Structural node attributes computed on the training graph.
_NODE_FEATURES: Final[tuple[str, ...]] = (
    "in_degree",
    "out_degree",
    "pagerank",
    "clustering",
)

# Degree columns receive ``log1p``; pagerank/clustering are left as-is.
_LOG_COLUMNS: Final[frozenset[str]] = frozenset({"in_degree", "out_degree"})


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a split parquet, failing loudly if the upstream phase is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing '{path}'. Run Phase 1 (src/data_prep.py) to produce the "
            "processed splits before executing Phase 2."
        )
    return pd.read_parquet(path)


def _core_subgraph(pruned: pd.DataFrame) -> nx.DiGraph:
    """Aggregate parallel edges on the pruned core and build the DiGraph.

    Repeated ``nameOrig -> nameDest`` pairs collapse into a single weighted edge
    carrying ``tx_count`` and ``total_amount``.
    """
    edges: pd.DataFrame = (
        pruned.groupby([ID_ORIG, ID_DEST], sort=False, observed=True)
        .agg(
            tx_count=(AMOUNT_COL, "size"),
            total_amount=(AMOUNT_COL, "sum"),
        )
        .reset_index()
    )

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        (
            (orig, dest, float(total))
            for orig, dest, total in zip(
                edges[ID_ORIG], edges[ID_DEST], edges["total_amount"]
            )
        )
    )
    for orig, dest, count in zip(
        edges[ID_ORIG], edges[ID_DEST], edges["tx_count"]
    ):
        graph[orig][dest]["tx_count"] = int(count)

    logger.info(
        "Core subgraph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


def _compute_features(train: pd.DataFrame) -> pd.DataFrame:
    """Compute exact degrees over all nodes and core-only pagerank/clustering."""
    out_degree: pd.Series = train[ID_ORIG].value_counts(sort=False)
    in_degree: pd.Series = train[ID_DEST].value_counts(sort=False)

    # Total undirected degree per node = originator count + destination count.
    total_degree: pd.Series = out_degree.add(in_degree, fill_value=0)
    core_nodes: pd.Index = total_degree.index[total_degree >= MIN_CORE_DEGREE]
    logger.info(
        "Degree pruning: %d nodes total, %d core (>= %d), %d pruned",
        len(total_degree),
        len(core_nodes),
        MIN_CORE_DEGREE,
        len(total_degree) - len(core_nodes),
    )

    # Prune edges to keep only those touching a core node, before touching
    # NetworkX; this discards the ~6M single-transaction leaf accounts.
    orig_mask = train[ID_ORIG].isin(core_nodes)
    dest_mask = train[ID_DEST].isin(core_nodes)
    pruned = train.loc[orig_mask | dest_mask, [ID_ORIG, ID_DEST, AMOUNT_COL]]
    del orig_mask, dest_mask

    graph = _core_subgraph(pruned)
    del pruned

    # Pagerank and clustering are computed only on the core subgraph; every core
    # node has total degree >= 2, so clustering is exact there.
    pagerank: pd.Series = pd.Series(
        nx.pagerank(
            graph,
            alpha=PAGERANK_ALPHA,
            max_iter=PAGERANK_MAX_ITER,
            tol=PAGERANK_TOL,
        ),
        dtype="float64",
        name="pagerank",
    )
    clustering: pd.Series = pd.Series(
        nx.clustering(graph.to_undirected(as_view=True)),
        dtype="float64",
        name="clustering",
    )
    del graph

    # Assemble the full per-node feature frame (all nodes; non-core default 0.0).
    features = pd.DataFrame(
        {
            "in_degree": in_degree.reindex(total_degree.index, fill_value=0).astype("int64"),
            "out_degree": out_degree.reindex(total_degree.index, fill_value=0).astype("int64"),
            "pagerank": pagerank,
            "clustering": clustering,
        }
    ).fillna(0.0)
    features.index.name = "node"
    return features


def _map_features(
    df: pd.DataFrame, features: pd.DataFrame, id_col: str, prefix: str
) -> pd.DataFrame:
    """Join node features onto one side of the transaction table.

    Cold-start accounts (present in the split but not in the training graph)
    produce a NaN during the join and are filled with 0.0.
    """
    renamed = features.rename(columns={c: f"{prefix}_{c}" for c in _NODE_FEATURES})
    joined = df.join(renamed, on=id_col)

    for feature in _NODE_FEATURES:
        col = f"{prefix}_{feature}"
        joined[col] = joined[col].fillna(0.0).astype("float64")
        if feature in _LOG_COLUMNS:
            joined[col] = np.log1p(joined[col])

    return joined


def _engineer_split(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Map features onto both endpoints, then drop raw string ID columns."""
    # Restrict the feature index to nodes actually touched by this split to keep
    # the join small and fast.
    touched = pd.Index(
        pd.unique(pd.concat([df[ID_ORIG], df[ID_DEST]], ignore_index=True))
    ).intersection(features.index)
    split_features = features.loc[touched]

    df = _map_features(df, split_features, ID_ORIG, "orig")
    df = _map_features(df, split_features, ID_DEST, "dest")
    return df.drop(columns=[ID_ORIG, ID_DEST])


def run(processed_dir: Path = PROCESSED_DIR) -> None:
    """Execute the full Phase 2 pipeline and persist the graph-enriched splits."""
    train = _load_parquet(processed_dir / "train.parquet")
    val = _load_parquet(processed_dir / "val.parquet")
    test = _load_parquet(processed_dir / "test.parquet")

    features = _compute_features(train)

    train_out = _engineer_split(train, features)
    del train
    val_out = _engineer_split(val, features)
    del val
    test_out = _engineer_split(test, features)
    del test

    outputs = {
        "train_graph.parquet": train_out,
        "val_graph.parquet": val_out,
        "test_graph.parquet": test_out,
    }
    for filename, frame in outputs.items():
        frame.to_parquet(processed_dir / filename, index=False)
        logger.info("Wrote %s (%d rows)", processed_dir / filename, len(frame))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
