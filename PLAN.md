# Project Blueprint: SafePay 2.0 (AML & Fraud Platform)
Objective: A CPU-optimized, reproducible Data Science pipeline combining Graph Machine Learning, XGBoost, and an LLM-driven Explainability module for transaction monitoring.

## Execution Phases
- Phase 1 (Data & Architecture): Create requirements.txt, src/data_prep.py (leakage removal + temporal split), and docs/ADR/001-temporal-split-and-leakage.md.
- Phase 2 (Graph Feature Engineering): Implement src/graph_engine.py using NetworkX (In/Out Degree, PageRank, Local Clustering Coefficient).
- Phase 3 (Model & Imbalance): Implement src/model.py with XGBoost optimized for Area Under the Precision-Recall Curve (AUPRC).
- Phase 4 (SHAP Explainability): Extract top 3 driving features per flagged transaction in src/explainer.py.
- Phase 5 (DeepSeek Investigator LLM): Generate 2-sentence plain-text AML investigator summaries from SHAP outputs.
- Phase 6 (Streamlit Dashboard): Build app/main.py following UI_RULES.md.
- Phase 7 (Automated Tests): Write pytest unit tests in tests/test_data_prep.py and tests/test_model.py.
- Phase 8 (Scientific README): Document methodology, PR-AUC benchmarks, and CPU scalability limitations.
