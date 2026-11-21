"""Phase 6: SafePay 2.0 AML Investigator Dashboard (Streamlit, dark mode).

A polished triage console over the Phase 3 model, Phase 4 TreeSHAP explanations,
and Phase 5 investigator narratives. Three tabs:

* Alert Triage & SAR Report — drill into any of the 25 queued high-risk alerts.
* Global SHAP & Ablation    — model-wide feature importance + graph-vs-tabular lift.
* Live What-If Simulator     — score a synthetic transaction and narrate it live.

SafePay 2.0 | Financial-crime triage UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Path bootstrap: allow importing `src.*` when launched via `streamlit run`
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainer import (  # noqa: E402  (after sys.path bootstrap)
    AMLTreeExplainer,
    generate_investigator_report,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = MODELS_DIR / "metrics.json"
GLOBAL_SHAP_PATH = REPORTS_DIR / "global_shap_importance.json"
SAMPLE_EXPL_PATH = REPORTS_DIR / "sample_explanations.json"

# Brand palette (dark slate financial-crime aesthetic).
BG = "#0E1117"
CARD_BG = "#161B22"
BORDER = "#30363D"
CRIMSON = "#FF4B4B"  # positive risk drivers
EMERALD = "#00CC96"  # negative / risk-lowering drivers
TEXT = "#E6EDF3"
MUTED = "#8B949E"

TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]

# --------------------------------------------------------------------------- #
# Page + theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SafePay 2.0 | AML Triage Console",
    page_icon="🛡️",
    layout="wide",
)

_CSS = f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {CARD_BG}; }}
    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 18px;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: "SFMono-Regular", Menlo, Consolas, monospace;
        color: {TEXT};
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    /* Compliance narrative card */
    .compliance-card {{
        background-color: {CARD_BG};
        border-left: 4px solid {CRIMSON};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 18px 20px;
        margin: 8px 0 16px 0;
        font-family: "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 0.95rem;
        line-height: 1.55;
        color: {TEXT};
    }}
    .compliance-card .cc-tag {{
        display: block;
        color: {MUTED};
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER};
    }}
    .compliance-card .cc-body {{
        display: block;
        margin-top: 4px;
    }}
    /* Risk badge */
    .risk-badge {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
    }}
    .risk-badge.flagged {{ background-color: rgba(255,75,75,0.15); color: {CRIMSON};
        border: 1px solid {CRIMSON}; }}
    .risk-badge.clear {{ background-color: rgba(0,204,150,0.15); color: {EMERALD};
        border: 1px solid {EMERALD}; }}
    h1, h2, h3 {{ color: {TEXT}; }}
    .app-subtitle {{ color: {MUTED}; font-size: 1.0rem; margin-top: -8px; }}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> Any:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner="Loading TreeSHAP explainer …")
def get_explainer() -> AMLTreeExplainer:
    return AMLTreeExplainer()


def _load_with_error(path: Path, label: str) -> Any | None:
    """Load a JSON artifact, surfacing a friendly error and returning None if absent."""
    if not path.is_file():
        st.error(
            f"**{label} not found.** Expected `{path.relative_to(PROJECT_ROOT)}`. "
            "Run the earlier pipeline phases (`src/model.py`, `src/explainer.py`) first."
        )
        return None
    try:
        return load_json(str(path))
    except Exception as exc:  # noqa: BLE001 - surface any parse error gracefully
        st.error(f"**Could not read {label}:** {exc}")
        return None


# --------------------------------------------------------------------------- #
# Plotly helpers
# --------------------------------------------------------------------------- #
def _base_layout(title: str) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=15)),
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="monospace"),
        margin=dict(l=10, r=20, t=50, b=20),
    )


def make_driver_bar(drivers: list[dict[str, Any]]) -> go.Figure:
    """Diverging horizontal bar of per-alert SHAP drivers (crimson +, emerald -)."""
    ordered = sorted(drivers, key=lambda d: d.get("shap_log_odds", 0.0))
    features = [d["feature"] for d in ordered]
    values = [float(d.get("shap_log_odds", 0.0)) for d in ordered]
    colors = [CRIMSON if v > 0 else EMERALD for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT, family="monospace"),
            hovertemplate="%{y}<br>SHAP %{x:+.4f} log-odds<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("TreeSHAP Drivers (log-odds)"))
    fig.update_layout(showlegend=False, height=max(260, 70 * len(features)))
    fig.add_vline(x=0, line_width=1, line_color=MUTED)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def make_global_bar(
    importance: list[dict[str, Any]], graph_features: set[str]
) -> go.Figure:
    """Global mean |SHAP| bar, color-coded by feature group (Tabular vs Graph)."""
    ordered = sorted(importance, key=lambda d: d["mean_abs_shap_log_odds"])
    features = [d["feature"] for d in ordered]
    values = [float(d["mean_abs_shap_log_odds"]) for d in ordered]
    colors = [EMERALD if f in graph_features else CRIMSON for f in features]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>Mean |SHAP| %{x:.4f}<extra></extra>",
        )
    )
    # Legend proxies (one crimson = tabular, one emerald = graph).
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=CRIMSON), name="Tabular"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=EMERALD), name="Inductive Graph"))
    fig.update_layout(**_base_layout("Global Mean |SHAP| (log-odds)"))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=max(320, 44 * len(features)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def make_ablation_bar(metrics: dict[str, Any]) -> go.Figure:
    """Ablation benchmark: tabular_only vs full_unweighted Test PR-AUC."""
    ablation = metrics.get("ablation", {})
    labels = ["Tabular Only", "Tabular + Graph"]
    keys = ["tabular_only", "full_unweighted"]
    values = [float(ablation.get(k, {}).get("test", {}).get("auprc", 0.0)) for k in keys]
    colors = [MUTED, EMERALD]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT, family="monospace"),
            hovertemplate="%{x}<br>Test PR-AUC %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Ablation: Test PR-AUC"))
    fig.update_layout(showlegend=False, height=360)
    fig.update_yaxes(gridcolor=BORDER, range=[0, max(values) * 1.25 if values else 1])
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def render_sar_card(report: str) -> None:
    safe = report.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="compliance-card">'
        f'<div class="cc-tag">Compliance Narrative · SAR Triage</div>'
        f'<div class="cc-body">{safe}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_badge(is_flagged: bool) -> None:
    cls = "flagged" if is_flagged else "clear"
    label = "FLAGGED · ABOVE THRESHOLD" if is_flagged else "CLEAR · BELOW THRESHOLD"
    st.markdown(f'<span class="risk-badge {cls}">{label}</span>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🛡️ SafePay 2.0 | AML Triage Console")
st.markdown(
    '<div class="app-subtitle">Inductive graph + tabular fraud model · exact TreeSHAP '
    "attributions · AI investigator narratives</div>",
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------------------- #
# Load artifacts
# --------------------------------------------------------------------------- #
metrics = _load_with_error(METRICS_PATH, "Model metrics")
global_shap = _load_with_error(GLOBAL_SHAP_PATH, "Global SHAP importance")
sample_explanations = _load_with_error(SAMPLE_EXPL_PATH, "Sample explanations")

# --------------------------------------------------------------------------- #
# KPI header row
# --------------------------------------------------------------------------- #
if metrics and sample_explanations is not None:
    ablation = metrics.get("ablation", {})
    full_auprc = float(ablation.get("full_unweighted", {}).get("test", {}).get("auprc", 0.0))
    tab_auprc = float(ablation.get("tabular_only", {}).get("test", {}).get("auprc", 0.0))
    delta = full_auprc - tab_auprc
    threshold = float(metrics.get("threshold", 0.0))

    top_graph_driver = "dest_in_degree"
    for entry in global_shap or []:
        if entry.get("feature", "").startswith(("orig_", "dest_")):
            top_graph_driver = entry["feature"]
            break

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Best Test PR-AUC",
        f"{full_auprc:.4f}",
        delta=f"{delta:+.4f} vs tabular",
        help="Full (graph+tabular) vs tabular-only baseline on the locked test split.",
    )
    k2.metric("Alert Threshold", f"{threshold:.4f}", help="F1-optimal operating point from validation.")
    k3.metric("Top Graph Driver", top_graph_driver, help="Highest mean |SHAP| inductive-graph feature.")
    k4.metric("Queued High-Risk Alerts", f"{len(sample_explanations)}", help="Alerts in this triage queue.")
    st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab1, tab2, tab3 = st.tabs(
    ["🚨 Alert Triage & SAR Report", "📊 Global SHAP & Ablation", "🧪 Live What-If Simulator"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Alert Triage & SAR Report
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Alert Triage & SAR Report")
    if not sample_explanations:
        st.info("No sample explanations available. Run Phase 4 (`src/explainer.py`).")
    else:
        n_alerts = len(sample_explanations)
        alert_no = st.slider(
            "Select Alert", min_value=1, max_value=n_alerts, value=1, step=1
        )
        alert = sample_explanations[alert_no - 1]

        prob = float(alert.get("predicted_probability", 0.0))
        raw = float(alert.get("raw_log_odds", 0.0))
        base = float(alert.get("base_log_odds", 0.0))
        flagged = bool(alert.get("is_flagged", False))
        threshold = float(alert.get("threshold", 0.0))

        left, right = st.columns([1, 3])
        with left:
            render_badge(flagged)
            st.markdown("")
            st.metric("Fraud Probability", f"{prob:.2%}")
            st.metric("Raw Log-Odds", f"{raw:+.4f}")
            st.caption(f"Base log-odds (φ₀): {base:+.4f} · threshold {threshold:.4f}")

        with right:
            render_sar_card(str(alert.get("investigator_report", "—")))
            st.plotly_chart(
                make_driver_bar(alert.get("top_drivers", [])),
                use_container_width=True,
            )

# --------------------------------------------------------------------------- #
# Tab 2: Global SHAP & Ablation
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Global SHAP & Ablation")
    if not global_shap or not metrics:
        st.info("Global SHAP or metrics artifacts are missing. Run Phase 3 and Phase 4 first.")
    else:
        graph_feature_set = set(metrics.get("graph_features", []))
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(
                make_global_bar(global_shap, graph_feature_set),
                use_container_width=True,
            )
        with c_right:
            st.plotly_chart(
                make_ablation_bar(metrics),
                use_container_width=True,
            )
            lift = metrics.get("ablation", {}).get("graph_feature_auprc_lift")
            if lift is not None:
                st.caption(f"Graph feature PR-AUC lift vs tabular-only: {float(lift):+.4f}")

# --------------------------------------------------------------------------- #
# Tab 3: Live What-If Simulator
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Live What-If Simulator")
    st.caption(
        "Score a synthetic transaction through the exact same TreeSHAP kernel used "
        "in training-time explanations, then narrate it."
    )

    explainer: AMLTreeExplainer | None
    try:
        explainer = get_explainer()
        feature_names = explainer.feature_names
    except Exception as exc:  # noqa: BLE001 - explainer needs model artifacts
        st.error(
            f"**Explainer unavailable:** {exc}. Ensure `models/xgb_aml.json` exists "
            "(run `src/model.py`)."
        )
        explainer = None
        feature_names = []

    if explainer is not None:
        with st.form("whatif_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                tx_type = st.selectbox("Transaction Type", TRANSACTION_TYPES, index=0)
                amount = st.number_input(
                    "Amount", min_value=0.0, value=2_500_000.0, step=100_000.0, format="%.2f"
                )
            with f2:
                step = st.number_input(
                    "Step (hour bucket)", min_value=1, max_value=744, value=400, step=1
                )
                dest_in_degree = st.number_input(
                    "dest_in_degree", min_value=0.0, value=1.0, step=1.0
                )
            with f3:
                dest_pagerank = st.number_input(
                    "dest_pagerank", min_value=0.0, value=0.0, step=0.001, format="%.6f"
                )
                with st.expander("More graph features"):
                    orig_in_degree = st.number_input("orig_in_degree", min_value=0.0, value=0.0, step=1.0)
                    orig_out_degree = st.number_input("orig_out_degree", min_value=0.0, value=0.0, step=1.0)
                    orig_pagerank = st.number_input("orig_pagerank", min_value=0.0, value=0.0, step=0.001, format="%.6f")
                    orig_clustering = st.number_input("orig_clustering", min_value=0.0, value=0.0, step=0.001, format="%.6f")
                    dest_out_degree = st.number_input("dest_out_degree", min_value=0.0, value=0.0, step=1.0)
                    dest_clustering = st.number_input("dest_clustering", min_value=0.0, value=0.0, step=0.001, format="%.6f")

            submitted = st.form_submit_button("⚡ Score & Narrate", type="primary")

        if submitted:
            row = {
                "step": int(step),
                "type": tx_type,
                "amount": float(amount),
                "orig_in_degree": float(orig_in_degree),
                "orig_out_degree": float(orig_out_degree),
                "orig_pagerank": float(orig_pagerank),
                "orig_clustering": float(orig_clustering),
                "dest_in_degree": float(dest_in_degree),
                "dest_out_degree": float(dest_out_degree),
                "dest_pagerank": float(dest_pagerank),
                "dest_clustering": float(dest_clustering),
            }
            # Restrict to the model's exact schema (drops any extras defensively).
            row_df = pd.DataFrame([row])[[c for c in feature_names if c in row]]

            try:
                payload = explainer.explain_transaction(row_df)
                payload["investigator_report"] = generate_investigator_report(payload)
            except Exception as exc:  # noqa: BLE001 - surface scoring errors in UI
                st.error(f"Scoring failed: {exc}")
            else:
                res_l, res_r = st.columns([1, 3])
                with res_l:
                    render_badge(bool(payload.get("is_flagged", False)))
                    st.markdown("")
                    st.metric("Fraud Probability", f"{float(payload['predicted_probability']):.2%}")
                    st.metric("Raw Log-Odds", f"{float(payload['raw_log_odds']):+.4f}")
                    st.caption(
                        f"Base log-odds (φ₀): {float(payload['base_log_odds']):+.4f} · "
                        f"threshold {float(payload['threshold']):.4f}"
                    )
                with res_r:
                    render_sar_card(payload["investigator_report"])
                    st.plotly_chart(
                        make_driver_bar(payload.get("top_drivers", [])),
                        use_container_width=True,
                    )

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    "SafePay 2.0 · Provider-agnostic narratives (offline deterministic NLG by default). "
    "Attributions are exact TreeSHAP in log-odds space; local accuracy enforced."
)
