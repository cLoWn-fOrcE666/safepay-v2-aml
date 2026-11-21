"""SafePay 2.0: AML compliance workstation (Streamlit).

A review desk over the Phase 3 model, Phase 4 TreeSHAP explanations and
Phase 5 investigator narratives.

* Case Investigation          - work through the queue of flagged alerts.
* Model Validation & Graph Lift - ablation and global SHAP, grouped by signal family.
* What-If Sandbox             - score a synthetic transaction live and watch the
                                graph features move it across the alert threshold.

Data contracts (unchanged):
    models/metrics.json
    reports/global_shap_importance.json
    reports/sample_explanations.json
    src.explainer.AMLTreeExplainer / generate_investigator_report
"""

from __future__ import annotations

import html
import json
import math
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

# --------------------------------------------------------------------------- #
# Palette and constants
# --------------------------------------------------------------------------- #
BG = "#F8FAFC"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"
INK = "#0F172A"
MUTED = "#475569"
SOFT = "#94A3B8"
ACCENT = "#2563EB"
ACCENT_TINT = "#EFF6FF"
RISK = "#E11D48"
RISK_TINT = "#FFF1F2"
SAFE = "#0D9488"
SAFE_TINT = "#F0FDFA"
TRANSPARENT = "rgba(0,0,0,0)"
FONT_BODY = "Inter, system-ui, sans-serif"

TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
FILTER_ALL = "All alerts"
FILTER_EXTREME = "Extreme > 99.5%"
EXTREME_CUTOFF = 0.995
SPOTLIGHT_GRAPH = {"dest_in_degree", "dest_pagerank"}

# What-if scenario presets. Values are illustrative starting points: tune them
# against the deployed model once you have seen how each one scores.
PRESETS: dict[str, dict[str, Any]] = {
    "Normal Retail Payment": dict(
        type="PAYMENT", amount=84.50, step=14, dest_in_degree=1, dest_pagerank=0.0,
        orig_in_degree=0.0, orig_out_degree=2.0, orig_pagerank=0.0, orig_clustering=0.0,
        dest_out_degree=0.0, dest_clustering=0.0,
    ),
    "High-Value Cash-Out": dict(
        type="CASH_OUT", amount=650_000.00, step=391, dest_in_degree=1, dest_pagerank=0.0,
        orig_in_degree=0.0, orig_out_degree=1.0, orig_pagerank=0.0, orig_clustering=0.0,
        dest_out_degree=0.0, dest_clustering=0.0,
    ),
    "Mule Collector Hub": dict(
        type="TRANSFER", amount=980_000.00, step=215, dest_in_degree=64, dest_pagerank=0.00015,
        orig_in_degree=0.0, orig_out_degree=1.0, orig_pagerank=0.0, orig_clustering=0.0,
        dest_out_degree=2.0, dest_clustering=0.0,
    ),
}
DEFAULT_PRESET = "High-Value Cash-Out"
WI = {name: f"wi_{name}" for name in PRESETS[DEFAULT_PRESET]}  # preset field -> session key

# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="SafePay 2.0 | Compliance Workstation", layout="wide")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
  --bg: #F8FAFC; --card: #FFFFFF; --border: #E2E8F0;
  --ink: #0F172A; --muted: #475569; --soft: #64748B;
  --accent: #2563EB; --accent-tint: #EFF6FF;
  --risk: #E11D48; --risk-tint: #FFF1F2;
  --safe: #0D9488; --safe-tint: #F0FDFA;
  --shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  --font-ui: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif;
  --font-body: 'Inter', system-ui, sans-serif;
}

/* ---- base ---- */
.stApp { background: var(--bg); color: var(--ink); font-family: var(--font-body); }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp [data-testid="stCaptionContainer"] { font-family: var(--font-body); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: var(--font-ui); color: var(--ink); letter-spacing: -0.02em;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px; }
.num, .stApp input[type="number"] {
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
}

/* ---- masthead ---- */
.sp-masthead { margin-bottom: 1.4rem; }
.sp-title {
  font-family: var(--font-ui); font-weight: 800; font-size: 2rem;
  letter-spacing: -0.03em; color: var(--ink); margin: 0; line-height: 1.15;
}
.sp-title span { color: var(--accent); }
.sp-lede { color: var(--muted); font-size: 1rem; margin: 6px 0 0 0; max-width: 62ch; line-height: 1.55; }

/* ---- stat strip ---- */
.sp-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 0 0 1.6rem 0; }
.sp-stat {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  box-shadow: var(--shadow); padding: 16px 18px;
}
.sp-stat-label { color: var(--soft); font-size: 0.82rem; font-weight: 500; }
.sp-stat-value {
  font-family: var(--font-ui); font-weight: 700; font-size: 1.55rem; color: var(--ink);
  letter-spacing: -0.02em; margin-top: 4px; line-height: 1.2;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
}
.sp-stat-note { color: var(--muted); font-size: 0.8rem; margin-top: 4px; }
.sp-stat-note.up { color: var(--safe); font-weight: 600; }
.sp-stat-note.down { color: var(--risk); font-weight: 600; }
.sp-stat-value.small { font-size: 1.15rem; }

/* ---- cards ---- */
.sp-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  box-shadow: var(--shadow); padding: 22px 24px;
}
[class*="st-key-card_"] {
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  box-shadow: var(--shadow); padding: 20px 22px;
}
.sp-card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.sp-card-title { font-family: var(--font-ui); font-weight: 700; font-size: 1.1rem; color: var(--ink); margin: 0; letter-spacing: -0.01em; }
.sp-card-sub { color: var(--muted); font-size: 0.88rem; margin: 4px 0 0 0; line-height: 1.5; max-width: 62ch; }
.sp-bignum {
  font-family: var(--font-ui); font-weight: 800; font-size: 3rem; letter-spacing: -0.04em;
  color: var(--ink); line-height: 1; margin-top: 18px;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
}
.sp-bignum-label { color: var(--muted); font-size: 0.88rem; margin-top: 6px; }

/* ---- pills ---- */
.pill {
  display: inline-block; padding: 3px 11px; border-radius: 999px;
  font-family: var(--font-ui); font-weight: 600; font-size: 0.76rem; white-space: nowrap;
}
.pill-risk { background: var(--risk-tint); color: var(--risk); border: 1px solid #FECDD3; }
.pill-safe { background: var(--safe-tint); color: var(--safe); border: 1px solid #99F6E4; }
.pill-accent { background: var(--accent-tint); color: var(--accent); border: 1px solid #BFDBFE; }

/* ---- probability meter ---- */
.meter { position: relative; height: 10px; background: #F1F5F9; border-radius: 999px; margin: 22px 0 8px 0; }
.meter-fill { height: 100%; border-radius: 999px; }
.meter-fill.risk { background: var(--risk); }
.meter-fill.safe { background: var(--safe); }
.meter-tick { position: absolute; top: -5px; width: 2px; height: 20px; background: var(--ink); border-radius: 1px; }
.meter-scale { position: relative; height: 18px; font-size: 0.76rem; color: var(--soft); font-variant-numeric: tabular-nums; }
.meter-scale .l { position: absolute; left: 0; }
.meter-scale .r { position: absolute; right: 0; }
.meter-scale .t { position: absolute; transform: translateX(-50%); color: var(--ink); font-weight: 600; white-space: nowrap; }
.meter-foot { color: var(--soft); font-size: 0.74rem; margin-top: 2px; }

/* ---- key/value rows ---- */
.kv { margin-top: 14px; }
.kv-row { display: flex; justify-content: space-between; gap: 16px; padding: 9px 0; border-top: 1px solid var(--border); font-size: 0.9rem; }
.kv-row .k { color: var(--muted); }
.kv-row .v { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }

/* ---- SAR note ---- */
.sar {
  margin-top: 16px; background: var(--accent-tint); border: 1px solid #BFDBFE;
  border-left: 4px solid var(--accent); border-radius: 12px; padding: 18px 22px 16px 22px;
}
.sar-title { font-family: var(--font-ui); font-weight: 700; font-size: 0.98rem; color: var(--ink); margin: 0 0 8px 0; }
.sar-body { color: var(--ink); font-size: 0.95rem; line-height: 1.65; margin: 0; max-width: 70ch; }
.sar-foot { color: var(--muted); font-size: 0.78rem; margin-top: 12px; padding-top: 10px; border-top: 1px solid #BFDBFE; }

/* ---- driver rows ---- */
.drv { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 0; border-top: 1px solid var(--border); font-size: 0.9rem; }
.drv:first-child { border-top: none; }
.drv .name { font-weight: 600; color: var(--ink); }
.drv .val { color: var(--muted); font-variant-numeric: tabular-nums; }
.drv .shap { font-weight: 600; font-variant-numeric: tabular-nums; min-width: 64px; text-align: right; }
.drv .shap.up { color: var(--risk); }
.drv .shap.down { color: var(--safe); }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
.sp-side-brand { font-family: var(--font-ui); font-weight: 800; font-size: 1.2rem; letter-spacing: -0.03em; color: var(--ink); }
.sp-side-brand span { color: var(--accent); }
.sp-side-title { font-family: var(--font-ui); font-weight: 700; font-size: 0.98rem; color: var(--ink); margin: 18px 0 2px 0; }
.sp-side-sub { color: var(--muted); font-size: 0.82rem; margin-bottom: 10px; line-height: 1.45; }

/* filter as a segmented control */
.st-key-filter div[role="radiogroup"] { display: flex; gap: 0; background: #F1F5F9; border-radius: 10px; padding: 3px; }
.st-key-filter div[role="radiogroup"] > label { flex: 1; justify-content: center; margin: 0; padding: 6px 4px; border-radius: 8px; cursor: pointer; }
.st-key-filter div[role="radiogroup"] > label > div:first-child { display: none; }
.st-key-filter div[role="radiogroup"] > label p { font-family: var(--font-ui); font-weight: 600; font-size: 0.78rem; color: var(--muted); text-align: center; margin: 0; }
.st-key-filter div[role="radiogroup"] > label:has(input:checked) { background: var(--card); box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08); }
.st-key-filter div[role="radiogroup"] > label:has(input:checked) p { color: var(--ink); }

/* queue rows */
.st-key-queue div[role="radiogroup"] { gap: 6px; }
.st-key-queue div[role="radiogroup"] > label {
  width: 100%; margin: 0; padding: 10px 12px; cursor: pointer;
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
}
.st-key-queue div[role="radiogroup"] > label > div:first-child { display: none; }
.st-key-queue div[role="radiogroup"] > label:hover { border-color: #93C5FD; }
.st-key-queue div[role="radiogroup"] > label:has(input:checked) {
  background: var(--accent-tint); border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset;
}
.st-key-queue div[role="radiogroup"] > label p { margin: 0; font-size: 0.86rem; line-height: 1.45; font-variant-numeric: tabular-nums; }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { padding: 10px 0; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-family: var(--font-ui); font-weight: 600; font-size: 0.95rem; color: var(--soft); }
.stTabs [aria-selected="true"] p { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* ---- buttons and inputs ---- */
.stApp .stButton > button {
  font-family: var(--font-ui); font-weight: 600; font-size: 0.88rem; border-radius: 10px;
  border: 1px solid var(--border); background: var(--card); color: var(--ink);
  box-shadow: var(--shadow); padding: 0.55rem 0.9rem;
}
.stApp .stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stApp .stButton > button[kind="primary"] { background: var(--accent-tint); border: 1px solid var(--accent); color: var(--accent); }
.stApp div[data-baseweb="input"], .stApp div[data-baseweb="select"] > div { border-radius: 10px; }
.stApp div[data-testid="stNumberInput"] input { font-variant-numeric: tabular-nums; }
.sp-field-title { font-family: var(--font-ui); font-weight: 700; font-size: 0.95rem; color: var(--ink); margin: 4px 0 8px 0; }

/* ---- small screens ---- */
@media (max-width: 900px) { .sp-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
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


@st.cache_data(show_spinner=False, max_entries=512)
def score_scenario(_explainer: AMLTreeExplainer, row_items: tuple) -> dict[str, Any]:
    """Score one synthetic transaction. Cached so live reruns never repeat work
    (or repeat an LLM call when an API key is configured)."""
    row = dict(row_items)
    feature_names = _explainer.feature_names
    row_df = pd.DataFrame([row])[[c for c in feature_names if c in row]]
    # top_k = all features so the UI can split the score into signal families.
    payload = _explainer.explain_transaction(row_df, top_k=len(feature_names))
    report_input = {**payload, "top_drivers": payload["top_drivers"][:3]}
    payload["investigator_report"] = generate_investigator_report(report_input)
    return payload


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def compact_amount(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def collapse(markup: str) -> str:
    """Strip indentation/newlines so Markdown never mistakes HTML for a code block."""
    return "".join(line.strip() for line in markup.splitlines())


def html_block(markup: str) -> None:
    st.markdown(collapse(markup), unsafe_allow_html=True)


def show_chart(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure full-width, across Streamlit API generations."""
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
    except ValueError:
        major, minor = 1, 0
    config = {"displayModeBar": False}
    if (major, minor) >= (1, 50):
        st.plotly_chart(fig, width="stretch", theme=None, config=config, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key=key)


def pill(text: str, tone: str) -> str:
    return f'<span class="pill pill-{tone}">{esc(text)}</span>'


def stat_tile(label: str, value: str, note: str = "", tone: str = "", small: bool = False) -> str:
    note_html = f'<div class="sp-stat-note {tone}">{esc(note)}</div>' if note else ""
    size = " small" if small else ""
    return (
        f'<div class="sp-stat"><div class="sp-stat-label">{esc(label)}</div>'
        f'<div class="sp-stat-value{size}">{esc(value)}</div>{note_html}</div>'
    )


# --------------------------------------------------------------------------- #
# Plotly helpers
# --------------------------------------------------------------------------- #
def _layout(height: int, left: int = 10, right: int = 16, top: int = 10, bottom: int = 10) -> dict[str, Any]:
    return dict(
        template="plotly_white",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(family=FONT_BODY, color=INK, size=13),
        margin=dict(l=left, r=right, t=top, b=bottom),
        height=height,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#FFFFFF", bordercolor=BORDER,
            font=dict(family=FONT_BODY, color=INK, size=13),
        ),
    )


def make_driver_lollipop(drivers: list[dict[str, Any]], max_items: int | None = None) -> go.Figure:
    """Diverging lollipop of TreeSHAP drivers (rose raises risk, teal lowers it)."""
    items = sorted(drivers, key=lambda d: abs(float(d.get("shap_log_odds", 0.0))), reverse=True)
    if max_items:
        items = items[:max_items]

    labels = [f"<b>{esc(d['feature'])}</b><br>{esc(fmt_value(d.get('value')))}" for d in items]
    values = [float(d.get("shap_log_odds", 0.0)) for d in items]
    colors = [RISK if v > 0 else SAFE for v in values]
    reach = max([abs(v) for v in values] + [0.5]) * 1.4

    fig = go.Figure()
    fig.add_vrect(x0=0, x1=reach, fillcolor=RISK_TINT, opacity=0.65, line_width=0, layer="below")
    fig.add_vrect(x0=-reach, x1=0, fillcolor=SAFE_TINT, opacity=0.65, line_width=0, layer="below")
    for label, value, color in zip(labels, values, colors):
        fig.add_trace(go.Scatter(
            x=[0, value], y=[label, label], mode="lines",
            line=dict(color=color, width=4), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=values, y=labels, mode="markers+text",
        marker=dict(color=colors, size=17, line=dict(color="#FFFFFF", width=3)),
        text=[f"{v:+.3f}" for v in values],
        textposition=["middle right" if v >= 0 else "middle left" for v in values],
        textfont=dict(color=colors, size=13, family=FONT_BODY),
        cliponaxis=False,
        customdata=[
            [d["feature"], fmt_value(d.get("value")),
             "Raises risk" if v > 0 else "Lowers risk"]
            for d, v in zip(items, values)
        ],
        hovertemplate=(
            "<b>%{customdata[0]}</b> = %{customdata[1]}<br>"
            "%{x:+.4f} log-odds<br>%{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_width=1, line_color=SOFT)
    fig.add_annotation(x=-reach, y=1.0, xref="x", yref="paper", text="Lowers risk",
                       showarrow=False, xanchor="left", yanchor="bottom", yshift=4,
                       font=dict(color=SAFE, size=12))
    fig.add_annotation(x=reach, y=1.0, xref="x", yref="paper", text="Raises risk",
                       showarrow=False, xanchor="right", yanchor="bottom", yshift=4,
                       font=dict(color=RISK, size=12))
    fig.update_layout(**_layout(height=100 + 76 * len(items), left=10, right=24, top=34, bottom=10))
    fig.update_xaxes(range=[-reach, reach], title=dict(text="Contribution in log-odds", font=dict(size=12, color=MUTED)),
                     showgrid=False, zeroline=False, tickformat="+.1f", tickfont=dict(color=MUTED, size=12))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12.5))
    return fig


def make_ablation_chart(metrics: dict[str, Any]) -> tuple[go.Figure, float]:
    """Tabular-only vs full (unweighted) Test PR-AUC. Returns (figure, lift)."""
    ablation = metrics.get("ablation", {})
    spec = [("tabular_only", "Tabular only", SOFT), ("full_unweighted", "Tabular + graph", ACCENT)]
    values = [float(ablation.get(k, {}).get("test", {}).get("auprc", 0.0)) for k, _, _ in spec]
    iters = [ablation.get(k, {}).get("best_iteration", "n/a") for k, _, _ in spec]
    lift = values[1] - values[0]
    top = max(values + [0.01])

    fig = go.Figure(go.Bar(
        x=[label for _, label, _ in spec], y=values,
        marker=dict(color=[c for _, _, c in spec]),
        text=[f"{v:.4f}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(size=14, color=INK, family=FONT_BODY),
        width=0.46, customdata=[[i] for i in iters],
        hovertemplate="<b>%{x}</b><br>Test PR-AUC %{y:.4f}<br>Best iteration %{customdata[0]}<extra></extra>",
    ))
    try:  # rounded bar ends need a recent Plotly; fall back to square silently
        fig.update_traces(marker_cornerradius=8)
    except ValueError:
        pass
    fig.add_annotation(
        x=1, y=values[1], text=f"<b>{lift:+.4f}</b> PR-AUC lift", showarrow=False, yshift=44,
        bgcolor=ACCENT_TINT, bordercolor="#BFDBFE", borderwidth=1, borderpad=6,
        font=dict(color=ACCENT, size=13, family=FONT_BODY),
    )
    fig.update_layout(**_layout(height=380, left=10, right=10, top=20, bottom=10))
    fig.update_yaxes(range=[0, top * 1.38], gridcolor="#F1F5F9", zeroline=False,
                     title=dict(text="Test PR-AUC", font=dict(size=12, color=MUTED)),
                     tickfont=dict(color=MUTED, size=12))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=13))
    return fig, lift


def make_global_chart(importance: list[dict[str, Any]], graph_set: set[str]) -> tuple[go.Figure, float]:
    """Mean |SHAP| grouped by signal family. Returns (figure, graph share of total)."""
    rows = [(d["feature"], float(d["mean_abs_shap_log_odds"])) for d in importance]
    tx = sorted([r for r in rows if r[0] not in graph_set], key=lambda r: -r[1])
    gr = sorted([r for r in rows if r[0] in graph_set], key=lambda r: -r[1])
    order = [f for f, _ in tx] + [f for f, _ in gr]
    total = sum(v for _, v in rows) or 1.0
    share = sum(v for _, v in gr) / total
    reach = max([v for _, v in rows] + [0.01]) * 1.22

    fig = go.Figure()
    groups = (
        ("Transaction signals", tx, [SOFT] * len(tx)),
        ("Inductive graph topology", gr,
         [ACCENT if f in SPOTLIGHT_GRAPH else "#93C5FD" for f, _ in gr]),
    )
    for name, grp, colors in groups:
        if not grp:
            continue
        fig.add_trace(go.Bar(
            x=[v for _, v in grp], y=[f for f, _ in grp], orientation="h", name=name,
            marker=dict(color=colors), width=0.62,
            text=[f"{v:.3f}" for _, v in grp], textposition="outside", cliponaxis=False,
            textfont=dict(size=12, color=INK, family=FONT_BODY),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP| %{x:.4f} log-odds<extra>" + name + "</extra>",
        ))
    try:
        fig.update_traces(marker_cornerradius=6)
    except ValueError:
        pass
    fig.update_layout(**_layout(height=max(360, 40 * len(order) + 70), left=10, right=30, top=30, bottom=10))
    fig.update_layout(
        showlegend=True, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=12, color=MUTED)),
    )
    fig.update_xaxes(range=[0, reach], gridcolor="#F1F5F9", zeroline=False,
                     title=dict(text="Mean |SHAP| in log-odds", font=dict(size=12, color=MUTED)),
                     tickfont=dict(color=MUTED, size=12))
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed",
                     showgrid=False, tickfont=dict(size=12.5))
    return fig, share


def make_logodds_meter(raw: float, base: float, thr_logit: float, flagged: bool) -> go.Figure:
    """Number line: where the raw log-odds score sits against the alert threshold."""
    lo = min(raw, base, thr_logit) - 1.5
    hi = max(raw, base, thr_logit) + 1.5
    color = RISK if flagged else SAFE
    fig = go.Figure()
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=0.35, y1=0.65, fillcolor="#F1F5F9", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=min(base, raw), x1=max(base, raw), y0=0.35, y1=0.65,
                  fillcolor=color, opacity=0.9, line_width=0, layer="below")
    fig.add_shape(type="line", x0=thr_logit, x1=thr_logit, y0=0.1, y1=0.9,
                  line=dict(color=INK, width=2, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[raw], y=[0.5], mode="markers",
        marker=dict(color="#FFFFFF", size=18, line=dict(color=color, width=4)),
        hovertemplate=f"Score {raw:+.3f} log-odds<br>Threshold {thr_logit:+.3f}<br>Base {base:+.3f}<extra></extra>",
    ))
    fig.add_annotation(x=thr_logit, y=0.9, text=f"Alert threshold {thr_logit:+.2f}", showarrow=False,
                       yshift=12, font=dict(color=INK, size=12))
    fig.add_annotation(x=raw, y=0.35, text=f"<b>Score {raw:+.2f}</b>", showarrow=False,
                       yshift=-16, font=dict(color=color, size=13))
    fig.update_layout(**_layout(height=130, left=10, right=10, top=14, bottom=6))
    fig.update_xaxes(range=[lo, hi], showgrid=False, zeroline=False, tickformat="+.0f",
                     tickfont=dict(color=SOFT, size=11))
    fig.update_yaxes(range=[0, 1], visible=False)
    return fig


# --------------------------------------------------------------------------- #
# HTML components
# --------------------------------------------------------------------------- #
def risk_meter_html(prob: float, threshold: float, flagged: bool) -> str:
    lo = max(0.0, math.floor((min(prob, threshold) - 0.03) * 100) / 100)
    span = (1.0 - lo) or 1.0

    def pos(x: float) -> float:
        return max(0.0, min(1.0, (x - lo) / span)) * 100

    t_label = min(max(pos(threshold), 14.0), 86.0)
    tone = "risk" if flagged else "safe"
    return (
        f'<div class="meter"><div class="meter-fill {tone}" style="width:{pos(prob):.2f}%"></div>'
        f'<div class="meter-tick" style="left:{pos(threshold):.2f}%"></div></div>'
        f'<div class="meter-scale"><span class="l">{lo:.0%}</span>'
        f'<span class="t" style="left:{t_label:.2f}%">Threshold {threshold:.2%}</span>'
        f'<span class="r">100%</span></div>'
        f'<div class="meter-foot">Scale starts at {lo:.0%} so the gap to the threshold stays readable.</div>'
    )


def case_summary_html(title: str, alert: dict[str, Any]) -> str:
    prob = float(alert.get("predicted_probability", 0.0))
    raw = float(alert.get("raw_log_odds", 0.0))
    base = float(alert.get("base_log_odds", 0.0))
    threshold = float(alert.get("threshold", 0.0))
    flagged = bool(alert.get("is_flagged", False))
    margin_pts = (prob - threshold) * 100
    status = pill("Above alert threshold", "risk") if flagged else pill("Below alert threshold", "safe")
    return (
        '<div class="sp-card">'
        f'<div class="sp-card-head"><h3 class="sp-card-title">{esc(title)}</h3>{status}</div>'
        f'<div class="sp-bignum">{prob:.2%}</div>'
        '<div class="sp-bignum-label">Calibrated fraud probability</div>'
        f'{risk_meter_html(prob, threshold, flagged)}'
        '<div class="kv">'
        f'<div class="kv-row"><span class="k">Alert threshold</span><span class="v">{threshold:.2%}</span></div>'
        f'<div class="kv-row"><span class="k">Margin over threshold</span><span class="v">{margin_pts:+.2f} pts</span></div>'
        f'<div class="kv-row"><span class="k">Raw log-odds</span><span class="v">{raw:+.4f}</span></div>'
        f'<div class="kv-row"><span class="k">Base log-odds (φ₀)</span><span class="v">{base:+.4f}</span></div>'
        '</div></div>'
    )


def sar_note_html(report: str) -> str:
    return (
        '<div class="sar"><div class="sar-title">SAR compliance note</div>'
        f'<p class="sar-body">{esc(report)}</p>'
        '<div class="sar-foot">Drafted from exact TreeSHAP attributions in log-odds space. '
        'Review before filing.</div></div>'
    )


def driver_rows_html(drivers: list[dict[str, Any]]) -> str:
    rows = []
    for d in drivers:
        v = float(d.get("shap_log_odds", 0.0))
        up = d.get("direction") == "INCREASES_RISK" or v > 0
        rows.append(
            '<div class="drv"><div>'
            f'<div class="name">{esc(d.get("feature", ""))}</div>'
            f'<div class="val">{esc(fmt_value(d.get("value")))}</div></div>'
            f'{pill("Raises risk", "risk") if up else pill("Lowers risk", "safe")}'
            f'<div class="shap {"up" if up else "down"}">{v:+.3f}</div></div>'
        )
    return "".join(rows)


# --------------------------------------------------------------------------- #
# Load artifacts
# --------------------------------------------------------------------------- #
metrics = _load_with_error(METRICS_PATH, "Model metrics")
global_shap = _load_with_error(GLOBAL_SHAP_PATH, "Global SHAP importance")
sample_explanations = _load_with_error(SAMPLE_EXPL_PATH, "Sample explanations")

graph_feature_set: set[str] = set((metrics or {}).get("graph_features", []))


def feature_group(name: str) -> str:
    if graph_feature_set:
        return "graph" if name in graph_feature_set else "tx"
    return "graph" if name.startswith(("orig_", "dest_")) else "tx"


# --------------------------------------------------------------------------- #
# Sidebar: case navigator
# --------------------------------------------------------------------------- #
selected_idx: int | None = None
with st.sidebar:
    html_block('<div class="sp-side-brand">Safe<span>Pay</span> 2.0</div>')
    html_block(
        '<div class="sp-side-title">Case queue</div>'
        '<div class="sp-side-sub">Flagged alerts ranked by calibrated fraud probability. '
        'Pick a case to open it in Case Investigation.</div>'
    )
    if not sample_explanations:
        st.caption("No alerts loaded.")
    else:
        alert_filter = st.radio(
            "Queue filter", [FILTER_ALL, FILTER_EXTREME], horizontal=True,
            key="filter", label_visibility="collapsed",
        )
        total_alerts = len(sample_explanations)
        visible = [
            i for i, a in enumerate(sample_explanations)
            if alert_filter == FILTER_ALL
            or float(a.get("predicted_probability", 0.0)) > EXTREME_CUTOFF
        ]
        st.caption(f"Showing {len(visible)} of {total_alerts} alerts")

        def queue_label(i: int) -> str:
            a = sample_explanations[i]
            drivers = {d.get("feature"): d.get("value") for d in a.get("top_drivers", [])}
            meta = []
            if "type" in drivers:
                meta.append(str(drivers["type"]))
            if isinstance(drivers.get("amount"), (int, float)):
                meta.append(compact_amount(float(drivers["amount"])))
            p = float(a.get("predicted_probability", 0.0))
            tone = "red" if p > EXTREME_CUTOFF else "orange"
            head = f"**Alert {i + 1:02d}**   :{tone}-background[{p:.2%}]"
            return head + (f"  \n:gray[{'  ·  '.join(meta)}]" if meta else "")

        if not visible:
            st.info("No alerts match this filter.")
        else:
            with st.container(height=560, border=False):
                selected_idx = st.radio(
                    "Alerts", visible, format_func=queue_label,
                    key="queue", label_visibility="collapsed",
                )

# --------------------------------------------------------------------------- #
# Masthead + stat strip
# --------------------------------------------------------------------------- #
html_block(
    """
    <div class="sp-masthead">
      <h1 class="sp-title">Safe<span>Pay</span> 2.0</h1>
      <p class="sp-lede">Alert review for the compliance desk. Every score is explained with exact
      TreeSHAP attributions and checked against a locked test split.</p>
    </div>
    """
)

if metrics and sample_explanations is not None:
    ablation = metrics.get("ablation", {})
    full_auprc = float(ablation.get("full_unweighted", {}).get("test", {}).get("auprc", 0.0))
    tab_auprc = float(ablation.get("tabular_only", {}).get("test", {}).get("auprc", 0.0))
    delta = full_auprc - tab_auprc
    threshold_global = float(metrics.get("threshold", 0.0))

    top_graph_driver = "dest_in_degree"
    for entry in global_shap or []:
        if feature_group(entry.get("feature", "")) == "graph":
            top_graph_driver = entry["feature"]
            break

    html_block(
        '<div class="sp-stats">'
        + stat_tile("Best test PR-AUC", f"{full_auprc:.4f}", f"{delta:+.4f} vs tabular only",
                    "up" if delta >= 0 else "down")
        + stat_tile("Alert threshold", f"{threshold_global:.4f}", "F1-optimal on validation")
        + stat_tile("Top graph driver", top_graph_driver, "Highest mean |SHAP| graph feature", small=True)
        + stat_tile("Queued high-risk alerts", f"{len(sample_explanations)}", "In the current review queue")
        + "</div>"
    )

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab1, tab2, tab3 = st.tabs(["Case Investigation", "Model Validation & Graph Lift", "What-If Sandbox"])

# --------------------------------------------------------------------------- #
# Tab 1: Case Investigation
# --------------------------------------------------------------------------- #
with tab1:
    if not sample_explanations:
        st.info("No sample explanations available. Run Phase 4 (`src/explainer.py`).")
    elif selected_idx is None:
        st.info("No alerts match the current queue filter. Switch the filter in the sidebar.")
    else:
        alert = sample_explanations[selected_idx]
        drivers = alert.get("top_drivers", [])
        left, right = st.columns([5, 6], gap="large")

        with left:
            html_block(case_summary_html(f"Alert {selected_idx + 1:02d}", alert))
            html_block(sar_note_html(str(alert.get("investigator_report", "No narrative available."))))
            with st.expander("Copy note text"):
                st.code(str(alert.get("investigator_report", "")), language=None)

        with right:
            with st.container(key="card_drivers"):
                html_block(
                    '<h3 class="sp-card-title">Why this alert fired</h3>'
                    '<p class="sp-card-sub">The three strongest TreeSHAP contributions to this score, in '
                    'log-odds. Rose pushes toward fraud, teal pulls away from it.</p>'
                )
                show_chart(make_driver_lollipop(drivers), key=f"drivers_{selected_idx}")
                html_block(driver_rows_html(drivers))

# --------------------------------------------------------------------------- #
# Tab 2: Model Validation & Graph Lift
# --------------------------------------------------------------------------- #
with tab2:
    if not global_shap or not metrics:
        st.info("Global SHAP or metrics artifacts are missing. Run Phase 3 and Phase 4 first.")
    else:
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            with st.container(key="card_ablation"):
                html_block(
                    '<h3 class="sp-card-title">Does graph topology add lift?</h3>'
                    '<p class="sp-card-sub">Test PR-AUC for the tabular-only model against the same model with '
                    'inductive graph features, both trained without class weighting.</p>'
                )
                ablation_fig, lift_value = make_ablation_chart(metrics)
                show_chart(ablation_fig, key="ablation")
                stored_lift = metrics.get("ablation", {}).get("graph_feature_auprc_lift")
                note = (
                    f"The metrics file stores {float(stored_lift):+.4f} for the selected class-weighted model "
                    "against tabular only; the unweighted pair above isolates the effect of the graph features."
                ) if stored_lift is not None else ""
                html_block(f'<p class="sp-card-sub">{esc(note)}</p>')

        with c_right:
            with st.container(key="card_global"):
                global_fig, graph_share = make_global_chart(global_shap, graph_feature_set)
                html_block(
                    '<h3 class="sp-card-title">What the model leans on</h3>'
                    '<p class="sp-card-sub">Global mean |SHAP| by signal family. Graph topology carries '
                    f'{graph_share:.0%} of total attribution, led by dest_in_degree and dest_pagerank.</p>'
                )
                show_chart(global_fig, key="global_shap")

        test = metrics.get("test", {})
        selected_model = metrics.get("selected_model", "selected model")
        st.markdown("")
        html_block(
            f'<div class="sp-field-title">Operating point of the deployed model ({esc(selected_model)})</div>'
            '<div class="sp-stats">'
            + stat_tile("Precision", f"{float(test.get('precision', 0.0)):.2%}", "Share of alerts that are fraud")
            + stat_tile("Recall", f"{float(test.get('recall', 0.0)):.2%}", "Share of fraud caught")
            + stat_tile("Test alert volume", f"{int(test.get('alert_volume', 0)):,}", "Transactions flagged")
            + stat_tile("False-positive rate", f"{float(test.get('fpr', 0.0)):.4%}", "Legitimate flagged")
            + "</div>"
        )

# --------------------------------------------------------------------------- #
# Tab 3: What-If Sandbox
# --------------------------------------------------------------------------- #
def apply_preset(name: str) -> None:
    for field, value in PRESETS[name].items():
        st.session_state[WI[field]] = value
    st.session_state["wi_active"] = name


def clear_preset() -> None:
    st.session_state["wi_active"] = None


if "wi_active" not in st.session_state:
    apply_preset(DEFAULT_PRESET)

with tab3:
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

    html_block(
        '<h3 class="sp-card-title">Try a scenario</h3>'
        '<p class="sp-card-sub">Load a preset, then raise dest_in_degree or dest_pagerank and watch the score '
        'move toward the alert threshold. The result updates as you edit.</p>'
    )
    preset_cols = st.columns(len(PRESETS))
    for col, name in zip(preset_cols, PRESETS):
        col.button(
            name, key=f"preset_{name}", on_click=apply_preset, args=(name,),
            type="primary" if st.session_state.get("wi_active") == name else "secondary",
        )

    if explainer is not None:
        in_col, out_col = st.columns([5, 7], gap="large")

        with in_col:
            with st.container(key="card_inputs"):
                html_block('<div class="sp-field-title">Transaction signals</div>')
                a, b = st.columns(2)
                tx_type = a.selectbox("Transaction type", TRANSACTION_TYPES, key=WI["type"], on_change=clear_preset)
                step = b.number_input("Step (hour bucket)", min_value=1, max_value=744, step=1,
                                      key=WI["step"], on_change=clear_preset)
                amount = st.number_input("Amount", min_value=0.0, step=100_000.0, format="%.2f",
                                         key=WI["amount"], on_change=clear_preset)

                html_block('<div class="sp-field-title" style="margin-top:14px">Destination topology</div>')
                dest_in_degree = st.slider("dest_in_degree", min_value=0, max_value=200, step=1,
                                           key=WI["dest_in_degree"], on_change=clear_preset,
                                           help="Distinct senders paying this destination account.")
                dest_pagerank = st.number_input("dest_pagerank", min_value=0.0, step=0.00005, format="%.6f",
                                                key=WI["dest_pagerank"], on_change=clear_preset)

                with st.expander("Counterparty and remaining graph features"):
                    orig_in_degree = st.number_input("orig_in_degree", min_value=0.0, step=1.0,
                                                     key=WI["orig_in_degree"], on_change=clear_preset)
                    orig_out_degree = st.number_input("orig_out_degree", min_value=0.0, step=1.0,
                                                      key=WI["orig_out_degree"], on_change=clear_preset)
                    orig_pagerank = st.number_input("orig_pagerank", min_value=0.0, step=0.00005, format="%.6f",
                                                    key=WI["orig_pagerank"], on_change=clear_preset)
                    orig_clustering = st.number_input("orig_clustering", min_value=0.0, step=0.001, format="%.6f",
                                                      key=WI["orig_clustering"], on_change=clear_preset)
                    dest_out_degree = st.number_input("dest_out_degree", min_value=0.0, step=1.0,
                                                      key=WI["dest_out_degree"], on_change=clear_preset)
                    dest_clustering = st.number_input("dest_clustering", min_value=0.0, step=0.001, format="%.6f",
                                                      key=WI["dest_clustering"], on_change=clear_preset)

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

        with out_col:
            try:
                payload = score_scenario(explainer, tuple(sorted(row.items())))
            except Exception as exc:  # noqa: BLE001 - surface scoring errors in UI
                st.error(f"Scoring failed: {exc}")
            else:
                prob = float(payload["predicted_probability"])
                raw = float(payload["raw_log_odds"])
                base = float(payload["base_log_odds"])
                thr = float(payload["threshold"])
                flagged = bool(payload.get("is_flagged", False))
                thr_logit = logit(thr)
                all_drivers = payload.get("top_drivers", [])
                tx_sum = sum(float(d["shap_log_odds"]) for d in all_drivers if feature_group(d["feature"]) == "tx")
                gr_sum = sum(float(d["shap_log_odds"]) for d in all_drivers if feature_group(d["feature"]) == "graph")
                gap = raw - thr_logit

                with st.container(key="card_results"):
                    verdict = pill("Above alert threshold", "risk") if flagged else pill("Below alert threshold", "safe")
                    html_block(
                        f'<div class="sp-card-head"><h3 class="sp-card-title">Live score</h3>{verdict}</div>'
                        '<div class="sp-stats" style="grid-template-columns:repeat(3,minmax(0,1fr));margin:14px 0 4px 0">'
                        + stat_tile("Fraud probability", f"{prob:.2%}", f"Threshold {thr:.2%}")
                        + stat_tile("Gap to threshold", f"{gap:+.2f}", "log-odds", "down" if gap >= 0 else "up")
                        + stat_tile("Graph topology effect", f"{gr_sum:+.2f}", f"Transaction signals {tx_sum:+.2f}")
                        + "</div>"
                    )
                    show_chart(make_logodds_meter(raw, base, thr_logit, flagged), key="wi_meter")
                    html_block(
                        f'<p class="sp-card-sub">Score = base {base:+.2f} + transaction signals {tx_sum:+.2f} '
                        f'+ graph topology {gr_sum:+.2f}. Dotted line marks the alert threshold in log-odds.</p>'
                    )
                    html_block(sar_note_html(str(payload.get("investigator_report", ""))))

                with st.container(key="card_drivers_wi"):
                    html_block(
                        '<h3 class="sp-card-title">Strongest drivers right now</h3>'
                        '<p class="sp-card-sub">Top five TreeSHAP contributions for this scenario.</p>'
                    )
                    show_chart(make_driver_lollipop(all_drivers, max_items=5), key="wi_drivers")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown("")
st.caption(
    "SafePay 2.0. Narratives come from a deterministic offline template unless an LLM key is configured. "
    "Attributions are exact TreeSHAP in log-odds space with local accuracy enforced."
)
