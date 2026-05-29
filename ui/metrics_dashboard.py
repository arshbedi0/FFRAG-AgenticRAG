"""
metrics_dashboard.py
────────────────────
Interactive retrieval metrics dashboard for Streamlit UI
Displays Precision@K, Recall@K, MRR comparisons between baseline vs hybrid
"""

import streamlit as st
import json
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any


# ════════════════════════════════════════════════════════════════
# THEME COLORS — matching app.py dark blue scheme
# ════════════════════════════════════════════════════════════════
THEME = {
    "dark_bg": "#0a1020",
    "sidebar_bg": "#060b16",
    "border": "#1e3050",
    "accent": "#00d9ff",
    "success": "#00ff88",
    "warning": "#ffaa00",
    "error": "#ff4444",
    "text": "#c8d0e0",
    "text_muted": "#334455",
    "hybrid_color": "#00d9ff",
    "baseline_color": "#ff6b6b",
}

HISTORY_FILE = Path("evaluation/metrics_history.json")


def load_metrics() -> Dict[str, Any]:
    """Load retrieval metrics from JSON file."""
    metrics_file = Path("evaluation/retrieval_metrics.json")
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return None


def load_history() -> List[Dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_history(history: List[Dict[str, Any]]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


def sync_history(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append current metrics to history if changed."""
    history = load_history()
    hybrid = metrics.get("hybrid", {})
    baseline = metrics.get("baseline", {})

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "hybrid": {
            "precision": hybrid.get("mean_precision", 0),
            "recall": hybrid.get("mean_recall", 0),
            "mrr": hybrid.get("mrr", 0),
        },
        "baseline": {
            "precision": baseline.get("mean_precision", 0),
            "recall": baseline.get("mean_recall", 0),
            "mrr": baseline.get("mrr", 0),
        },
    }

    if not history or history[-1].get("hybrid") != entry["hybrid"]:
        history.append(entry)
        save_history(history)

    return history


def create_metric_card(label: str, value: float, delta: float = None, is_percentage: bool = True):
    """Create a styled metric card."""
    fmt = f"{value:.1%}" if is_percentage else f"{value:.4f}"
    delta_text = ""
    delta_color = THEME["success"]
    
    if delta is not None:
        delta_fmt = f"{delta:.1%}" if is_percentage else f"{delta:.4f}"
        delta_text = f"<span style='color:{delta_color};font-weight:bold'>↑ {delta_fmt}</span>"
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #0f1e38 0%, #1a2f52 100%);
        border-left: 3px solid {THEME['accent']};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    ">
        <div style="color:{THEME['text_muted']};font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">{label}</div>
        <div style="color:{THEME['accent']};font-size:28px;font-weight:bold;">{fmt}</div>
        <div style="color:{THEME['text_muted']};font-size:11px;margin-top:4px;">{delta_text}</div>
    </div>
    """


def create_animated_kpi(label: str, value: float, delta: float = None) -> str:
    pct = max(0.0, min(1.0, value))
    ring = int(pct * 100)
    delta_text = ""
    delta_class = "kpi-delta"
    if delta is not None:
        if delta >= 0:
            delta_text = f"+{delta:.1%}"
            delta_class = "kpi-delta kpi-delta-up"
        else:
            delta_text = f"{delta:.1%}"
            delta_class = "kpi-delta kpi-delta-down"
    else:
        delta_text = "Δ —"

    return (
        f"<div class=\"kpi-card\">"
        f"<div class=\"kpi-ring\" style=\"--p:{ring};\"></div>"
        f"<div class=\"{delta_class}\">{delta_text}</div>"
        f"<div class=\"kpi-label\">{label}</div>"
        f"<div class=\"kpi-value\">{pct:.1%}</div>"
        f"<div class=\"kpi-sweep\"></div>"
        f"</div>"
    )


def render_comparison_bar_chart(metrics: Dict) -> go.Figure:
    """Create comparison bars for Precision, Recall, MRR."""
    baseline = metrics.get("baseline", {})
    hybrid = metrics.get("hybrid", {})
    
    metrics_list = ["Precision@K", "Recall@K", "MRR"]
    baseline_vals = [
        baseline.get("mean_precision", 0),
        baseline.get("mean_recall", 0),
        baseline.get("mrr", 0),
    ]
    hybrid_vals = [
        hybrid.get("mean_precision", 0),
        hybrid.get("mean_recall", 0),
        hybrid.get("mrr", 0),
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            name="Baseline",
            x=metrics_list,
            y=baseline_vals,
            marker=dict(color=THEME["baseline_color"]),
            opacity=0.8,
            text=[f"{v:.1%}" for v in baseline_vals],
            textposition="auto",
        ),
        go.Bar(
            name="Hybrid",
            x=metrics_list,
            y=hybrid_vals,
            marker=dict(color=THEME["hybrid_color"]),
            opacity=0.8,
            text=[f"{v:.1%}" for v in hybrid_vals],
            textposition="auto",
        ),
    ])
    
    fig.update_layout(
        title="Retrieval Performance: Baseline vs Hybrid",
        barmode="group",
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(10, 16, 32, 0.5)",
        paper_bgcolor="rgba(10, 16, 32, 0.3)",
        font=dict(family="IBM Plex Sans, sans-serif", color=THEME["text"], size=12),
        xaxis=dict(showgrid=False, color=THEME["text_muted"]),
        yaxis=dict(showgrid=True, gridcolor="rgba(30, 48, 80, 0.3)", color=THEME["text_muted"]),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor=THEME["border"],
            borderwidth=1,
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400,
    )
    
    return fig


def render_per_query_performance(metrics: Dict) -> go.Figure:
    """Create scatter plot of per-query performance."""
    hybrid_queries = metrics.get("hybrid", {}).get("per_query", [])
    baseline_queries = metrics.get("baseline", {}).get("per_query", [])
    
    # Create mapping of baseline by query
    baseline_map = {q["query"]: q for q in baseline_queries}
    
    hybrid_precision = []
    hybrid_recall = []
    baseline_precision = []
    baseline_recall = []
    query_labels = []
    
    for hq in hybrid_queries:
        query = hq["query"][:40] + "..." if len(hq["query"]) > 40 else hq["query"]
        query_labels.append(query)
        hybrid_precision.append(hq["precision_at_k"])
        hybrid_recall.append(hq["recall_at_k"])
        
        if query in baseline_map or hq["query"] in baseline_map:
            bq = baseline_map.get(query, baseline_map.get(hq["query"], {}))
            baseline_precision.append(bq.get("precision_at_k", 0))
            baseline_recall.append(bq.get("recall_at_k", 0))
        else:
            baseline_precision.append(0)
            baseline_recall.append(0)
    
    x_pos = list(range(len(query_labels)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=baseline_precision,
        name="Baseline Precision",
        mode="lines+markers",
        line=dict(color=THEME["baseline_color"], width=2, dash="dash"),
        marker=dict(size=8),
        opacity=0.7,
    ))
    
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=hybrid_precision,
        name="Hybrid Precision",
        mode="lines+markers",
        line=dict(color=THEME["hybrid_color"], width=2),
        marker=dict(size=8),
        opacity=0.9,
    ))
    
    fig.update_layout(
        title="Per-Query Precision Performance",
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(10, 16, 32, 0.5)",
        paper_bgcolor="rgba(10, 16, 32, 0.3)",
        font=dict(family="IBM Plex Sans, sans-serif", color=THEME["text"], size=11),
        xaxis=dict(
            ticktext=["Q" + str(i+1) for i in range(len(query_labels))],
            tickvals=x_pos,
            showgrid=False,
            color=THEME["text_muted"],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(30, 48, 80, 0.3)",
            color=THEME["text_muted"],
            range=[0, 1.1],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor=THEME["border"],
            borderwidth=1,
        ),
        margin=dict(l=50, r=50, t=80, b=100),
        height=400,
    )
    
    return fig


def render_trend_chart(history: List[Dict[str, Any]]) -> go.Figure:
    if not history:
        return go.Figure()

    timestamps = [h.get("timestamp") for h in history]
    hybrid = [h.get("hybrid", {}) for h in history]
    baseline = [h.get("baseline", {}) for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[h.get("precision", 0) for h in hybrid],
        name="Hybrid Precision",
        mode="lines+markers",
        line=dict(color=THEME["hybrid_color"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[h.get("recall", 0) for h in hybrid],
        name="Hybrid Recall",
        mode="lines+markers",
        line=dict(color=THEME["success"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[h.get("mrr", 0) for h in hybrid],
        name="Hybrid MRR",
        mode="lines+markers",
        line=dict(color=THEME["warning"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[b.get("precision", 0) for b in baseline],
        name="Baseline Precision",
        mode="lines",
        line=dict(color=THEME["baseline_color"], width=1, dash="dot"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[b.get("recall", 0) for b in baseline],
        name="Baseline Recall",
        mode="lines",
        line=dict(color="#ffb3b3", width=1, dash="dot"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[b.get("mrr", 0) for b in baseline],
        name="Baseline MRR",
        mode="lines",
        line=dict(color="#ffc38f", width=1, dash="dot"),
        opacity=0.6,
    ))

    fig.update_layout(
        title="Metric Trends Over Time",
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(10, 16, 32, 0.5)",
        paper_bgcolor="rgba(10, 16, 32, 0.3)",
        font=dict(family="IBM Plex Sans, sans-serif", color=THEME["text"], size=11),
        xaxis=dict(showgrid=False, color=THEME["text_muted"]),
        yaxis=dict(showgrid=True, gridcolor="rgba(30, 48, 80, 0.3)", color=THEME["text_muted"], range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=100),
        height=420,
    )

    return fig


def render_mrr_interpretation(mrr: float) -> str:
    """Render MRR interpretation text."""
    avg_rank = 1.0 / mrr if mrr > 0 else float('inf')
    
    if mrr >= 0.8:
        quality = "✅ Excellent"
        color = THEME["success"]
        interpretation = "First relevant result typically within top 2 results"
    elif mrr >= 0.6:
        quality = "🟡 Good"
        color = THEME["warning"]
        interpretation = "First relevant result typically within top 3 results"
    else:
        quality = "⚠️ Needs Improvement"
        color = THEME["error"]
        interpretation = "First relevant result may be beyond top 5"
    
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
        border: 1px solid {THEME['accent']};
        padding: 16px;
        border-radius: 8px;
        margin-top: 8px;
    ">
        <div style="color:{color};font-size:14px;font-weight:bold;margin-bottom:8px;">{quality}</div>
        <div style="color:{THEME['text']};font-size:12px;line-height:1.6;">
            <b>MRR = {mrr:.4f}</b> → First relevant chunk appears at average rank <b>{avg_rank:.1f}</b><br>
            {interpretation}
        </div>
    </div>
    """


def render_metrics_table(metrics: Dict) -> pd.DataFrame:
    """Create detailed per-query metrics table."""
    hybrid_queries = metrics.get("hybrid", {}).get("per_query", [])
    
    data = []
    for i, q in enumerate(hybrid_queries, 1):
        data.append({
            "Query #": i,
            "Query": q["query"][:35] + "..." if len(q["query"]) > 35 else q["query"],
            "P@K": f"{q['precision_at_k']:.1%}",
            "R@K": f"{q['recall_at_k']:.1%}",
            "MRR": f"{q['reciprocal_rank']:.2f}",
            "Relevant": q["total_relevant"],
        })
    
    return pd.DataFrame(data)


def render_summary_section(metrics: Dict):
    """Render comprehensive summary section."""
    baseline = metrics.get("baseline", {})
    hybrid = metrics.get("hybrid", {})
    precision_delta = hybrid.get("mean_precision", 0) - baseline.get("mean_precision", 0)
    recall_delta = hybrid.get("mean_recall", 0) - baseline.get("mean_recall", 0)
    mrr_delta = hybrid.get("mrr", 0) - baseline.get("mrr", 0)
    eval_date = metrics.get("eval_date") or "Latest run"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a2f52 0%, #0f1e38 100%);
        border: 1px solid {THEME['accent']};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <div style="color:{THEME['accent']};font-size:16px;font-weight:bold;margin-bottom:16px;">📊 Evaluation Summary</div>
        <div style="color:{THEME['text']};font-size:12px;line-height:1.8;">
            <div><b>Evaluation Date:</b> {eval_date}</div>
            <div><b>Queries Evaluated:</b> {metrics.get('n_queries', 0)}</div>
            <div style="margin-top:12px;"><b>Key Findings:</b></div>
            <ul style="margin-top:6px;margin-bottom:0;">
                <li>Precision@K <b>{hybrid.get('mean_precision', 0):.1%}</b> vs baseline <b>{baseline.get('mean_precision', 0):.1%}</b> (Δ {precision_delta:+.1%})</li>
                <li>Recall@K <b>{hybrid.get('mean_recall', 0):.1%}</b> vs baseline <b>{baseline.get('mean_recall', 0):.1%}</b> (Δ {recall_delta:+.1%})</li>
                <li>MRR <b>{hybrid.get('mrr', 0):.1%}</b> vs baseline <b>{baseline.get('mrr', 0):.1%}</b> (Δ {mrr_delta:+.1%})</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard_content():
    """Dashboard content rendering (without page config)."""
    metrics = load_metrics()
    
    if metrics is None:
        st.error("❌ No metrics file found. Run evaluation first: `python evaluation/eval_retrieval_metrics.py`")
        return
    
    # ── HEADER ──
    st.markdown("""
    <div style="
        border-bottom: 2px solid #1e3050;
        padding-bottom: 16px;
        margin-bottom: 24px;
    ">
        <div style="color:#00d9ff;font-size:28px;font-weight:bold;">📊 Retrieval Metrics Dashboard</div>
        <div style="color:#334455;font-size:13px;margin-top:4px;">Performance analysis: Hybrid vs Baseline retriever</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── SUMMARY SECTION ──
    render_summary_section(metrics)

    history = sync_history(metrics)
    
    # ── ANIMATED KPIS ──
    st.markdown("### ⚡ Live Metrics Pulse")

    hybrid = metrics.get("hybrid", {})
    baseline = metrics.get("baseline", {})

    prev = history[-2] if len(history) >= 2 else None
    delta_precision = None
    delta_recall = None
    delta_mrr = None
    if prev:
        delta_precision = hybrid.get("mean_precision", 0) - prev.get("hybrid", {}).get("precision", 0)
        delta_recall = hybrid.get("mean_recall", 0) - prev.get("hybrid", {}).get("recall", 0)
        delta_mrr = hybrid.get("mrr", 0) - prev.get("hybrid", {}).get("mrr", 0)
    else:
        delta_precision = hybrid.get("mean_precision", 0) - baseline.get("mean_precision", 0)
        delta_recall = hybrid.get("mean_recall", 0) - baseline.get("mean_recall", 0)
        delta_mrr = hybrid.get("mrr", 0) - baseline.get("mrr", 0)

    st.markdown("""
    <style>
    .kpi-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .kpi-card {
        position: relative;
        background: radial-gradient(120px 120px at 20% 20%, rgba(0,217,255,0.12), rgba(10,16,32,0.6));
        border: 1px solid rgba(0,217,255,0.2);
        border-radius: 14px;
        padding: 16px 16px 14px 16px;
        overflow: hidden;
        min-height: 150px;
    }
    .kpi-ring {
        width: 92px;
        height: 92px;
        border-radius: 50%;
        background: conic-gradient(#00d9ff calc(var(--p)*1%), rgba(15,30,56,0.8) 0);
        mask: radial-gradient(farthest-side, transparent 62%, #000 63%);
        animation: pulse 2.2s ease-in-out infinite;
        margin-bottom: 6px;
    }
    .kpi-sweep {
        position: absolute;
        top: -20%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, rgba(0,217,255,0) 0deg, rgba(0,217,255,0.12) 50deg, rgba(0,217,255,0) 110deg);
        animation: sweep 3.5s linear infinite;
        pointer-events: none;
    }
    .kpi-label { color: #6c88aa; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
    .kpi-value { color: #c8d0e0; font-size: 22px; font-weight: 700; }
    .kpi-delta {
        position: absolute;
        top: 12px;
        right: 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        background: rgba(6, 12, 24, 0.7);
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .kpi-delta-up { color: #3ddc84; text-shadow: 0 0 8px rgba(61,220,132,0.5); }
    .kpi-delta-down { color: #ff6b6b; text-shadow: 0 0 8px rgba(255,107,107,0.5); }
    @keyframes sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.05); opacity: 0.85; } }
    @media (max-width: 900px) { .kpi-grid { grid-template-columns: 1fr; } }
    </style>
    """, unsafe_allow_html=True)

    kpi_html = "".join([
        create_animated_kpi("Precision@K", hybrid.get("mean_precision", 0), delta_precision),
        create_animated_kpi("Recall@K", hybrid.get("mean_recall", 0), delta_recall),
        create_animated_kpi("MRR", hybrid.get("mrr", 0), delta_mrr),
    ])
    st.markdown(
        f"<div class='kpi-grid'>{kpi_html}</div>",
        unsafe_allow_html=True,
    )

    # ── MRR INTERPRETATION ──
    st.markdown(render_mrr_interpretation(hybrid.get("mrr", 0)), unsafe_allow_html=True)
    
    # ── CHARTS ──
    st.markdown("### 📊 Visualizations")
    
    col_chart1 = st.container()
    fig_comparison = render_comparison_bar_chart(metrics)
    col_chart1.plotly_chart(fig_comparison, use_container_width=True)
    
    col_chart2 = st.container()
    fig_per_query = render_per_query_performance(metrics)
    col_chart2.plotly_chart(fig_per_query, use_container_width=True)

    st.markdown("### 🧭 Trendline")
    fig_trend = render_trend_chart(history)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # ── DETAILED TABLE ──
    st.markdown("### 📋 Per-Query Breakdown (Hybrid Retriever)")
    metrics_df = render_metrics_table(metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # ── INSIGHTS ──
    st.markdown("### 💡 Insights")
    
    best_query_idx = max(range(len(hybrid.get("per_query", []))), 
                         key=lambda i: hybrid["per_query"][i]["precision_at_k"]+hybrid["per_query"][i]["recall_at_k"])
    worst_query_idx = min(range(len(hybrid.get("per_query", []))), 
                          key=lambda i: hybrid["per_query"][i]["precision_at_k"]+hybrid["per_query"][i]["recall_at_k"])
    
    best_q = hybrid["per_query"][best_query_idx] if hybrid.get("per_query") else {}
    worst_q = hybrid["per_query"][worst_query_idx] if hybrid.get("per_query") else {}
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%);
            border-left: 3px solid {THEME['success']};
            padding: 16px;
            border-radius: 8px;
        ">
            <div style="color:{THEME['success']};font-weight:bold;margin-bottom:8px;">✅ Best Performing Query</div>
            <div style="color:{THEME['text']};font-size:12px;margin-bottom:8px;"><i>{best_q.get('query', 'N/A')[:60]}</i></div>
            <div style="color:{THEME['text_muted']};font-size:11px;">
                Precision: {best_q.get('precision_at_k', 0):.1%} | Recall: {best_q.get('recall_at_k', 0):.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255, 170, 0, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%);
            border-left: 3px solid {THEME['warning']};
            padding: 16px;
            border-radius: 8px;
        ">
            <div style="color:{THEME['warning']};font-weight:bold;margin-bottom:8px;">⚠️ Needs Improvement</div>
            <div style="color:{THEME['text']};font-size:12px;margin-bottom:8px;"><i>{worst_q.get('query', 'N/A')[:60]}</i></div>
            <div style="color:{THEME['text_muted']};font-size:11px;">
                Precision: {worst_q.get('precision_at_k', 0):.1%} | Recall: {worst_q.get('recall_at_k', 0):.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_dashboard():
    """Main dashboard rendering function (standalone mode)."""
    st.set_page_config(page_title="FFRAG Metrics Dashboard", layout="wide")
    render_dashboard_content()


if __name__ == "__main__":
    render_dashboard()
