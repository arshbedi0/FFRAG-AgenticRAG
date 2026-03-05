"""
metrics_dashboard.py
────────────────────
Interactive retrieval metrics dashboard for Streamlit UI
Displays Precision@K, Recall@K, MRR comparisons between baseline vs hybrid
"""

import streamlit as st
import json
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


def load_metrics() -> Dict[str, Any]:
    """Load retrieval metrics from JSON file."""
    metrics_file = Path("evaluation/retrieval_metrics.json")
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return None


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
            <div><b>Evaluation Date:</b> {metrics.get('eval_date', 'N/A')}</div>
            <div><b>Queries Evaluated:</b> {metrics.get('n_queries', 0)}</div>
            <div style="margin-top:12px;"><b>Key Findings:</b></div>
            <ul style="margin-top:6px;margin-bottom:0;">
                <li>Hybrid achieves <b>{hybrid.get('mean_precision', 0):.1%}</b> precision vs baseline <b>{baseline.get('mean_precision', 0):.1%}</b></li>
                <li>Recall improved by <b>{(hybrid.get('mean_recall', 0) - baseline.get('mean_recall', 0)):.1%}</b> absolute points</li>
                <li>MRR ranking improved by <b>{(hybrid.get('mrr', 0) - baseline.get('mrr', 0)):.1%}</b> (faster relevant results)</li>
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
    
    # ── KEY METRICS CARDS ──
    st.markdown("### 📈 Key Metrics Comparison")
    
    baseline = metrics.get("baseline", {})
    hybrid = metrics.get("hybrid", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card(
            "Precision@K",
            hybrid.get("mean_precision", 0),
            hybrid.get("mean_precision", 0) - baseline.get("mean_precision", 0)
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Recall@K",
            hybrid.get("mean_recall", 0),
            hybrid.get("mean_recall", 0) - baseline.get("mean_recall", 0)
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Mean Reciprocal Rank",
            hybrid.get("mrr", 0),
            hybrid.get("mrr", 0) - baseline.get("mrr", 0)
        ), unsafe_allow_html=True)
    
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
