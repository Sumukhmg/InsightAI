"""InsightAI UI & Visualization Utilities
Enterprise design system tokens, CSS styling, and Plotly charting helpers.
"""

from typing import Any, Dict, List, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Enterprise Theme Palette
COLOR_PRIMARY = "#3B82F6"      # Indigo / Electric Blue
COLOR_SECONDARY = "#10B981"    # Emerald Green
COLOR_ACCENT = "#8B5CF6"       # Violet
COLOR_WARNING = "#F59E0B"      # Amber
COLOR_DANGER = "#EF4444"       # Crimson
COLOR_BG_CARD = "rgba(255, 255, 255, 0.05)"
COLOR_BORDER = "rgba(255, 255, 255, 0.12)"

PALETTE = [
    "#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899",
    "#06B6D4", "#6366F1", "#14B8A6", "#F97316", "#84CC16"
]


def inject_custom_css() -> None:
    """Injects modern, premium CSS styling for Streamlit."""
    custom_css = """
    <style>
    /* Global Typography & Variables */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .kpi-delta.positive { color: #10B981; }
    .kpi-delta.negative { color: #EF4444; }
    .kpi-delta.neutral { color: #94A3B8; }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-primary { background: rgba(59, 130, 246, 0.18); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-success { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-danger  { background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-purple  { background: rgba(139, 92, 246, 0.18); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }

    /* Insight Narrative Section Styling */
    .insight-fact {
        border-left: 3px solid #3B82F6;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    .insight-pattern {
        border-left: 3px solid #8B5CF6;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    .insight-hypothesis {
        border-left: 3px solid #F59E0B;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    .insight-rec {
        border-left: 3px solid #10B981;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    
    /* Code & Preformatted Text */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def format_currency(value: float, symbol: str = "£") -> str:
    """Formats a float as currency with comma separation or compact notation."""
    if value is None or (isinstance(value, float) and value != value):
        return f"{symbol}0.00"
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.1f}K"
    return f"{symbol}{value:,.2f}"


def format_number(value: float) -> str:
    """Formats large integer or float numbers into readable string."""
    if value is None:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value):,}" if value == int(value) else f"{value:.2f}"


def render_kpi(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "neutral",
    tooltip: Optional[str] = None
) -> None:
    """Renders a modern glassmorphic KPI card."""
    delta_html = ""
    if delta:
        arrow = "▲" if delta_type == "positive" else ("▼" if delta_type == "negative" else "•")
        delta_html = f'<div class="kpi-delta {delta_type}">{arrow} {delta}</div>'

    html = f"""
    <div class="kpi-card" title="{tooltip or ''}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Applies a consistent, clean enterprise theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=20, r=20, t=50 if title else 30, b=20),
        title=dict(
            text=title or "",
            font=dict(size=15, color="#F8FAFC", family="Inter, sans-serif")
        ) if title else None,
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, sans-serif"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="rgba(255,255,255,0.15)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="rgba(255,255,255,0.15)"
        ),
        colorway=PALETTE
    )
    return fig
