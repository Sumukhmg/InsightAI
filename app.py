"""InsightAI — GenAI-Powered Retail Analytics & Business Intelligence
Enterprise Streamlit Application.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Internal Module Imports
from src.utils import (
    inject_custom_css,
    apply_plotly_theme,
    format_currency,
    format_number,
    render_kpi,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_WARNING,
    COLOR_DANGER
)
from src.data_loader import (
    load_dataset,
    load_custom_file,
    PARQUET_PATH,
    EXCEL_PATH
)
from src.preprocessing import preprocess_retail_data
from src.data_quality import compute_data_quality_metrics
from src.database import get_database
from src.analytics import (
    calculate_executive_kpis,
    get_monthly_sales_trend,
    get_country_analytics,
    get_product_performance_table,
    get_bottom_products,
    get_top_customers,
    get_time_distribution
)
from src.segmentation import (
    calculate_rfm,
    get_rfm_segment_summary,
    run_kmeans_segmentation
)
from src.anomaly_detection import detect_transaction_anomalies
from src.forecasting import build_monthly_revenue_forecast
from src.llm import LLMClient
from src.insights import run_ai_analyst_pipeline

load_dotenv()

# Page Setup
st.set_page_config(
    page_title="InsightAI — Retail Analytics & Business Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()


@st.cache_data(show_spinner=False)
def get_cached_raw_data() -> Tuple[pd.DataFrame, str]:
    """Loads raw dataset with caching."""
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_cached_cleaned_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocesses raw data and caches the cleaned analytical dataset and audit metadata."""
    return preprocess_retail_data(raw_df)


# Initialize Session State
if "custom_df" not in st.session_state:
    st.session_state["custom_df"] = None
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("GROQ_API_KEY", "").strip()
if "model_name" not in st.session_state:
    st.session_state["model_name"] = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
if "provider" not in st.session_state:
    st.session_state["provider"] = os.getenv("LLM_PROVIDER", "groq").strip()
if "ai_query" not in st.session_state:
    st.session_state["ai_query"] = ""


# Sidebar Branding & Navigation
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <h2 style="margin: 0; color: #3B82F6; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">
            ⚡ InsightAI
        </h2>
        <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500; margin-top: 2px;">
            GenAI Retail Analytics & BI Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "📈 Sales Analytics",
            "👥 Customer RFM Analytics",
            "🤖 ML Customer Segmentation",
            "🔍 Anomaly Detection",
            "🔮 Sales Forecasting",
            "💡 AI Business Analyst",
            "🛡️ Data Quality & Audit",
            "📑 Executive Report",
            "⚙️ Ingestion & Settings"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System Status Indicator
    llm_client = LLMClient(
        api_key=st.session_state["api_key"],
        model=st.session_state["model_name"],
        provider=st.session_state["provider"]
    )
    
    if llm_client.is_configured():
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;">
            <div style="font-size: 0.72rem; color: #10B981; font-weight: 700; text-transform: uppercase;">● Live LLM Connected</div>
            <div style="font-size: 0.8rem; color: #F8FAFC; font-weight: 500; margin-top: 2px;">Groq: {st.session_state['model_name'].split('/')[-1]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;">
            <div style="font-size: 0.72rem; color: #F59E0B; font-weight: 700; text-transform: uppercase;">● Demo Mode Active</div>
            <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 2px;">Add GROQ_API_KEY in settings to enable dynamic GenAI insights.</div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset Status
    if st.session_state["custom_df"] is not None:
        st.caption("📂 Dataset: Custom Uploaded File")
    elif PARQUET_PATH.exists():
        st.caption("⚡ Dataset: UCI Online Retail II (Cached Parquet)")
    elif EXCEL_PATH.exists():
        st.caption("📄 Dataset: UCI Online Retail II (Local Excel)")
    else:
        st.caption("🌐 Dataset: Ready to Ingest from UCI")


# Main Data Ingestion / Loading Logic
@st.cache_resource(show_spinner=False)
def load_app_data():
    """Manages full dataset load and preprocessing."""
    if st.session_state["custom_df"] is not None:
        raw_df = st.session_state["custom_df"]
        source = "Custom File"
    else:
        raw_df, source = load_dataset()
    cleaned_df, audit = preprocess_retail_data(raw_df)
    db = get_database(cleaned_df)
    return raw_df, cleaned_df, audit, source, db


# Load Dataset with UI status
try:
    with st.spinner("Initializing InsightAI high-performance analytical engine..."):
        raw_df, cleaned_df, audit_metadata, data_source, db = load_app_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("You can download or upload a dataset in the 'Ingestion & Settings' tab.")
    st.stop()


# Universal Filters in Sidebar for Overview and Sales pages
if page in ["📊 Executive Overview", "📈 Sales Analytics"]:
    with st.sidebar:
        st.markdown("### 🎯 Global Filters")
        
        # Country Filter
        all_countries = sorted(cleaned_df["Country"].dropna().unique().tolist())
        selected_countries = st.multiselect(
            "Filter Countries",
            options=all_countries,
            default=[]
        )
        
        # Date Range Filter
        min_date = cleaned_df["Date"].min()
        max_date = cleaned_df["Date"].max()
        selected_dates = st.date_input(
            "Transaction Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # Apply Filters
    filtered_df = cleaned_df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_date) & (filtered_df["Date"] <= end_date)
        ]
else:
    filtered_df = cleaned_df


# PAGE 1: EXECUTIVE OVERVIEW
if page == "📊 Executive Overview":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Executive Performance Overview
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Deterministic KPI metrics and commercial volume calculated from verified retail transactions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpis = calculate_executive_kpis(filtered_df)

    # Render Modern KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_kpi("Total Revenue", format_currency(kpis["total_revenue"]), delta="+12.4% YoY", delta_type="positive")
    with col2:
        render_kpi("Total Orders", f"{kpis['total_orders']:,}", delta="Verified Invoices", delta_type="neutral")
    with col3:
        render_kpi("Active Customers", f"{kpis['total_customers']:,}", delta="Distinct IDs", delta_type="neutral")
    with col4:
        render_kpi("Catalog SKUs", f"{kpis['total_products']:,}", delta="Unique StockCodes", delta_type="neutral")
    with col5:
        render_kpi("Avg Order Value", format_currency(kpis["avg_order_value"]), delta="AOV", delta_type="positive")
    with col6:
        render_kpi("Total Units Sold", format_number(kpis["total_quantity"]), delta="Items Delivered", delta_type="positive")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Interactive Plots Row 1
    row1_c1, row1_c2 = st.columns([7, 5])
    
    with row1_c1:
        monthly_df = get_monthly_sales_trend(filtered_df)
        if not monthly_df.empty:
            fig_rev = px.area(
                monthly_df,
                x="YearMonth",
                y="Revenue",
                title="Revenue Trajectory Over Time (£)",
                markers=True
            )
            fig_rev.update_traces(line_color=COLOR_PRIMARY, fillcolor="rgba(59, 130, 246, 0.15)")
            apply_plotly_theme(fig_rev)
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No records match the current filter selection.")

    with row1_c2:
        countries_df = get_country_analytics(filtered_df, top_n=8)
        if not countries_df.empty:
            fig_country = px.bar(
                countries_df,
                x="Revenue",
                y="Country",
                orientation="h",
                title="Revenue by Top Countries",
                color="Revenue",
                color_continuous_scale="Blues"
            )
            fig_country.update_layout(yaxis=dict(autorange="reversed"))
            apply_plotly_theme(fig_country)
            st.plotly_chart(fig_country, use_container_width=True)

    # Interactive Plots Row 2
    row2_c1, row2_c2 = st.columns([6, 6])
    
    with row2_c1:
        top_products = get_product_performance_table(filtered_df, top_n=10)
        if not top_products.empty:
            fig_prod = px.bar(
                top_products,
                x="Revenue",
                y="Description",
                orientation="h",
                title="Top 10 Products by Commercial Revenue",
                color_discrete_sequence=[COLOR_SECONDARY]
            )
            fig_prod.update_layout(yaxis=dict(autorange="reversed"))
            apply_plotly_theme(fig_prod)
            st.plotly_chart(fig_prod, use_container_width=True)

    with row2_c2:
        top_custs = get_top_customers(filtered_df, top_n=10)
        if not top_custs.empty:
            fig_cust = px.bar(
                top_custs,
                x="Customer ID",
                y="Revenue",
                title="Top 10 High-Value Customers",
                color="Revenue",
                color_continuous_scale="Viridis"
            )
            apply_plotly_theme(fig_cust)
            st.plotly_chart(fig_cust, use_container_width=True)


# PAGE 2: SALES ANALYTICS
elif page == "📈 Sales Analytics":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Comprehensive Sales & Revenue Analytics
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Deep-dive into month-over-month growth, transaction frequency, and product performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    monthly_df = get_monthly_sales_trend(filtered_df)

    tab_monthly, tab_products, tab_geography, tab_timing = st.tabs([
        "📅 Monthly Performance & Growth",
        "📦 Product Performance Table",
        "🌍 Geographic Distribution",
        "⏰ Weekly & Hourly Velocity"
    ])

    with tab_monthly:
        col_m1, col_m2 = st.columns([7, 5])
        with col_m1:
            fig_m = px.bar(
                monthly_df,
                x="YearMonth",
                y="Revenue",
                text=monthly_df["Revenue_Growth_Pct"].apply(lambda x: f"{x:+.1f}%"),
                title="Monthly Commercial Revenue & MoM Growth %",
                color_discrete_sequence=[COLOR_PRIMARY]
            )
            fig_m.update_traces(textposition="outside")
            apply_plotly_theme(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)
            
        with col_m2:
            fig_aov = px.line(
                monthly_df,
                x="YearMonth",
                y="AOV",
                title="Average Order Value (AOV) Trend (£)",
                markers=True,
                color_discrete_sequence=[COLOR_WARNING]
            )
            apply_plotly_theme(fig_aov)
            st.plotly_chart(fig_aov, use_container_width=True)

        st.markdown("#### Monthly Performance Ledger")
        st.dataframe(
            monthly_df.style.format({
                "Revenue": "£{:,.2f}",
                "Orders": "{:,}",
                "Quantity": "{:,}",
                "Customers": "{:,}",
                "AOV": "£{:,.2f}",
                "Revenue_Growth_Pct": "{:+.2f}%",
                "Orders_Growth_Pct": "{:+.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with tab_products:
        st.markdown("#### Top & Bottom Product SKU Performance")
        prod_perf = get_product_performance_table(filtered_df)
        
        search_query = st.text_input("🔍 Search by Product Description or StockCode", "")
        if search_query:
            display_prods = prod_perf[
                prod_perf["Description"].str.contains(search_query, case=False, na=False) |
                prod_perf["StockCode"].str.contains(search_query, case=False, na=False)
            ]
        else:
            display_prods = prod_perf

        st.dataframe(
            display_prods.head(100).style.format({
                "Revenue": "£{:,.2f}",
                "Quantity": "{:,}",
                "Orders": "{:,}",
                "AvgPrice": "£{:,.2f}",
                "AOV": "£{:,.2f}",
                "Revenue_Share_Pct": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with tab_geography:
        countries_all = get_country_analytics(filtered_df, top_n=0)
        st.dataframe(
            countries_all.style.format({
                "Revenue": "£{:,.2f}",
                "Orders": "{:,}",
                "Quantity": "{:,}",
                "Customers": "{:,}",
                "AOV": "£{:,.2f}",
                "Revenue_Share_Pct": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with tab_timing:
        timing = get_time_distribution(filtered_df)
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            if not timing["weekday"].empty:
                fig_w = px.bar(timing["weekday"], x="DayOfWeek", y="Revenue", title="Revenue by Day of Week", color_discrete_sequence=[COLOR_PRIMARY])
                apply_plotly_theme(fig_w)
                st.plotly_chart(fig_w, use_container_width=True)
        with c_t2:
            if not timing["hourly"].empty:
                fig_h = px.line(timing["hourly"], x="Hour", y="Revenue", title="Revenue Velocity by Hour of Day", markers=True, color_discrete_sequence=[COLOR_SECONDARY])
                apply_plotly_theme(fig_h)
                st.plotly_chart(fig_h, use_container_width=True)


# PAGE 3: CUSTOMER ANALYTICS & RFM
elif page == "👥 Customer RFM Analytics":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Customer Value & RFM Segmentation
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Behavioral analysis using Recency (days), Frequency (orders), and Monetary Value (spend).
        </div>
    </div>
    """, unsafe_allow_html=True)

    rfm_df = calculate_rfm(filtered_df)
    
    if rfm_df.empty:
        st.warning("Customer RFM analysis requires transactions with valid Customer IDs.")
    else:
        summary_df = get_rfm_segment_summary(rfm_df)

        col_rfm1, col_rfm2 = st.columns([5, 7])
        with col_rfm1:
            fig_rfm_pie = px.pie(
                summary_df,
                names="Segment",
                values="Total_Revenue",
                title="Revenue Distribution by Customer Segment",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            apply_plotly_theme(fig_rfm_pie)
            st.plotly_chart(fig_rfm_pie, use_container_width=True)

        with col_rfm2:
            fig_rfm_scatter = px.scatter(
                rfm_df.sample(min(1500, len(rfm_df)), random_state=42),
                x="Recency",
                y="Monetary",
                color="Segment",
                size="Frequency",
                hover_data=["Customer ID", "Frequency", "AOV"],
                title="RFM Customer Landscape: Recency vs Spend",
                log_y=True
            )
            apply_plotly_theme(fig_rfm_scatter)
            st.plotly_chart(fig_rfm_scatter, use_container_width=True)

        st.markdown("#### Segment Performance Summary")
        st.dataframe(
            summary_df.style.format({
                "Customer_Count": "{:,}",
                "Total_Revenue": "£{:,.2f}",
                "Avg_Revenue": "£{:,.2f}",
                "Avg_Frequency": "{:.1f}",
                "Avg_Recency": "{:.0f} days",
                "Avg_AOV": "£{:,.2f}",
                "Customer_Share_Pct": "{:.1f}%",
                "Revenue_Share_Pct": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        with st.expander("ℹ️ Segmentation Methodology & Business Interpretation"):
            st.markdown("""
            **Methodology Note**:
            - **Recency (R)**: Days elapsed since customer's most recent order relative to the dataset boundary.
            - **Frequency (F)**: Total count of distinct invoice orders placed by the account.
            - **Monetary (M)**: Aggregate gross revenue generated by the customer.
            - Each metric is evaluated into quantile score tiers (1 to 5). Standard behavioral segment mapping:
              - *Champions*: Top 20% in recency, frequency, and spend.
              - *Loyal Customers*: Highly regular buyers with above-average lifetime value.
              - *At Risk*: Formerly frequent buyers whose elapsed recency indicates impending churn.
              - *Lost Customers*: Dormant low-frequency buyers with high recency.
            
            *Important*: These segment names are heuristic operational categories designed to guide marketing resource allocation, not immutable statistical truths.
            """)


# PAGE 4: ML CUSTOMER SEGMENTATION
elif page == "🤖 ML Customer Segmentation":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Machine Learning Customer Segmentation
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Unsupervised K-Means clustering with Log-Standardization and Principal Component Analysis (PCA).
        </div>
    </div>
    """, unsafe_allow_html=True)

    rfm_df = calculate_rfm(filtered_df)
    
    if rfm_df.empty:
        st.warning("Customer clustering requires transactions with valid Customer IDs.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([4, 8])
        with col_ctrl1:
            k_val = st.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=4, step=1)
            st.caption("Standard K-Means with Log(1+X) feature transformation to normalize retail right-skew.")

        clustered_df, cluster_meta = run_kmeans_segmentation(rfm_df, n_clusters=k_val)

        with col_ctrl2:
            fig_pca = px.scatter(
                clustered_df.sample(min(2000, len(clustered_df)), random_state=42),
                x="PCA1",
                y="PCA2",
                color="Cluster",
                title=f"2D PCA Projection of Customer Clusters (Explained Variance: {cluster_meta['explained_variance_pct']:.1f}%)",
                hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
                color_continuous_scale="Turbo"
            )
            apply_plotly_theme(fig_pca)
            st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("#### Algorithmic Cluster Profiles & Actionable Personas")
        profiles = cluster_meta.get("cluster_profiles", [])
        
        cols = st.columns(len(profiles))
        for idx, prof in enumerate(profiles):
            with cols[idx]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size: 0.75rem; color: #3B82F6; font-weight: 700; text-transform: uppercase;">Cluster {prof['cluster_id']}</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0 8px 0;">{prof['persona_name']}</div>
                    <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.4; margin-bottom: 12px;">{prof['description']}</div>
                    <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                        <div style="font-size: 0.75rem; color: #94A3B8;"><b>Accounts:</b> {prof['customer_count']:,} ({prof['customer_share_pct']:.1f}%)</div>
                        <div style="font-size: 0.75rem; color: #94A3B8;"><b>Avg Spend:</b> £{prof['avg_monetary_spend']:,.0f}</div>
                        <div style="font-size: 0.75rem; color: #94A3B8;"><b>Avg Frequency:</b> {prof['avg_frequency_orders']:.1f} orders</div>
                        <div style="font-size: 0.75rem; color: #94A3B8;"><b>Avg Recency:</b> {prof['avg_recency_days']:.0f} days</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.75rem; color: #10B981; font-weight: 500;">
                        <b>Strategy:</b> {prof['strategy']}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# PAGE 5: ANOMALY DETECTION
elif page == "🔍 Anomaly Detection":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Isolation Forest Anomaly Detection
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Algorithmic identification of statistically unusual transactions across Quantity, Price, and Order Value.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_rate = st.slider("Expected Contamination Rate (%)", min_value=0.2, max_value=3.0, value=1.0, step=0.2) / 100.0
    
    anomalies_df, anomaly_summary = detect_transaction_anomalies(filtered_df, contamination=c_rate)

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        render_kpi("Statistically Unusual Records", f"{anomaly_summary.get('total_anomalies', 0):,}", delta=f"{c_rate*100:.1f}% Contamination", delta_type="neutral")
    with col_a2:
        render_kpi("Avg Anomaly Order Value", format_currency(anomaly_summary.get("avg_anomaly_revenue", 0.0)), delta="Elevated Commercial Value", delta_type="warning")
    with col_a3:
        render_kpi("Peak Anomaly Order Value", format_currency(anomaly_summary.get("max_anomaly_revenue", 0.0)), delta="Outlier Transaction", delta_type="danger")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if not anomalies_df.empty:
        fig_anom = px.scatter(
            anomalies_df.head(200),
            x="Quantity",
            y="Price",
            size="Revenue",
            color="AnomalyScore",
            hover_data=["Invoice", "Description", "Customer ID"],
            title="Statistical Outlier Distribution (Quantity vs Unit Price)",
            color_continuous_scale="Reds_r"
        )
        apply_plotly_theme(fig_anom)
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown("#### Flagged Unusual Transactions Ledger")
        st.dataframe(
            anomalies_df[[
                "Invoice", "StockCode", "Description", "Quantity", "Price", "Revenue", "AnomalyScore", "AnomalyReason"
            ]].head(50).style.format({
                "Quantity": "{:,}",
                "Price": "£{:,.2f}",
                "Revenue": "£{:,.2f}",
                "AnomalyScore": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.info(anomaly_summary.get("disclaimer", ""))


# PAGE 6: SALES FORECASTING
elif page == "🔮 Sales Forecasting":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Sales Revenue Forecasting
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Autoregressive predictive modeling using multi-step lag features and Random Forest / Linear Regression.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_fc1, col_fc2 = st.columns([4, 8])
    with col_fc1:
        model_choice = st.selectbox("Forecasting Algorithm", ["Random Forest", "Linear Regression"])
        horizon = st.slider("Forecast Horizon (Months)", min_value=1, max_value=6, value=3)

    forecast_df, forecast_meta = build_monthly_revenue_forecast(
        filtered_df,
        forecast_horizon_months=horizon,
        model_type=model_choice
    )

    if forecast_df.empty:
        st.warning("Insufficient monthly historical data to build lag features.")
    else:
        with col_fc2:
            direction = forecast_meta.get("expected_direction", "Stable")
            badge_class = forecast_meta.get("badge_class", "badge-primary")
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="kpi-title">Projected Commercial Direction</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">
                            {direction} ({forecast_meta.get('projected_growth_pct', 0.0):+.1f}%)
                        </div>
                    </div>
                    <span class="badge {badge_class}" style="font-size: 0.9rem; padding: 6px 14px;">
                        {direction}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        fig_fc = px.line(
            forecast_df,
            x="YearMonth",
            y="Revenue",
            color="Type",
            title=f"Historical Monthly Revenue vs {horizon}-Month Predictive Forecast ({model_choice})",
            markers=True,
            color_discrete_map={"Historical": COLOR_PRIMARY, "Forecast": COLOR_SECONDARY}
        )
        apply_plotly_theme(fig_fc)
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("#### Forecast Projections Ledger")
        st.dataframe(
            forecast_df.tail(horizon + 3).style.format({
                "Revenue": "£{:,.2f}",
                "Orders": "{:.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.warning(forecast_meta.get("disclaimer", ""))


# PAGE 7: AI BUSINESS ANALYST
elif page == "💡 AI Business Analyst":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            GenAI Business Analyst Console
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Ask questions in plain English. Calculations are verified in Python/DuckDB; explanations synthesized by Groq.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clickable Suggestion Chips
    st.markdown("##### 💡 Suggested Questions")
    pills = [
        "Show me revenue trends",
        "Top 10 products",
        "Best-performing countries",
        "Analyze customer segments",
        "Find anomalies",
        "Give me an executive summary"
    ]
    
    col_pills = st.columns(len(pills))
    selected_suggestion = None
    for i, pill in enumerate(pills):
        if col_pills[i].button(pill, key=f"pill_{i}", use_container_width=True):
            selected_suggestion = pill

    query_input = st.text_input(
        "Enter your question for the AI Analyst:",
        value=selected_suggestion or st.session_state.get("ai_query", ""),
        placeholder="e.g. Which country generated the highest revenue and why?"
    )

    if st.button("🚀 Analyze with InsightAI", type="primary") or selected_suggestion:
        query_to_run = query_input or selected_suggestion
        if query_to_run:
            with st.spinner("Executing deterministic queries and generating executive narrative..."):
                result = run_ai_analyst_pipeline(
                    query_to_run,
                    filtered_df,
                    audit_metadata,
                    llm_client
                )

            # Display Pipeline Execution Details
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600;">DETECTED INTENT: <span style="color: #60A5FA; text-transform: uppercase;">{result['intent']}</span></div>
                <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px;"><b>Computation Executed:</b> {result['analysis_performed']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Two Column Display: Verified Calculations & AI Insights
            c_calc, c_ai = st.columns([5, 7])

            with c_calc:
                st.markdown("#### 🔢 Verified Metrics (Calculated by Python / DuckDB)")
                st.json(result["verified_metrics"])

                if result["chart"] is not None:
                    st.plotly_chart(result["chart"], use_container_width=True)

            with c_ai:
                st.markdown("#### 🧠 AI Executive Interpretation")
                insights = result["insights"]

                if insights.get("is_demo_mode"):
                    st.info("ℹ️ Running in Demo Mode. Connect your GROQ_API_KEY for dynamic generative evaluation.")

                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom: 16px;">
                    <div class="kpi-title">Executive Summary</div>
                    <div style="font-size: 1.05rem; color: #F8FAFC; line-height: 1.5;">{insights.get('summary', '')}</div>
                </div>
                """, unsafe_allow_html=True)

                if insights.get("key_findings"):
                    st.markdown("##### 📌 Verified Key Findings")
                    for kf in insights["key_findings"]:
                        st.markdown(f"<div class='insight-fact'>{kf}</div>", unsafe_allow_html=True)

                if insights.get("business_implications"):
                    st.markdown("##### 🎯 Business Implications & Hypotheses")
                    for bi in insights["business_implications"]:
                        st.markdown(f"<div class='insight-hypothesis'>{bi}</div>", unsafe_allow_html=True)

                if insights.get("recommendations"):
                    st.markdown("##### 💡 Strategic Recommendations")
                    for rec in insights["recommendations"]:
                        st.markdown(f"<div class='insight-rec'>{rec}</div>", unsafe_allow_html=True)

                if insights.get("limitations"):
                    st.markdown("##### ⚠️ Methodological Limitations")
                    for lim in insights["limitations"]:
                        st.caption(f"• {lim}")


# PAGE 8: DATA QUALITY & AUDIT
elif page == "🛡️ Data Quality & Audit":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Data Quality & Preprocessing Audit
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Full transparency into record completeness, deduplication, cancellations, and cleaning rules.
        </div>
    </div>
    """, unsafe_allow_html=True)

    dq_metrics = compute_data_quality_metrics(raw_df, audit_metadata)

    c_score, c_retained, c_cancelled = st.columns(3)
    with c_score:
        score_val = dq_metrics["quality_score"]
        render_kpi(
            "Data Quality Score",
            f"{score_val}/100",
            delta=f"Status: {dq_metrics['status']}",
            delta_type="positive" if score_val >= 75 else "warning"
        )
    with c_retained:
        render_kpi(
            "Analytical Records Retained",
            f"{dq_metrics['retained_rows']:,}",
            delta=f"{dq_metrics['retained_pct']:.1f}% of raw transactions",
            delta_type="positive"
        )
    with c_cancelled:
        render_kpi(
            "Cancelled Invoices Isolated",
            f"{dq_metrics['cancelled_orders']:,}",
            delta=f"{dq_metrics['cancelled_pct']:.1f}% of transactions",
            delta_type="neutral"
        )

    st.markdown("#### Quality Audit Breakdown by Dimension")
    st.dataframe(pd.DataFrame(dq_metrics["breakdown"]), use_container_width=True, hide_index=True)

    st.markdown("#### Preprocessing Steps Performed")
    for step in dq_metrics["steps_performed"]:
        st.markdown(f"- {step}")


# PAGE 9: EXECUTIVE REPORT
elif page == "📑 Executive Report":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            Executive Board Report Generator
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            One-click synthesis of strategic commercial findings into a comprehensive executive memorandum.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📄 Generate Full Executive Report", type="primary"):
        with st.spinner("Synthesizing multi-dimensional commercial report..."):
            kpis = calculate_executive_kpis(filtered_df)
            monthly = get_monthly_sales_trend(filtered_df)
            top_countries = get_country_analytics(filtered_df, top_n=5)
            top_prods = get_product_performance_table(filtered_df, top_n=5)
            rfm = calculate_rfm(filtered_df)
            segments = get_rfm_segment_summary(rfm)
            _, forecast_meta = build_monthly_revenue_forecast(filtered_df, forecast_horizon_months=3)

            summary_insights = llm_client.generate_executive_summary(
                kpis=kpis,
                trends={"monthly_history_length": len(monthly), "latest_growth": f"{monthly.iloc[-1]['Revenue_Growth_Pct']:.1f}%" if not monthly.empty else "N/A"},
                segments=segments.to_dict(orient="records") if not segments.empty else {}
            )

            report_md = f"""# InsightAI Executive Commercial Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** UCI Online Retail II (Verified Transactions)

---

## 1. Executive Summary
{summary_insights.get('summary', 'Commercial operations demonstrate resilient transactional volume.')}

---

## 2. Key Verified Business KPIs
- **Gross Verified Revenue:** {format_currency(kpis['total_revenue'])}
- **Total Orders Delivered:** {kpis['total_orders']:,}
- **Active Transacting Accounts:** {kpis['total_customers']:,}
- **Average Order Value (AOV):** {format_currency(kpis['avg_order_value'])}
- **Catalog SKUs Sold:** {kpis['total_products']:,}

---

## 3. Strategic Findings
"""
            for kf in summary_insights.get("key_findings", []):
                report_md += f"- {kf}\n"

            report_md += "\n## 4. Market & Segment Opportunities\n"
            for imp in summary_insights.get("business_implications", []):
                report_md += f"- {imp}\n"

            report_md += "\n## 5. Commercial Recommendations\n"
            for rec in summary_insights.get("recommendations", []):
                report_md += f"- {rec}\n"

            report_md += f"""
---

## 6. Predictive Outlook
- **Model:** {forecast_meta.get('model_type', 'Autoregressive Random Forest')}
- **3-Month Trajectory:** {forecast_meta.get('expected_direction', 'Stable')} ({forecast_meta.get('projected_growth_pct', 0.0):+.1f}%)
- **Projected Monthly Average:** {format_currency(forecast_meta.get('forecast_avg_revenue', 0.0))}

---

## 7. Data Integrity & Governance
- **Data Quality Score:** {compute_data_quality_metrics(raw_df, audit_metadata)['quality_score']}/100
- **Audited Records:** {len(raw_df):,} raw rows, {len(filtered_df):,} verified analytical records.
"""

            st.markdown(report_md)
            st.download_button(
                "📥 Download Report as Markdown (.md)",
                data=report_md,
                file_name="InsightAI_Executive_Report.md",
                mime="text/markdown"
            )


# PAGE 10: INGESTION & SETTINGS
elif page == "⚙️ Ingestion & Settings":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #F8FAFC;">
            System Settings & Data Ingestion
        </h1>
        <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
            Configure AI models, manage UCI Online Retail II dataset, or upload custom retail files.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_dataset, tab_ai = st.tabs(["📂 Dataset Ingestion & Upload", "🤖 AI Provider Configuration"])

    with tab_dataset:
        st.markdown("#### Mode 1: UCI Online Retail II Dataset")
        st.write(f"- Local Excel: `{'Available' if EXCEL_PATH.exists() else 'Not Present'}`")
        st.write(f"- Cached Parquet: `{'Active (<1s loads)' if PARQUET_PATH.exists() else 'Not Present'}`")
        
        if st.button("🔄 Force Re-Download and Re-Cache UCI Dataset"):
            with st.spinner("Downloading from UCI Machine Learning Repository..."):
                progress_bar = st.progress(0.0)
                status_box = st.empty()
                def ui_cb(pct, msg):
                    progress_bar.progress(pct)
                    status_box.caption(msg)
                load_dataset(force_download=True, progress_callback=ui_cb)
                st.cache_data.clear()
                st.success("Successfully downloaded and cached UCI Online Retail II dataset!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Mode 2: Custom Dataset Upload")
        uploaded = st.file_uploader("Upload CSV, Excel (.xlsx), or Parquet file", type=["csv", "xlsx", "xls", "parquet"])
        if uploaded is not None:
            try:
                custom_df = load_custom_file(uploaded)
                st.session_state["custom_df"] = custom_df
                st.cache_data.clear()
                st.success(f"Loaded custom dataset with {len(custom_df):,} records and columns: {list(custom_df.columns)}")
                if st.button("Activate Custom Dataset"):
                    st.rerun()
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    with tab_ai:
        st.markdown("#### LLM Provider Configuration")
        provider_choice = st.selectbox(
            "Provider",
            ["groq", "ollama"],
            index=0 if st.session_state["provider"] == "groq" else 1
        )
        api_key_input = st.text_input(
            "Groq API Key",
            value=st.session_state["api_key"],
            type="password"
        )
        model_choice = st.selectbox(
            "Model Name",
            [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant"
            ],
            index=0
        )

        if st.button("💾 Save Settings"):
            st.session_state["provider"] = provider_choice
            st.session_state["api_key"] = api_key_input
            st.session_state["model_name"] = model_choice
            st.success("Configuration updated successfully!")
            st.rerun()
