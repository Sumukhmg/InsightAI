"""InsightAI AI Analytics Pipeline & Intent Router
Maps natural-language user queries to deterministic Python/DuckDB analytical functions,
computes verified metrics, generates charts, and passes verified results to the LLM.
"""

from typing import Any, Dict, Optional, Tuple
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import apply_plotly_theme, format_currency
from src.analytics import (
    calculate_executive_kpis,
    get_monthly_sales_trend,
    get_country_analytics,
    get_product_performance_table,
    get_top_customers
)
from src.segmentation import calculate_rfm, get_rfm_segment_summary, run_kmeans_segmentation
from src.anomaly_detection import detect_transaction_anomalies
from src.forecasting import build_monthly_revenue_forecast
from src.data_quality import compute_data_quality_metrics
from src.llm import LLMClient


def detect_intent(query: str) -> str:
    """Classifies user query into one of the core analytical intents using pattern matching."""
    q = query.lower().strip()
    
    if any(k in q for k in ["anomal", "unusual", "outlier", "fraud", "irregular"]):
        return "anomaly"
    elif any(k in q for k in ["forecast", "predict", "future", "next month", "projection"]):
        return "forecast"
    elif any(k in q for k in ["segment", "cluster", "at risk", "champions", "rfm", "loyal"]):
        return "segmentation"
    elif any(k in q for k in ["country", "countries", "geograph", "region", "nation", "uk", "germany", "france"]):
        return "country_analysis"
    elif any(k in q for k in ["product", "item", "bestsell", "best-sell", "stock", "sku", "declining"]):
        return "product_analysis"
    elif any(k in q for k in ["customer", "client", "buyer", "shopper", "highest-value"]):
        return "customer_analysis"
    elif any(k in q for k in ["trend", "over time", "monthly", "growth", "history", "trajectory"]):
        return "trend"
    elif any(k in q for k in ["quality", "missing", "clean", "duplicate", "invalid"]):
        return "data_quality"
    else:
        return "summary"


def run_ai_analyst_pipeline(
    query: str,
    df: pd.DataFrame,
    audit_metadata: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Executes the complete end-to-end AI Analyst Pipeline.
    
    Architecture:
    User Question -> Intent Detection -> Deterministic Calculation -> Verified Results -> LLM -> Business Narrative
    """
    intent = detect_intent(query)
    verified_metrics: Dict[str, Any] = {}
    chart: Optional[go.Figure] = None
    analysis_desc = ""

    # Execute deterministic analytics according to detected intent
    if intent == "summary":
        analysis_desc = "Computed verified Executive KPIs and overall revenue volume across the analytical dataset."
        kpis = calculate_executive_kpis(df)
        verified_metrics = {
            "Total Revenue": format_currency(kpis["total_revenue"]),
            "Total Invoices": f"{kpis['total_orders']:,}",
            "Total Unique Customers": f"{kpis['total_customers']:,}",
            "Average Order Value (AOV)": format_currency(kpis["avg_order_value"]),
            "Total Units Sold": f"{kpis['total_quantity']:,}"
        }
        monthly = get_monthly_sales_trend(df)
        if not monthly.empty:
            chart = px.line(
                monthly,
                x="YearMonth",
                y="Revenue",
                title="Historical Revenue Trajectory",
                markers=True
            )
            apply_plotly_theme(chart)

    elif intent == "country_analysis":
        analysis_desc = "Aggregated commercial revenue and order distribution grouped by Country."
        countries = get_country_analytics(df, top_n=10)
        top_country = countries.iloc[0] if not countries.empty else None
        
        verified_metrics = {
            "Top Country": str(top_country["Country"]) if top_country is not None else "N/A",
            "Top Country Revenue": format_currency(top_country["Revenue"]) if top_country is not None else "£0",
            "Top Country Revenue Share": f"{top_country['Revenue_Share_Pct']:.1f}%" if top_country is not None else "0%",
            "Total Countries Served": int(df["Country"].nunique()),
            "Top 5 Countries by Revenue": [
                {"Country": row["Country"], "Revenue": format_currency(row["Revenue"]), "Share": f"{row['Revenue_Share_Pct']:.1f}%"}
                for _, row in countries.head(5).iterrows()
            ]
        }
        if not countries.empty:
            chart = px.bar(
                countries.head(10),
                x="Country",
                y="Revenue",
                color="Revenue",
                title="Top 10 Countries by Revenue",
                labels={"Revenue": "Revenue (£)"}
            )
            apply_plotly_theme(chart)

    elif intent == "product_analysis":
        analysis_desc = "Calculated SKU-level product revenue, transaction volume, and unit demand."
        products = get_product_performance_table(df, top_n=10)
        top_prod = products.iloc[0] if not products.empty else None
        
        verified_metrics = {
            "Top Product": str(top_prod["Description"]) if top_prod is not None else "N/A",
            "Top Product Revenue": format_currency(top_prod["Revenue"]) if top_prod is not None else "£0",
            "Top Product Units Sold": f"{int(top_prod['Quantity']):,}" if top_prod is not None else "0",
            "Distinct SKUs Sold": int(df["StockCode"].nunique()),
            "Top 5 Products": [
                {"Description": row["Description"][:35], "Revenue": format_currency(row["Revenue"]), "Units": f"{int(row['Quantity']):,}"}
                for _, row in products.head(5).iterrows()
            ]
        }
        if not products.empty:
            chart = px.bar(
                products.head(8),
                y="Description",
                x="Revenue",
                orientation="h",
                title="Top Products by Verified Revenue",
                color="Revenue"
            )
            chart.update_layout(yaxis=dict(autorange="reversed"))
            apply_plotly_theme(chart)

    elif intent == "trend":
        analysis_desc = "Aggregated monthly time series to calculate MoM revenue velocity and growth percentages."
        monthly = get_monthly_sales_trend(df)
        last_month = monthly.iloc[-1] if not monthly.empty else None
        
        verified_metrics = {
            "Total Monitored Months": len(monthly),
            "Latest Month Revenue": format_currency(last_month["Revenue"]) if last_month is not None else "£0",
            "Latest Month Growth %": f"{last_month['Revenue_Growth_Pct']:.1f}%" if last_month is not None else "0%",
            "Highest Revenue Month": str(monthly.loc[monthly["Revenue"].idxmax()]["YearMonth"]) if not monthly.empty else "N/A",
            "Peak Monthly Revenue": format_currency(monthly["Revenue"].max()) if not monthly.empty else "£0"
        }
        if not monthly.empty:
            chart = px.bar(
                monthly,
                x="YearMonth",
                y="Revenue",
                text=monthly["Revenue_Growth_Pct"].apply(lambda x: f"{x:+.1f}%"),
                title="Monthly Revenue & MoM Growth %"
            )
            apply_plotly_theme(chart)

    elif intent == "customer_analysis":
        analysis_desc = "Queried top high-value customer accounts and spending patterns."
        top_cust = get_top_customers(df, top_n=10)
        verified_metrics = {
            "Top Customer ID": str(top_cust.iloc[0]["Customer ID"]) if not top_cust.empty else "N/A",
            "Top Customer Spend": format_currency(top_cust.iloc[0]["Revenue"]) if not top_cust.empty else "£0",
            "Top Customer Orders": int(top_cust.iloc[0]["Orders"]) if not top_cust.empty else 0,
            "Top 5 Customers": [
                {"Customer ID": str(r["Customer ID"]), "Spend": format_currency(r["Revenue"]), "Orders": int(r["Orders"])}
                for _, r in top_cust.head(5).iterrows()
            ]
        }
        if not top_cust.empty:
            chart = px.bar(
                top_cust.head(10),
                x="Customer ID",
                y="Revenue",
                title="Top 10 High-Value Customers by Lifetime Spend",
                color="Revenue"
            )
            apply_plotly_theme(chart)

    elif intent == "segmentation":
        analysis_desc = "Executed Recency, Frequency, and Monetary (RFM) customer value segmentation."
        rfm_df = calculate_rfm(df)
        summary = get_rfm_segment_summary(rfm_df)
        
        verified_metrics = {
            "Total Segmented Customers": len(rfm_df),
            "Active Champions": int(summary.loc[summary["Segment"] == "Champions", "Customer_Count"].sum()) if not summary.empty else 0,
            "At Risk Customers": int(summary.loc[summary["Segment"] == "At Risk", "Customer_Count"].sum()) if not summary.empty else 0,
            "Segment Breakdown": [
                {"Segment": r["Segment"], "Customers": int(r["Customer_Count"]), "Revenue": format_currency(r["Total_Revenue"])}
                for _, r in summary.iterrows()
            ]
        }
        if not summary.empty:
            chart = px.pie(
                summary,
                names="Segment",
                values="Total_Revenue",
                title="Revenue Contribution by Customer Segment",
                hole=0.4
            )
            apply_plotly_theme(chart)

    elif intent == "anomaly":
        analysis_desc = "Fitted Isolation Forest machine learning model on Quantity, UnitPrice, and Order Revenue."
        anomalies_df, anomaly_summary = detect_transaction_anomalies(df, contamination=0.01)
        
        verified_metrics = {
            "Total Flagged Unusual Transactions": anomaly_summary.get("total_anomalies", 0),
            "Average Anomaly Transaction Value": format_currency(anomaly_summary.get("avg_anomaly_revenue", 0)),
            "Maximum Anomaly Transaction Value": format_currency(anomaly_summary.get("max_anomaly_revenue", 0)),
            "Sample Flagged Transactions": [
                {
                    "Invoice": str(r["Invoice"]),
                    "Product": str(r["Description"])[:30],
                    "Quantity": int(r["Quantity"]),
                    "Price": f"£{r['Price']:.2f}",
                    "Revenue": format_currency(r["Revenue"])
                }
                for _, r in anomalies_df.head(5).iterrows()
            ]
        }
        if not anomalies_df.empty:
            chart = px.scatter(
                anomalies_df.head(100),
                x="Quantity",
                y="Price",
                size="Revenue",
                color="AnomalyScore",
                title="Top Statistically Unusual Transactions (Isolation Forest)",
                hover_data=["Invoice", "Description"]
            )
            apply_plotly_theme(chart)

    elif intent == "forecast":
        analysis_desc = "Trained autoregressive machine learning model on historical monthly revenue to project next 3 months."
        forecast_df, forecast_meta = build_monthly_revenue_forecast(df, forecast_horizon_months=3)
        
        verified_metrics = {
            "Forecast Model": forecast_meta.get("model_type", "Random Forest"),
            "Projected 3-Month Trend": forecast_meta.get("expected_direction", "Stable"),
            "Projected Growth %": f"{forecast_meta.get('projected_growth_pct', 0.0):+.1f}%",
            "Projected Monthly Average": format_currency(forecast_meta.get("forecast_avg_revenue", 0.0)),
            "Historical Last Month Revenue": format_currency(forecast_meta.get("last_actual_revenue", 0.0))
        }
        if not forecast_df.empty:
            chart = px.line(
                forecast_df,
                x="YearMonth",
                y="Revenue",
                color="Type",
                title="Monthly Revenue: Historical vs 3-Month Forecast",
                markers=True
            )
            apply_plotly_theme(chart)

    elif intent == "data_quality":
        analysis_desc = "Evaluated dataset completeness, duplication, price bounds, and cancellation rates."
        dq_metrics = compute_data_quality_metrics(df, audit_metadata)
        
        verified_metrics = {
            "Data Quality Score": f"{dq_metrics['quality_score']}/100",
            "Health Status": dq_metrics["status"],
            "Raw Transaction Rows": f"{dq_metrics['raw_rows']:,}",
            "Cleaned Analytical Rows": f"{dq_metrics['retained_rows']:,}",
            "Cancelled Transactions": f"{dq_metrics['cancelled_orders']:,} ({dq_metrics['cancelled_pct']:.1f}%)"
        }

    # Pass ONLY the verified deterministic metrics to the LLM
    llm_response = llm_client.generate_business_insight(query, intent, verified_metrics)

    return {
        "query": query,
        "intent": intent,
        "analysis_performed": analysis_desc,
        "verified_metrics": verified_metrics,
        "chart": chart,
        "insights": llm_response
    }
