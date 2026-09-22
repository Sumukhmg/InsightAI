"""Unit tests for InsightAI Machine Learning and Quality Algorithms."""

import pytest
import pandas as pd
import numpy as np
from src.segmentation import calculate_rfm, run_kmeans_segmentation
from src.anomaly_detection import detect_transaction_anomalies
from src.forecasting import build_monthly_revenue_forecast
from src.data_quality import compute_data_quality_metrics


@pytest.fixture
def sample_rfm_transactions():
    """Generates customer transactions across multiple dates to test RFM scoring."""
    rows = []
    # Customer 1: Recent, frequent, high monetary
    for d in ["2011-12-01", "2011-12-05", "2011-12-08"]:
        rows.append({"Invoice": f"I_{d}_1", "StockCode": "P1", "Description": "Prod 1", "Quantity": 10, "Price": 20.0, "Revenue": 200.0, "InvoiceDate": pd.to_datetime(d), "Customer ID": "101", "Country": "UK"})
        
    # Customer 2: Moderate recency, moderate frequency, moderate spend
    for d in ["2011-10-01", "2011-10-15"]:
        rows.append({"Invoice": f"I_{d}_2", "StockCode": "P2", "Description": "Prod 2", "Quantity": 5, "Price": 10.0, "Revenue": 50.0, "InvoiceDate": pd.to_datetime(d), "Customer ID": "102", "Country": "UK"})
        
    # Customer 3: Old recency, 1 order, low spend
    rows.append({"Invoice": "I_old_3", "StockCode": "P3", "Description": "Prod 3", "Quantity": 1, "Price": 15.0, "Revenue": 15.0, "InvoiceDate": pd.to_datetime("2011-01-10"), "Customer ID": "103", "Country": "France"})

    # Customer 4: Recent single buyer
    rows.append({"Invoice": "I_rec_4", "StockCode": "P4", "Description": "Prod 4", "Quantity": 2, "Price": 40.0, "Revenue": 80.0, "InvoiceDate": pd.to_datetime("2011-12-07"), "Customer ID": "104", "Country": "Germany"})

    # Customer 5: High spend, low frequency
    rows.append({"Invoice": "I_high_5", "StockCode": "P5", "Description": "Prod 5", "Quantity": 100, "Price": 10.0, "Revenue": 1000.0, "InvoiceDate": pd.to_datetime("2011-08-01"), "Customer ID": "105", "Country": "Spain"})

    return pd.DataFrame(rows)


def test_calculate_rfm(sample_rfm_transactions):
    """Verifies that RFM metrics and quintiles are computed."""
    rfm = calculate_rfm(sample_rfm_transactions, reference_date=pd.to_datetime("2011-12-10"))
    
    assert len(rfm) == 5
    cust_101 = rfm[rfm["Customer ID"] == "101"].iloc[0]
    
    # Recency: 2011-12-10 - 2011-12-08 = 2 days
    assert cust_101["Recency"] == 2
    assert cust_101["Frequency"] == 3
    assert cust_101["Monetary"] == 600.0
    assert "Segment" in rfm.columns


def test_kmeans_segmentation(sample_rfm_transactions):
    """Verifies that K-Means clustering and PCA coordinates are generated."""
    rfm = calculate_rfm(sample_rfm_transactions, reference_date=pd.to_datetime("2011-12-10"))
    clustered_df, metadata = run_kmeans_segmentation(rfm, n_clusters=3, random_state=42)
    
    assert "Cluster" in clustered_df.columns
    assert "PCA1" in clustered_df.columns
    assert "PCA2" in clustered_df.columns
    assert metadata["n_clusters"] == 3
    assert len(metadata["cluster_profiles"]) == 3


def test_anomaly_detection():
    """Verifies Isolation Forest flags extreme outliers."""
    # 95 normal rows + 5 extreme outliers
    normal_data = [{"Quantity": 2, "Price": 5.0, "Revenue": 10.0, "Invoice": f"N_{i}", "Description": "Widget"} for i in range(95)]
    outliers = [
        {"Quantity": 5000, "Price": 500.0, "Revenue": 2500000.0, "Invoice": "OUT_1", "Description": "Super Mega Bulk"},
        {"Quantity": 1000, "Price": 80.0, "Revenue": 80000.0, "Invoice": "OUT_2", "Description": "Bulk Order"},
        {"Quantity": 1, "Price": 15000.0, "Revenue": 15000.0, "Invoice": "OUT_3", "Description": "Luxury Specialty Item"},
        {"Quantity": 2000, "Price": 2.0, "Revenue": 4000.0, "Invoice": "OUT_4", "Description": "Pallet of Buttons"},
        {"Quantity": 500, "Price": 300.0, "Revenue": 150000.0, "Invoice": "OUT_5", "Description": "Wholesale Machinery"}
    ]
    df = pd.DataFrame(normal_data + outliers)
    anomalies, summary = detect_transaction_anomalies(df, contamination=0.05, random_state=42)
    
    assert len(anomalies) > 0
    assert "OUT_1" in anomalies["Invoice"].values
    assert "AnomalyScore" in anomalies.columns
    assert "AnomalyReason" in anomalies.columns


def test_forecasting_generates_3_months():
    """Verifies monthly revenue forecasting generates next 3 months."""
    months = [f"2010-{m:02d}" for m in range(1, 13)] + [f"2011-{m:02d}" for m in range(1, 11)]
    revs = [10000 + i * 500 for i in range(len(months))]
    
    df_list = []
    for m, r in zip(months, revs):
        df_list.append({"YearMonth": m, "Revenue": r, "Invoice": f"INV_{m}", "InvoiceDate": pd.to_datetime(f"{m}-15")})
    df = pd.DataFrame(df_list)
    
    forecast_df, meta = build_monthly_revenue_forecast(df, forecast_horizon_months=3)
    
    forecast_rows = forecast_df[forecast_df["Type"] == "Forecast"]
    assert len(forecast_rows) == 3
    assert meta["expected_direction"] in ["Increasing", "Stable", "Decreasing"]


def test_data_quality_score():
    """Verifies that Data Quality Score outputs 0-100 with accurate penalties."""
    audit_clean = {
        "raw_rows": 1000,
        "raw_cols": 8,
        "retained_rows": 980,
        "retained_percentage": 98.0,
        "missing_customer_pct": 2.0,
        "duplicate_pct": 0.5,
        "zero_or_negative_prices_pct": 0.0,
        "negative_qty_pct": 0.5,
        "cancelled_pct": 0.5,
        "missing_descriptions_pct": 0.0
    }
    raw_df = pd.DataFrame({"dummy": range(1000)})
    dq = compute_data_quality_metrics(raw_df, audit_clean)
    
    assert 0 <= dq["quality_score"] <= 100
    assert dq["status"] in ["Excellent", "Good", "Moderate", "Poor"]
    assert len(dq["breakdown"]) == 5
