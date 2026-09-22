"""Unit tests for InsightAI Analytics Engine."""

import pytest
import pandas as pd
from src.analytics import (
    calculate_executive_kpis,
    get_monthly_sales_trend,
    get_country_analytics,
    get_product_performance_table
)


@pytest.fixture
def sample_clean_df():
    """Generates cleaned transaction records for analytics testing."""
    data = [
        {"Invoice": "1001", "StockCode": "A", "Description": "Product A", "Quantity": 2, "Price": 10.0, "Revenue": 20.0, "InvoiceDate": pd.to_datetime("2010-01-15"), "YearMonth": "2010-01", "Customer ID": "C1", "Country": "United Kingdom"},
        {"Invoice": "1001", "StockCode": "B", "Description": "Product B", "Quantity": 1, "Price": 30.0, "Revenue": 30.0, "InvoiceDate": pd.to_datetime("2010-01-15"), "YearMonth": "2010-01", "Customer ID": "C1", "Country": "United Kingdom"},
        {"Invoice": "1002", "StockCode": "A", "Description": "Product A", "Quantity": 5, "Price": 10.0, "Revenue": 50.0, "InvoiceDate": pd.to_datetime("2010-02-10"), "YearMonth": "2010-02", "Customer ID": "C2", "Country": "Germany"},
        {"Invoice": "1003", "StockCode": "C", "Description": "Product C", "Quantity": 2, "Price": 25.0, "Revenue": 50.0, "InvoiceDate": pd.to_datetime("2010-02-20"), "YearMonth": "2010-02", "Customer ID": "C1", "Country": "United Kingdom"}
    ]
    return pd.DataFrame(data)


def test_calculate_executive_kpis(sample_clean_df):
    """Verifies that KPIs match known mathematical values."""
    kpis = calculate_executive_kpis(sample_clean_df)
    
    assert kpis["total_revenue"] == 150.0
    assert kpis["total_orders"] == 3
    assert kpis["total_customers"] == 2
    assert kpis["total_products"] == 3
    assert kpis["total_quantity"] == 10
    assert kpis["avg_order_value"] == 50.0
    assert pytest.approx(kpis["avg_items_per_order"], 0.01) == 10 / 3


def test_monthly_growth_calculation(sample_clean_df):
    """Verifies that MoM growth % is calculated accurately."""
    monthly = get_monthly_sales_trend(sample_clean_df)
    
    # 2010-01 Revenue = 50.0
    # 2010-02 Revenue = 100.0
    # Growth = (100 - 50) / 50 * 100 = +100.0%
    assert len(monthly) == 2
    assert monthly.iloc[0]["Revenue"] == 50.0
    assert monthly.iloc[1]["Revenue"] == 100.0
    assert monthly.iloc[1]["Revenue_Growth_Pct"] == 100.0


def test_country_analytics(sample_clean_df):
    """Verifies geographic revenue distribution."""
    countries = get_country_analytics(sample_clean_df)
    
    # UK: 20 + 30 + 50 = 100
    # Germany: 50
    uk_row = countries[countries["Country"] == "United Kingdom"].iloc[0]
    germany_row = countries[countries["Country"] == "Germany"].iloc[0]
    
    assert uk_row["Revenue"] == 100.0
    assert pytest.approx(uk_row["Revenue_Share_Pct"], 0.1) == (100.0 / 150.0 * 100)
    assert germany_row["Revenue"] == 50.0
