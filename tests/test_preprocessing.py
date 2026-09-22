"""Unit tests for InsightAI Preprocessing Pipeline."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_retail_data


@pytest.fixture
def sample_raw_data():
    """Generates synthetic retail transaction records covering all edge cases."""
    data = [
        # Normal valid transactions
        {"Invoice": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "Price": 2.55, "Customer ID": 17850, "Country": "United Kingdom"},
        {"Invoice": "536365", "StockCode": "71053", "Description": "WHITE METAL LANTERN", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "Price": 3.39, "Customer ID": 17850, "Country": "United Kingdom"},
        # Duplicate row
        {"Invoice": "536365", "StockCode": "71053", "Description": "WHITE METAL LANTERN", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "Price": 3.39, "Customer ID": 17850, "Country": "United Kingdom"},
        # Cancelled transaction
        {"Invoice": "C536379", "StockCode": "D", "Description": "Discount", "Quantity": -1, "InvoiceDate": "2010-12-01 09:41:00", "Price": 27.50, "Customer ID": 14527, "Country": "United Kingdom"},
        # Negative quantity without C
        {"Invoice": "536380", "StockCode": "84029G", "Description": "KNITTED UNION FLAG", "Quantity": -5, "InvoiceDate": "2010-12-01 09:45:00", "Price": 3.75, "Customer ID": 17850, "Country": "United Kingdom"},
        # Zero price
        {"Invoice": "536381", "StockCode": "84029E", "Description": "RED WOOLLY HOTTIE", "Quantity": 1, "InvoiceDate": "2010-12-01 09:50:00", "Price": 0.00, "Customer ID": 17850, "Country": "United Kingdom"},
        # Negative price
        {"Invoice": "536382", "StockCode": "B", "Description": "Bank Charges", "Quantity": 1, "InvoiceDate": "2010-12-01 09:55:00", "Price": -50.00, "Customer ID": 17850, "Country": "United Kingdom"},
        # Missing Customer ID
        {"Invoice": "536383", "StockCode": "22752", "Description": "SET 7 BABUSHKA BOXES", "Quantity": 2, "InvoiceDate": "2010-12-01 10:00:00", "Price": 8.50, "Customer ID": np.nan, "Country": "France"}
    ]
    return pd.DataFrame(data)


def test_preprocessing_filters_invalid_records(sample_raw_data):
    """Verifies that duplicates, cancellations, negative quantities, and zero/negative prices are filtered."""
    cleaned_df, audit = preprocess_retail_data(sample_raw_data)
    
    # Raw rows = 8, duplicate = 1, cancelled = 1, neg_qty = 1, zero_price = 1, neg_price = 1
    # Valid rows should be: row 0, row 1 (row 2 was duplicate), and row 7 (valid row without Customer ID) = 3 rows
    assert len(cleaned_df) == 3
    assert audit["raw_rows"] == 8
    assert audit["duplicate_rows"] == 1
    assert audit["cancelled_invoices"] == 1
    assert audit["zero_or_negative_prices"] == 2
    assert audit["negative_quantities"] == 2


def test_preprocessing_computes_revenue(sample_raw_data):
    """Verifies that Revenue = Quantity * Price is accurately calculated."""
    cleaned_df, audit = preprocess_retail_data(sample_raw_data)
    
    # Row 0: 6 * 2.55 = 15.30
    # Row 1: 6 * 3.39 = 20.34
    # Row 7: 2 * 8.50 = 17.00
    expected_total_rev = 15.30 + 20.34 + 17.00
    assert pytest.approx(cleaned_df["Revenue"].sum(), 0.01) == expected_total_rev
    assert pytest.approx(audit["retained_revenue"], 0.01) == expected_total_rev


def test_preprocessing_temporal_features(sample_raw_data):
    """Verifies that datetime parsing generates YearMonth and DayOfWeek."""
    cleaned_df, _ = preprocess_retail_data(sample_raw_data)
    assert "YearMonth" in cleaned_df.columns
    assert "DayOfWeek" in cleaned_df.columns
    assert (cleaned_df["YearMonth"] == "2010-12").all()
