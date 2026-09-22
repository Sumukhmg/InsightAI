"""InsightAI Preprocessing Pipeline
Handles data cleaning, missing value accounting, cancellation extraction,
date parsing, and derived metric calculation with full audit transparency.
"""

from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np


def preprocess_retail_data(
    df: pd.DataFrame,
    filter_missing_customers_for_analysis: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans and standardizes raw retail transaction data while generating an audit log.
    
    Args:
        df: Raw retail DataFrame with columns:
            Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
        filter_missing_customers_for_analysis: If True, also filters out transactions
            with missing Customer ID (useful for customer-level RFM and segmentation).
            
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Cleaned and enriched analytical DataFrame
            - Audit metadata dictionary detailing every filter and record count
    """
    raw_rows = len(df)
    raw_cols = len(df.columns)
    
    # 1. Auditing Initial Missing Values
    missing_customer_count = int(df["Customer ID"].isna().sum()) if "Customer ID" in df.columns else 0
    missing_desc_count = int(df["Description"].isna().sum()) if "Description" in df.columns else 0
    
    # 2. Auditing Duplicates
    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    df_dedup = df[~duplicate_mask].copy()
    
    # Standardize string representations
    if "Invoice" in df_dedup.columns:
        df_dedup["Invoice"] = df_dedup["Invoice"].astype(str).str.strip()
    if "StockCode" in df_dedup.columns:
        df_dedup["StockCode"] = df_dedup["StockCode"].astype(str).str.strip()
    if "Description" in df_dedup.columns:
        df_dedup["Description"] = df_dedup["Description"].fillna("Unknown Product").astype(str).str.strip()
    if "Country" in df_dedup.columns:
        df_dedup["Country"] = df_dedup["Country"].fillna("Unspecified").astype(str).str.strip()
    if "Customer ID" in df_dedup.columns:
        df_dedup["Customer ID"] = (
            df_dedup["Customer ID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
        )
        df_dedup.loc[df_dedup["Customer ID"].isin(["nan", "None", "<NA>", ""]), "Customer ID"] = np.nan

    # 3. Numeric Conversions
    df_dedup["Quantity"] = pd.to_numeric(df_dedup["Quantity"], errors="coerce").fillna(0)
    df_dedup["Price"] = pd.to_numeric(df_dedup["Price"], errors="coerce").fillna(0.0)
    
    # 4. Auditing Cancellations & Negative Quantities
    is_cancellation = df_dedup["Invoice"].str.upper().str.startswith("C", na=False)
    cancelled_count = int(is_cancellation.sum())
    
    negative_qty_mask = df_dedup["Quantity"] <= 0
    negative_qty_count = int(negative_qty_mask.sum())
    
    zero_or_negative_price_mask = df_dedup["Price"] <= 0
    zero_or_negative_price_count = int(zero_or_negative_price_mask.sum())
    
    # 5. Build Valid Analytical Filter
    # A valid sales transaction must be non-cancelled, positive quantity, positive unit price
    valid_mask = (~is_cancellation) & (~negative_qty_mask) & (~zero_or_negative_price_mask)
    
    if filter_missing_customers_for_analysis and "Customer ID" in df_dedup.columns:
        valid_mask = valid_mask & df_dedup["Customer ID"].notna()

    cleaned_df = df_dedup[valid_mask].copy()
    
    # 6. Parse Dates & Derive Temporal Metrics
    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["InvoiceDate"])
    
    cleaned_df["Year"] = cleaned_df["InvoiceDate"].dt.year
    cleaned_df["Month"] = cleaned_df["InvoiceDate"].dt.month
    cleaned_df["YearMonth"] = cleaned_df["InvoiceDate"].dt.to_period("M").astype(str)
    cleaned_df["Date"] = cleaned_df["InvoiceDate"].dt.date
    cleaned_df["DayOfWeek"] = cleaned_df["InvoiceDate"].dt.day_name()
    cleaned_df["Hour"] = cleaned_df["InvoiceDate"].dt.hour
    
    # 7. Derived Commercial Metrics
    cleaned_df["Revenue"] = cleaned_df["Quantity"] * cleaned_df["Price"]
    
    retained_rows = len(cleaned_df)
    filtered_rows = raw_rows - retained_rows
    retained_revenue = float(cleaned_df["Revenue"].sum())
    
    # 8. Compile Comprehensive Audit Metadata
    audit_metadata = {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "retained_rows": retained_rows,
        "filtered_rows": filtered_rows,
        "retained_percentage": (retained_rows / raw_rows * 100) if raw_rows > 0 else 0.0,
        "retained_revenue": retained_revenue,
        "missing_customer_ids": missing_customer_count,
        "missing_customer_pct": (missing_customer_count / raw_rows * 100) if raw_rows > 0 else 0.0,
        "missing_descriptions": missing_desc_count,
        "missing_descriptions_pct": (missing_desc_count / raw_rows * 100) if raw_rows > 0 else 0.0,
        "duplicate_rows": duplicate_count,
        "duplicate_pct": (duplicate_count / raw_rows * 100) if raw_rows > 0 else 0.0,
        "cancelled_invoices": cancelled_count,
        "cancelled_pct": (cancelled_count / raw_rows * 100) if raw_rows > 0 else 0.0,
        "negative_quantities": negative_qty_count,
        "negative_qty_pct": (negative_qty_count / raw_rows * 100) if raw_rows > 0 else 0.0,
        "zero_or_negative_prices": zero_or_negative_price_count,
        "zero_or_negative_prices_pct": (zero_or_negative_price_count / raw_rows * 100) if raw_rows > 0 else 0.0,
        "steps_performed": [
            f"Removed {duplicate_count:,} duplicate rows to ensure transaction uniqueness.",
            f"Parsed and standardized 'InvoiceDate' to datetime objects and generated temporal features (YearMonth, Day, Hour).",
            f"Identified {cancelled_count:,} cancelled orders (Invoice starting with 'C') and {negative_qty_count:,} negative quantity return adjustments.",
            f"Identified and filtered {zero_or_negative_price_count:,} records with price <= 0.00 (administrative adjustments and zero-value items).",
            f"Derived commercial metric 'Revenue = Quantity * Price' across all valid rows.",
            f"Retained {retained_rows:,} analytical records ({retained_rows / raw_rows * 100:.1f}% of total) accounting for £{retained_revenue:,.2f} in verified commercial volume."
        ]
    }
    
    return cleaned_df, audit_metadata
