"""InsightAI Data Quality Module
Computes comprehensive Data Quality Score (0-100), detailed audit breakdowns,
and health diagnostics across the retail dataset.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np


def compute_data_quality_metrics(
    raw_df: pd.DataFrame,
    audit_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes a robust enterprise data quality score (0-100) and breakdown components.
    
    The score uses weighted penalties for integrity violations:
    - Missing Customer IDs (impacts customer analytics, though normal for guest checkouts)
    - Duplicate Transactions (impacts revenue accuracy)
    - Negative/Zero Prices (corrupts financial metrics)
    - Negative Quantities / Invalid Values (data entry/system anomalies)
    - Missing Descriptions (catalog incompleteness)
    
    Returns:
        Dict[str, Any]: Detailed scoring and audit report
    """
    raw_rows = audit_metadata.get("raw_rows", len(raw_df))
    if raw_rows == 0:
        return {
            "quality_score": 0,
            "status": "Critical",
            "breakdown": {},
            "summary_text": "Dataset is empty."
        }
        
    missing_cust_pct = audit_metadata.get("missing_customer_pct", 0.0)
    dup_pct = audit_metadata.get("duplicate_pct", 0.0)
    invalid_price_pct = audit_metadata.get("zero_or_negative_prices_pct", 0.0)
    neg_qty_pct = audit_metadata.get("negative_qty_pct", 0.0)
    cancelled_pct = audit_metadata.get("cancelled_pct", 0.0)
    missing_desc_pct = audit_metadata.get("missing_descriptions_pct", 0.0)
    
    # Weighted deductions calculation
    # Guest checkout is common in retail (missing customer ID penalty is weighted moderately)
    penalty_missing_cust = min(20.0, missing_cust_pct * 0.2)
    penalty_dups = min(15.0, dup_pct * 1.5)
    penalty_invalid_price = min(20.0, invalid_price_pct * 3.0)
    penalty_neg_qty = min(15.0, neg_qty_pct * 1.0)
    penalty_missing_desc = min(10.0, missing_desc_pct * 2.0)
    
    total_deductions = (
        penalty_missing_cust +
        penalty_dups +
        penalty_invalid_price +
        penalty_neg_qty +
        penalty_missing_desc
    )
    
    quality_score = max(0, min(100, int(round(100.0 - total_deductions))))
    
    if quality_score >= 85:
        status = "Excellent"
        status_color = "#10B981"
    elif quality_score >= 70:
        status = "Good"
        status_color = "#3B82F6"
    elif quality_score >= 50:
        status = "Moderate"
        status_color = "#F59E0B"
    else:
        status = "Poor"
        status_color = "#EF4444"
        
    breakdown = [
        {
            "category": "Missing Customer IDs",
            "metric": f"{missing_cust_pct:.2f}%",
            "count": audit_metadata.get("missing_customer_ids", 0),
            "penalty": f"-{penalty_missing_cust:.1f} pts",
            "notes": "Common in guest retail checkouts; handled cleanly for aggregate sales but isolated for customer RFM."
        },
        {
            "category": "Duplicate Records",
            "metric": f"{dup_pct:.2f}%",
            "count": audit_metadata.get("duplicate_rows", 0),
            "penalty": f"-{penalty_dups:.1f} pts",
            "notes": "Duplicate entries removed to prevent double-counting revenue."
        },
        {
            "category": "Zero / Negative Unit Prices",
            "metric": f"{invalid_price_pct:.2f}%",
            "count": audit_metadata.get("zero_or_negative_prices", 0),
            "penalty": f"-{penalty_invalid_price:.1f} pts",
            "notes": "Accounting adjustments, administrative charges, or sample products."
        },
        {
            "category": "Negative Quantities (Returns/Errors)",
            "metric": f"{neg_qty_pct:.2f}%",
            "count": audit_metadata.get("negative_quantities", 0),
            "penalty": f"-{penalty_neg_qty:.1f} pts",
            "notes": "Return records or inventory corrections."
        },
        {
            "category": "Missing Descriptions",
            "metric": f"{missing_desc_pct:.2f}%",
            "count": audit_metadata.get("missing_descriptions", 0),
            "penalty": f"-{penalty_missing_desc:.1f} pts",
            "notes": "Unlabeled SKUs mapped to 'Unknown Product'."
        }
    ]
    
    return {
        "quality_score": quality_score,
        "status": status,
        "status_color": status_color,
        "raw_rows": raw_rows,
        "raw_cols": audit_metadata.get("raw_cols", len(raw_df.columns)),
        "retained_rows": audit_metadata.get("retained_rows", 0),
        "retained_pct": audit_metadata.get("retained_percentage", 0.0),
        "cancelled_orders": audit_metadata.get("cancelled_invoices", 0),
        "cancelled_pct": cancelled_pct,
        "breakdown": breakdown,
        "steps_performed": audit_metadata.get("steps_performed", [])
    }
