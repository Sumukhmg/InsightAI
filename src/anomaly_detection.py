"""InsightAI Anomaly Detection Module
Uses Scikit-learn's Isolation Forest to identify statistically unusual transactions
based on Quantity, UnitPrice, and Revenue.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_transaction_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
    sample_size: int = 100000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Identifies statistically unusual transactions using Isolation Forest.
    
    Args:
        df: Cleaned analytical transaction DataFrame.
        contamination: The proportion of outliers expected in the data set (default 0.01 = 1%).
        random_state: Seed for reproducibility.
        sample_size: If dataset exceeds sample_size, fits on sample to ensure fast response.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame of flagged anomalies with scores and contextual explanations
            - Metadata summary dictionary
    """
    if df.empty:
        return pd.DataFrame(), {}

    features = ["Quantity", "Price", "Revenue"]
    data = df[features].copy()
    
    # Handle infinite/nan values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1
    )
    
    if len(data) > sample_size:
        sample_data = data.sample(sample_size, random_state=random_state)
        iso.fit(sample_data)
    else:
        iso.fit(data)

    # Predict anomaly flag (-1 = anomaly, 1 = normal)
    # Decision function: lower values mean more anomalous
    scores = iso.decision_function(data)
    preds = iso.predict(data)
    
    # Extract anomalies
    anomaly_indices = data.index[preds == -1]
    anomalies_df = df.loc[anomaly_indices].copy()
    anomalies_df["AnomalyScore"] = scores[preds == -1]
    
    # Sort by anomaly severity (lowest score first)
    anomalies_df = anomalies_df.sort_values("AnomalyScore", ascending=True)

    # Generate Plain-English Statistical Explanations
    median_qty = df["Quantity"].median()
    median_price = df["Price"].median()
    median_rev = df["Revenue"].median()
    p99_qty = df["Quantity"].quantile(0.99)
    p99_price = df["Price"].quantile(0.99)
    p99_rev = df["Revenue"].quantile(0.99)

    explanations = []
    for _, row in anomalies_df.iterrows():
        reasons = []
        if row["Quantity"] > p99_qty:
            reasons.append(f"extremely high volume ({row['Quantity']:,} units vs median {median_qty:.0f})")
        if row["Price"] > p99_price:
            reasons.append(f"premium unit price (£{row['Price']:.2f} vs median £{median_price:.2f})")
        if row["Revenue"] > p99_rev:
            reasons.append(f"exceptional transaction value (£{row['Revenue']:,.2f} vs median £{median_rev:.2f})")
            
        if not reasons:
            reasons.append("unusual multi-dimensional combination of price, quantity, and total value")

        explanation = f"Statistically unusual due to {', and '.join(reasons)}."
        explanations.append(explanation)

    anomalies_df["AnomalyReason"] = explanations

    summary = {
        "total_analyzed": len(df),
        "total_anomalies": len(anomalies_df),
        "contamination_rate": contamination,
        "avg_anomaly_revenue": float(anomalies_df["Revenue"].mean()) if not anomalies_df.empty else 0.0,
        "max_anomaly_revenue": float(anomalies_df["Revenue"].max()) if not anomalies_df.empty else 0.0,
        "max_anomaly_quantity": int(anomalies_df["Quantity"].max()) if not anomalies_df.empty else 0,
        "disclaimer": (
            "These records are flagged as statistically unusual based on statistical distribution distances. "
            "They frequently represent legitimate bulk B2B purchases, wholesale orders, or catalogue samples, "
            "and should not be interpreted as fraudulent without independent operational auditing."
        )
    }

    return anomalies_df, summary
