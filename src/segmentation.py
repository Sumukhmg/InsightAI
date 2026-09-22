"""InsightAI Customer Analytics & Machine Learning Segmentation Module
Implements RFM scoring, heuristic segment mapping, and unsupervised K-Means clustering with PCA.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def calculate_rfm(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Computes Recency, Frequency, and Monetary (RFM) metrics per customer.
    
    Args:
        df: Cleaned analytical transaction DataFrame.
        reference_date: Reference timestamp for recency calculation.
            If None, uses max(InvoiceDate) + 1 day.
            
    Returns:
        pd.DataFrame: Customer-level RFM metrics and quintile scores.
    """
    valid_df = df.dropna(subset=["Customer ID"]).copy()
    valid_df = valid_df[valid_df["Customer ID"].astype(str).str.strip() != ""]
    
    if valid_df.empty:
        return pd.DataFrame()

    if reference_date is None:
        reference_date = valid_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # 1. Aggregate RFM metrics
    rfm = (
        valid_df.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            Country=("Country", "first"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max")
        )
        .reset_index()
    )
    
    # Filter out customers with 0 or negative monetary totals
    rfm = rfm[rfm["Monetary"] > 0].copy()

    # 2. Quintile Scoring (1 to 5)
    # Recency: lower days = higher score (5)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    
    # Frequency: higher count = higher score (5)
    # Use rank-based qcut to handle tie frequencies cleanly
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Monetary: higher spend = higher score (5)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) +
        rfm["F_Score"].astype(str) +
        rfm["M_Score"].astype(str)
    )

    # 3. Rule-Based Segment Categorization
    rfm["Segment"] = rfm.apply(_assign_rfm_segment, axis=1)

    # 4. Approximate Customer Lifetime Value (CLV proxy)
    # Simple CLV proxy: AOV * Purchase Frequency * Customer Tenure Factor
    tenure_days = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days + 1
    rfm["TenureDays"] = tenure_days
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["CLV_Proxy"] = rfm["Monetary"] * (1 + np.log1p(rfm["Frequency"]))

    return rfm


def _assign_rfm_segment(row: pd.Series) -> str:
    """Assigns standard retail business segments based on RFM score combinations."""
    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New & Promising"
    elif r >= 3 and f <= 3 and m >= 3:
        return "Potential Loyalists"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2 and m >= 3:
        return "Need Attention"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost Customers"
    else:
        return "Occasional Shoppers"


def get_rfm_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes customer counts, revenue, and behavior across RFM segments."""
    if rfm_df.empty:
        return pd.DataFrame()

    total_customers = len(rfm_df)
    total_rev = rfm_df["Monetary"].sum()

    summary = (
        rfm_df.groupby("Segment")
        .agg(
            Customer_Count=("Customer ID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Revenue=("Monetary", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Recency=("Recency", "mean"),
            Avg_AOV=("AOV", "mean")
        )
        .reset_index()
        .sort_values("Total_Revenue", ascending=False)
    )
    
    summary["Customer_Share_Pct"] = summary["Customer_Count"] / total_customers * 100
    summary["Revenue_Share_Pct"] = summary["Total_Revenue"] / total_rev * 100

    return summary


def run_kmeans_segmentation(
    rfm_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs K-Means clustering on standardized log-transformed RFM features with PCA projection.
    
    Args:
        rfm_df: DataFrame output from calculate_rfm.
        n_clusters: Number of clusters (K).
        random_state: Reproducibility seed.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - rfm_df with 'Cluster' and PCA coordinates added
            - Cluster profile metrics and persona descriptions
    """
    if rfm_df.empty or len(rfm_df) < n_clusters:
        return rfm_df, {}

    df_cluster = rfm_df.copy()
    
    # 1. Log Transform to normalize positive-skewed retail distributions
    features = ["Recency", "Frequency", "Monetary"]
    X_log = np.log1p(df_cluster[features])

    # 2. Standardize Features (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # 3. Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    # 4. PCA for 2D Projection
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df_cluster["PCA1"] = coords[:, 0]
    df_cluster["PCA2"] = coords[:, 1]
    
    explained_variance = float(pca.explained_variance_ratio_.sum() * 100)

    # 5. Calculate Real-World Centroid Profiles
    cluster_profiles = []
    for c in range(n_clusters):
        c_subset = df_cluster[df_cluster["Cluster"] == c]
        count = len(c_subset)
        pct = count / len(df_cluster) * 100
        avg_r = float(c_subset["Recency"].mean())
        avg_f = float(c_subset["Frequency"].mean())
        avg_m = float(c_subset["Monetary"].mean())
        tot_m = float(c_subset["Monetary"].sum())

        persona = _generate_cluster_persona(avg_r, avg_f, avg_m, df_cluster)

        cluster_profiles.append({
            "cluster_id": c,
            "persona_name": persona["title"],
            "description": persona["description"],
            "strategy": persona["strategy"],
            "customer_count": count,
            "customer_share_pct": pct,
            "avg_recency_days": avg_r,
            "avg_frequency_orders": avg_f,
            "avg_monetary_spend": avg_m,
            "total_spend": tot_m
        })

    metadata = {
        "n_clusters": n_clusters,
        "explained_variance_pct": explained_variance,
        "cluster_profiles": cluster_profiles
    }

    return df_cluster, metadata


def _generate_cluster_persona(
    avg_r: float,
    avg_f: float,
    avg_m: float,
    pop_df: pd.DataFrame
) -> Dict[str, str]:
    """Generates an intuitive commercial persona based on cluster relative metrics."""
    med_r = pop_df["Recency"].median()
    med_f = pop_df["Frequency"].median()
    med_m = pop_df["Monetary"].median()

    if avg_m >= med_m * 2.0 and avg_f >= med_f * 2.0:
        return {
            "title": "High-Value VIPs",
            "description": f"Frequent premium buyers (£{avg_m:,.0f} avg spend, {avg_f:.1f} orders). Highest lifetime revenue driver.",
            "strategy": "Exclusive loyalty perks, white-glove account service, early product access."
        }
    elif avg_r <= med_r * 0.7 and avg_f <= med_f:
        return {
            "title": "Recent New Customers",
            "description": f"Shopped recently ({avg_r:.0f} days ago) but currently single or low-order frequency ({avg_f:.1f} orders).",
            "strategy": "Onboarding nurture sequences, personalized product recommendations, second-purchase incentives."
        }
    elif avg_r >= med_r * 1.5 and avg_m >= med_m:
        return {
            "title": "High-Spend Churn Risk",
            "description": f"Historically valuable customers (£{avg_m:,.0f} spend) who have not purchased in {avg_r:.0f} days.",
            "strategy": "Win-back discount campaigns, direct feedback outreach to discover churn root causes."
        }
    else:
        return {
            "title": "Low-Engagement Casuals",
            "description": f"Infrequent buyers with modest spend (£{avg_m:,.0f} spend, {avg_r:.0f} days since last order).",
            "strategy": "Automated seasonal promotional blasts, clearance alerts, price-sensitive offers."
        }
