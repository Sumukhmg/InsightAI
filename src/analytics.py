"""InsightAI Core Analytics Engine
Deterministic calculations for sales, products, countries, customers, and time trends.
All calculations are pure Python / DuckDB without LLM hallucinations.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


def calculate_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates top-level executive KPIs deterministically."""
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "total_customers": 0,
            "total_products": 0,
            "total_quantity": 0,
            "avg_order_value": 0.0,
            "avg_items_per_order": 0.0
        }

    total_revenue = float(df["Revenue"].sum())
    total_orders = int(df["Invoice"].nunique())
    
    # Exclude nulls/blanks for valid customer count
    valid_customers = df["Customer ID"].dropna()
    valid_customers = valid_customers[valid_customers.astype(str).str.strip() != ""]
    total_customers = int(valid_customers.nunique())
    
    total_products = int(df["StockCode"].nunique())
    total_quantity = int(df["Quantity"].sum())
    avg_order_value = float(total_revenue / total_orders) if total_orders > 0 else 0.0
    avg_items_per_order = float(total_quantity / total_orders) if total_orders > 0 else 0.0

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "total_products": total_products,
        "total_quantity": total_quantity,
        "avg_order_value": avg_order_value,
        "avg_items_per_order": avg_items_per_order
    }


def get_monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates monthly revenue, orders, quantity, AOV, and MoM growth %."""
    if df.empty or "InvoiceDate" not in df.columns:
        return pd.DataFrame()

    # Aggregate by YearMonth
    monthly = (
        df.groupby("YearMonth")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Invoice", "nunique"),
            Quantity=("Quantity", "sum"),
            Customers=("Customer ID", "nunique")
        )
        .reset_index()
        .sort_values("YearMonth")
    )
    
    # Calculate Average Order Value
    monthly["AOV"] = monthly["Revenue"] / monthly["Orders"]
    
    # Calculate Month-over-Month Growth %
    monthly["Revenue_Growth_Pct"] = monthly["Revenue"].pct_change() * 100
    monthly["Orders_Growth_Pct"] = monthly["Orders"].pct_change() * 100
    
    # Replace NaN for first period
    monthly["Revenue_Growth_Pct"] = monthly["Revenue_Growth_Pct"].fillna(0.0)
    monthly["Orders_Growth_Pct"] = monthly["Orders_Growth_Pct"].fillna(0.0)

    return monthly


def get_country_analytics(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Calculates geographic performance by country."""
    if df.empty or "Country" not in df.columns:
        return pd.DataFrame()

    total_rev = df["Revenue"].sum()
    
    country_df = (
        df.groupby("Country")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Invoice", "nunique"),
            Quantity=("Quantity", "sum"),
            Customers=("Customer ID", "nunique")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    
    country_df["AOV"] = country_df["Revenue"] / country_df["Orders"]
    country_df["Revenue_Share_Pct"] = (country_df["Revenue"] / total_rev * 100) if total_rev > 0 else 0.0

    if top_n > 0 and len(country_df) > top_n:
        top_countries = country_df.head(top_n).copy()
        other_rev = country_df.iloc[top_n:]["Revenue"].sum()
        other_orders = country_df.iloc[top_n:]["Orders"].sum()
        other_qty = country_df.iloc[top_n:]["Quantity"].sum()
        other_custs = country_df.iloc[top_n:]["Customers"].sum()
        
        other_row = pd.DataFrame([{
            "Country": "Other Countries",
            "Revenue": other_rev,
            "Orders": other_orders,
            "Quantity": other_qty,
            "Customers": other_custs,
            "AOV": other_rev / other_orders if other_orders > 0 else 0.0,
            "Revenue_Share_Pct": (other_rev / total_rev * 100) if total_rev > 0 else 0.0
        }])
        return pd.concat([top_countries, other_row], ignore_index=True)

    return country_df


def get_product_performance_table(df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """Generates detailed product performance table with sorting metrics."""
    if df.empty:
        return pd.DataFrame()

    total_rev = df["Revenue"].sum()
    
    prod_df = (
        df.groupby(["StockCode", "Description"])
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Invoice", "nunique"),
            AvgPrice=("Price", "mean")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    
    prod_df["AOV"] = prod_df["Revenue"] / prod_df["Orders"]
    prod_df["Revenue_Share_Pct"] = (prod_df["Revenue"] / total_rev * 100) if total_rev > 0 else 0.0

    if top_n is not None:
        return prod_df.head(top_n)
    return prod_df


def get_bottom_products(df: pd.DataFrame, bottom_n: int = 15) -> pd.DataFrame:
    """Finds lowest revenue-generating products that had at least 1 order."""
    prod_df = get_product_performance_table(df)
    if prod_df.empty:
        return pd.DataFrame()
    return prod_df[prod_df["Revenue"] > 0].tail(bottom_n).sort_values("Revenue", ascending=True)


def get_top_customers(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Calculates top customers by total spend with behavioral metrics."""
    valid_df = df.dropna(subset=["Customer ID"]).copy()
    valid_df = valid_df[valid_df["Customer ID"].astype(str).str.strip() != ""]
    
    if valid_df.empty:
        return pd.DataFrame()

    total_rev = valid_df["Revenue"].sum()

    cust_df = (
        valid_df.groupby("Customer ID")
        .agg(
            Country=("Country", "first"),
            Revenue=("Revenue", "sum"),
            Orders=("Invoice", "nunique"),
            TotalUnits=("Quantity", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    
    cust_df["AOV"] = cust_df["Revenue"] / cust_df["Orders"]
    cust_df["Revenue_Share_Pct"] = (cust_df["Revenue"] / total_rev * 100) if total_rev > 0 else 0.0

    return cust_df.head(top_n)


def get_time_distribution(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyzes sales distribution by Day of Week and Hour of Day."""
    if df.empty:
        return {"weekday": pd.DataFrame(), "hourly": pd.DataFrame()}

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    weekday_df = (
        df.groupby("DayOfWeek")
        .agg(Revenue=("Revenue", "sum"), Orders=("Invoice", "nunique"))
        .reindex(day_order)
        .dropna()
        .reset_index()
    )

    hourly_df = (
        df.groupby("Hour")
        .agg(Revenue=("Revenue", "sum"), Orders=("Invoice", "nunique"))
        .reset_index()
        .sort_values("Hour")
    )

    return {"weekday": weekday_df, "hourly": hourly_df}
