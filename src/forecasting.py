"""InsightAI Sales Forecasting Module
Aggregates monthly revenue, constructs autoregressive lag features, and forecasts
the next 3 months using Random Forest or Linear Regression.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def build_monthly_revenue_forecast(
    df: pd.DataFrame,
    forecast_horizon_months: int = 3,
    model_type: str = "Random Forest",
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Builds a deterministic 3-month sales revenue forecast using autoregressive lag features.
    
    Args:
        df: Cleaned analytical transaction DataFrame.
        forecast_horizon_months: Number of future months to forecast (default: 3).
        model_type: 'Random Forest' or 'Linear Regression'.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Combined historical and forecast DataFrame
            - Forecast performance metrics and directional trajectory metadata
    """
    if df.empty or "InvoiceDate" not in df.columns:
        return pd.DataFrame(), {}

    # 1. Aggregate Revenue by Month
    monthly = (
        df.groupby("YearMonth")
        .agg(Revenue=("Revenue", "sum"), Orders=("Invoice", "nunique"))
        .reset_index()
        .sort_values("YearMonth")
    )
    
    if len(monthly) < 6:
        # Not enough history for reliable autoregressive lag modeling
        return pd.DataFrame(), {"error": "Insufficient historical months (minimum 6 required)."}

    # 2. Build Autoregressive Lag Features
    # Create temporal index
    monthly["MonthIndex"] = np.arange(len(monthly))
    monthly["Lag_1"] = monthly["Revenue"].shift(1)
    monthly["Lag_2"] = monthly["Revenue"].shift(2)
    monthly["Rolling_3M"] = monthly["Revenue"].shift(1).rolling(2, min_periods=1).mean()

    # Drop early rows with NaN lags for model training
    train_data = monthly.dropna().copy()
    
    features = ["MonthIndex", "Lag_1", "Lag_2", "Rolling_3M"]
    X = train_data[features]
    y = train_data["Revenue"]

    # 3. Model Training
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    else:
        model = LinearRegression()

    model.fit(X, y)
    
    # In-sample error
    y_pred_in = model.predict(X)
    mape = float(mean_absolute_percentage_error(y, y_pred_in) * 100)

    # 4. Multi-step Recursive Out-of-Sample Forecasting
    last_year_month = pd.Period(monthly["YearMonth"].iloc[-1], freq="M")
    current_index = monthly["MonthIndex"].iloc[-1]
    
    # Buffer for recursive lags
    history_rev = list(monthly["Revenue"])
    
    future_rows = []
    for step in range(1, forecast_horizon_months + 1):
        future_period = str(last_year_month + step)
        future_idx = current_index + step
        lag_1 = history_rev[-1]
        lag_2 = history_rev[-2] if len(history_rev) >= 2 else lag_1
        rolling_3m = np.mean(history_rev[-2:])
        
        X_step = pd.DataFrame([{
            "MonthIndex": future_idx,
            "Lag_1": lag_1,
            "Lag_2": lag_2,
            "Rolling_3M": rolling_3m
        }])
        
        pred_rev = float(max(0.0, model.predict(X_step)[0]))
        history_rev.append(pred_rev)
        
        future_rows.append({
            "YearMonth": future_period,
            "Revenue": pred_rev,
            "Orders": np.nan,
            "Type": "Forecast"
        })

    # 5. Format Combined Output
    historical_df = monthly[["YearMonth", "Revenue", "Orders"]].copy()
    historical_df["Type"] = "Historical"
    
    forecast_df = pd.DataFrame(future_rows)
    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)

    # 6. Directional Trajectory Analysis
    last_actual_rev = monthly["Revenue"].iloc[-1]
    avg_forecast_rev = np.mean([r["Revenue"] for r in future_rows])
    growth_rate = ((avg_forecast_rev - last_actual_rev) / last_actual_rev * 100) if last_actual_rev > 0 else 0.0

    if growth_rate > 3.0:
        direction = "Increasing"
        badge_class = "badge-success"
    elif growth_rate < -3.0:
        direction = "Decreasing"
        badge_class = "badge-danger"
    else:
        direction = "Stable"
        badge_class = "badge-primary"

    metadata = {
        "model_type": model_type,
        "forecast_horizon_months": forecast_horizon_months,
        "in_sample_mape": mape,
        "last_actual_revenue": float(last_actual_rev),
        "forecast_avg_revenue": float(avg_forecast_rev),
        "projected_growth_pct": float(growth_rate),
        "expected_direction": direction,
        "badge_class": badge_class,
        "disclaimer": (
            "NOTICE: The figures above are algorithmic predictions produced by an autoregressive machine learning model. "
            "They are probabilistic forecasts subject to macro shocks, seasonal retail calendar shifts, and inventory constraints. "
            "They should be treated as guidance scenarios rather than established facts."
        )
    }

    return combined_df, metadata
