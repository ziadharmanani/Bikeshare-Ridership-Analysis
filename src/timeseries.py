"""
timeseries.py
Daily aggregation and time-series feature engineering for demand forecasting.

Reads the cleaned trip-level CSV produced by notebook 01 and builds a daily
time series with lag features, rolling statistics, cyclical encodings, and
calendar flags suitable for SARIMA / Prophet / XGBoost.
"""

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib

    _CA_HOLIDAYS = _holidays_lib.Canada(prov="ON")
except ImportError:
    _CA_HOLIDAYS = None


# ---------------------------------------------------------------------------
# 1. Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame, date_col: str = "Start Date") -> pd.DataFrame:
    """
    Collapse trip-level data into a daily time series.

    Returns a DataFrame indexed by date with columns:
        trip_count, mean_duration_min, median_duration_min,
        pct_annual_member, pct_peak_hour, pct_weekend
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = df.groupby(df[date_col].dt.date).agg(
        trip_count=("Trip_Duration_Min", "size"),
        mean_duration_min=("Trip_Duration_Min", "mean"),
        median_duration_min=("Trip_Duration_Min", "median"),
        pct_annual_member=("User_Type_Binary", "mean"),
        pct_peak_hour=("Peak_Hour_Binary", "mean"),
        pct_weekend=("Weekday_Binary", lambda x: 1 - x.mean()),
    )

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily = daily.sort_index()

    # Fill any gaps (unlikely but defensive)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range)
    daily.index.name = "date"

    # If a day had no trips (e.g. system outage), fill with 0 trips
    daily["trip_count"] = daily["trip_count"].fillna(0).astype(int)

    return daily


# ---------------------------------------------------------------------------
# 2. Calendar / cyclical features
# ---------------------------------------------------------------------------

def _cyclical_encode(series: pd.Series, period: int) -> pd.DataFrame:
    """Return sin and cos columns for a numeric series with given period."""
    angle = 2 * np.pi * series / period
    return pd.DataFrame({
        f"{series.name}_sin": np.sin(angle),
        f"{series.name}_cos": np.cos(angle),
    }, index=series.index)


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and cyclical features to a daily DataFrame (indexed by date).

    New columns:
        day_of_week (0=Mon), month, day_of_year,
        day_of_week_sin/cos, month_sin/cos, day_of_year_sin/cos,
        is_weekend, is_holiday
    """
    daily = daily.copy()
    idx = pd.DatetimeIndex(daily.index)

    daily["day_of_week"] = idx.dayofweek          # 0 = Monday
    daily["month"] = idx.month
    daily["day_of_year"] = idx.dayofyear

    # Cyclical sin/cos encodings
    for col, period in [("day_of_week", 7), ("month", 12), ("day_of_year", 365)]:
        cyc = _cyclical_encode(daily[col], period)
        daily = pd.concat([daily, cyc], axis=1)

    # Binary flags
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    if _CA_HOLIDAYS is not None:
        daily["is_holiday"] = idx.map(lambda d: int(d in _CA_HOLIDAYS))
    else:
        daily["is_holiday"] = 0

    return daily


# ---------------------------------------------------------------------------
# 3. Lag & rolling features
# ---------------------------------------------------------------------------

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_lag_features(
    daily: pd.DataFrame,
    target: str = "trip_count",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged versions of the target column."""
    daily = daily.copy()
    lags = lags or LAG_DAYS
    for lag in lags:
        daily[f"{target}_lag{lag}"] = daily[target].shift(lag)
    return daily


def add_rolling_features(
    daily: pd.DataFrame,
    target: str = "trip_count",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std of the target column."""
    daily = daily.copy()
    windows = windows or ROLLING_WINDOWS
    for w in windows:
        daily[f"{target}_rmean{w}"] = daily[target].shift(1).rolling(w).mean()
        daily[f"{target}_rstd{w}"] = daily[target].shift(1).rolling(w).std()
    return daily


# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------

def build_daily_features(
    df: pd.DataFrame,
    target: str = "trip_count",
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    End-to-end pipeline: aggregate → calendar → lags → rolling.

    Parameters
    ----------
    df : trip-level DataFrame (output of 01_Data_Cleaning)
    target : column to build lag/rolling features for
    lags : lag periods (default [1, 7, 14, 28])
    windows : rolling window sizes (default [7, 14, 28])
    drop_na : whether to drop rows with NaN from lag/rolling creation

    Returns
    -------
    pd.DataFrame indexed by date, ready for modeling.
    """
    daily = aggregate_daily(df)
    daily = add_calendar_features(daily)
    daily = add_lag_features(daily, target=target, lags=lags)
    daily = add_rolling_features(daily, target=target, windows=windows)

    if drop_na:
        daily = daily.dropna()

    return daily
