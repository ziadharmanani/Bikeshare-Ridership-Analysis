"""
features.py
All feature engineering for the Bike Share ridership dataset.
Trip Duration is converted to minutes ONCE here, no downstream conversion needed.
"""

import numpy as np
import pandas as pd

# Peak-hour definitions
PEAK_MORNING = [7, 8, 9]
PEAK_EVENING = [15, 16, 17, 18, 19]


# Core feature engineering
def parse_timestamps(df):
    """
    Split the raw Start Time / End Time strings into separate
    date and time columns, and extract day/month/hour components.

    Expects raw columns: 'Start Time', 'End Time' (format: 'MM/DD/YYYY HH:MM').
    """
    df = df.copy()

    # Start Time
    parts = df["Start Time"].str.split(" ", expand=True)
    df["Start Date"] = pd.to_datetime(parts[0], format="%m/%d/%Y")
    df["Start Time"] = parts[1]  # keep only HH:MM

    df["Start Day"] = df["Start Date"].dt.day
    df["Start Month"] = df["Start Date"].dt.month
    df["Start Hour"] = pd.to_datetime(df["Start Time"], format="%H:%M").dt.hour

    # End Time
    parts = df["End Time"].str.split(" ", expand=True)
    df["End Date"] = pd.to_datetime(parts[0], format="%m/%d/%Y")
    df["End Time"] = parts[1]

    df["End Day"] = df["End Date"].dt.day
    df["End Month"] = df["End Date"].dt.month

    return df


def add_derived_features(df):
    """
    Add Weekday/Weekend flag, Peak Hour category, and convert
    Trip Duration from seconds to minutes.
    """
    df = df.copy()

    # Trip Duration: seconds to minutes (single conversion point)
    df["Trip_Duration_Min"] = df["Trip Duration"] / 60

    # Weekday / Weekend
    df["Weekday_Weekend"] = df["Start Date"].dt.weekday.apply(
        lambda x: "Weekday" if x < 5 else "Weekend"
    )

    # Peak Hour (categorical)
    conditions = [
        df["Start Hour"].isin(PEAK_MORNING),
        df["Start Hour"].isin(PEAK_EVENING),
    ]
    choices = ["Morning", "Evening"]
    df["Peak_Hour"] = np.select(conditions, choices, default="Off Peak")

    # Display-friendly duration
    df["Trip_Duration_MMSS"] = df["Trip Duration"].apply(
        lambda s: f"{s // 60}:{s % 60:02d}"
    )

    return df


def reorder_columns(df):
    """Reorder columns into a logical reading order."""
    order = [
        "Trip_Duration_Min", "Trip_Duration_MMSS", "Trip Duration",
        "Start Station Id", "Start Station Name",
        "Start Time", "Start Hour", "Start Day", "Start Month", "Start Date",
        "End Station Id", "End Station Name",
        "End Time", "End Day", "End Month", "End Date",
        "User Type", "Weekday_Weekend", "Peak_Hour",
    ]
    # Keep only columns that exist (in case some were dropped)
    order = [c for c in order if c in df.columns]
    return df[order]

# Outlier removal (IQR-based — robust to skew)
def remove_outliers_iqr(df, column="Trip_Duration_Min", factor=1.5):
    """
    Remove outliers using the Inter-Quartile Range method.

    Also drops trips with duration <= 0 (docking errors).

    Parameters
    ----------
    factor : float
        IQR multiplier.  1.5 = standard, 3.0 = extreme-only.

    Returns
    -------
    pd.DataFrame with outliers removed and index reset.
    """
    df = df.copy()

    # Drop zero/negative durations (docking errors)
    df = df[df[column] > 0]

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    after = len(df)

    print(f"IQR outlier removal ({column}): {before - after:,} rows removed "
          f"({(before - after) / before:.2%}), {after:,} rows remaining.")
    print(f"  Bounds: [{lower:.2f}, {upper:.2f}] minutes")

    return df.reset_index(drop=True)

# Binary encodings for modeling
def add_binary_features(df):
    """Add binary-encoded versions of categorical columns for modeling."""
    df = df.copy()
    df["Peak_Hour_Binary"] = df["Peak_Hour"].map(
        {"Off Peak": 0, "Morning": 1, "Evening": 1}
    )
    df["Weekday_Binary"] = df["Weekday_Weekend"].map(
        {"Weekend": 0, "Weekday": 1}
    )
    df["User_Type_Binary"] = df["User Type"].map(
        {"Casual Member": 0, "Annual Member": 1}
    )
    return df

# Log transform
def add_log_duration(df):
    """Add log-transformed trip duration (log1p to handle near-zero values)."""
    df = df.copy()
    df["Log_Trip_Duration"] = np.log1p(df["Trip_Duration_Min"])
    return df

# Run full feature engineering pipeline
def engineer_features(df, remove_outliers=True, iqr_factor=1.5):
    """
    Run the complete feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame from load_bike_ridership_2023()
    remove_outliers : bool
    iqr_factor : float

    Returns
    -------
    pd.DataFrame fully engineered and ready for EDA / modeling.
    """
    df = parse_timestamps(df)
    df = add_derived_features(df)
    df = reorder_columns(df)

    if remove_outliers:
        df = remove_outliers_iqr(df, factor=iqr_factor)

    df = add_binary_features(df)
    df = add_log_duration(df)

    return df