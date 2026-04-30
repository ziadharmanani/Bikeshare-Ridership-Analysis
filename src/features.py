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
    Split Start Time and End Time into separate date and time columns.
    """
    df = df.copy()

    # Split "MM/DD/YYYY HH:MM" into date part and time part
    start_parts = df["Start Time"].str.split(" ", expand=True)
    df["Start Date"] = pd.to_datetime(start_parts[0], format="%m/%d/%Y")
    df["Start Time"] = start_parts[1]

    df["Start Day"] = df["Start Date"].dt.day
    df["Start Month"] = df["Start Date"].dt.month
    df["Start Hour"] = pd.to_datetime(df["Start Time"], format="%H:%M").dt.hour

    end_parts = df["End Time"].str.split(" ", expand=True)
    df["End Date"] = pd.to_datetime(end_parts[0], format="%m/%d/%Y")
    df["End Time"] = end_parts[1]

    df["End Day"] = df["End Date"].dt.day
    df["End Month"] = df["End Date"].dt.month

    return df


def add_derived_features(df):
    """
    Add trip duration in minutes, weekday/weekend label, and peak hour category.
    """
    df = df.copy()

    # Convert trip duration from seconds to minutes
    df["Trip_Duration_Min"] = df["Trip Duration"] / 60

    # Label each trip as Weekday or Weekend
    def get_weekday_weekend(date):
        if date.weekday() < 5:
            return "Weekday"
        else:
            return "Weekend"

    df["Weekday_Weekend"] = df["Start Date"].apply(get_weekday_weekend)

    # Label each trip by whether it falls in a peak hour
    def get_peak_hour(hour):
        if hour in PEAK_MORNING:
            return "Morning"
        elif hour in PEAK_EVENING:
            return "Evening"
        else:
            return "Off Peak"

    df["Peak_Hour"] = df["Start Hour"].apply(get_peak_hour)

    # Format duration as a readable MM:SS string
    def to_mmss(seconds):
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"

    df["Trip_Duration_MMSS"] = df["Trip Duration"].apply(to_mmss)

    return df


def reorder_columns(df):
    """
    Reorder columns into a logical reading order.
    """
    order = [
        "Trip_Duration_Min", "Trip_Duration_MMSS", "Trip Duration",
        "Start Station Id", "Start Station Name",
        "Start Time", "Start Hour", "Start Day", "Start Month", "Start Date",
        "End Station Id", "End Station Name",
        "End Time", "End Day", "End Month", "End Date",
        "User Type", "Weekday_Weekend", "Peak_Hour",
    ]
    # Keep only columns that actually exist in the dataframe
    order = [c for c in order if c in df.columns]
    return df[order]

# Outlier removal (IQR-based — robust to skew)
def remove_outliers_iqr(df, column="Trip_Duration_Min", factor=1.5):
    """
    Remove outliers from a column using the IQR method.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column to check for outliers.
        factor (float): How strict the cutoff is. 1.5 is standard, 3.0 keeps more data.

    Returns:
        pd.DataFrame: Dataframe with outliers removed and index reset.
    """
    df = df.copy()

    # Remove trips with zero or negative duration (likely docking errors)
    df = df[df[column] > 0]

    # Calculate the IQR boundaries
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
    """
    Add binary (0 or 1) versions of categorical columns for modeling.
    """
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
    """
    Add a log-transformed version of trip duration.
    """
    df = df.copy()
    df["Log_Trip_Duration"] = np.log1p(df["Trip_Duration_Min"])
    return df

# Run full feature engineering pipeline
def engineer_features(df, remove_outliers=True, iqr_factor=1.5):
    """
    Run the full feature engineering pipeline on the raw ridership data.

    Args:
        df (pd.DataFrame): Raw ridership data from load_bike_ridership_2023().
        remove_outliers (bool): Whether to remove outlier trips. Defaults to True.
        iqr_factor (float): IQR multiplier for outlier removal. Defaults to 1.5.

    Returns:
        pd.DataFrame: Fully engineered dataframe ready for analysis and modeling.
    """
    df = parse_timestamps(df)
    df = add_derived_features(df)
    df = reorder_columns(df)

    if remove_outliers:
        df = remove_outliers_iqr(df, factor=iqr_factor)

    df = add_binary_features(df)
    df = add_log_duration(df)

    return df