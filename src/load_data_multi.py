"""
load_data_multi.py
Multi-year data loader for Bike Share Toronto ridership data (2015–2026).

Downloads ZIP archives from the Toronto Open Data API, extracts monthly CSVs,
normalizes column names and User Type labels across all years, and returns a
single unified DataFrame.

Usage:
    from src.load_data_multi import load_bike_ridership

    # Load all available years
    df = load_bike_ridership()

    # Load specific years
    df = load_bike_ridership(years=[2019, 2020, 2021, 2022, 2023])

    # Load with SQL pre-filter (runs against in-memory SQLite)
    df = load_bike_ridership(
        years=[2023],
        sql_filter='SELECT * FROM ridership WHERE "Trip Duration" BETWEEN 60 AND 3600'
    )
"""

import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile


# Column name mapping: all known variants → canonical names
# Canonical schema: Trip Id, Trip Duration, Start Station Id, Start Time, Start Station Name, End Station Id, End Time, End Station Name, User Type
_COLUMN_MAP = {
    # Trip Id
    "trip_id": "Trip Id",
    "Trip Id": "Trip Id",
    "Trip_Id": "Trip Id",

    # Trip Duration (always in seconds in raw data)
    "trip_duration_seconds": "Trip Duration",
    "Trip Duration": "Trip Duration",
    "Trip  Duration": "Trip Duration",
    "trip_duration": "Trip Duration",
    "tripduration": "Trip Duration",
    "Trip_Duration": "Trip Duration",

    # Start Station Id
    "from_station_id": "Start Station Id",
    "Start Station Id": "Start Station Id",
    "start_station_id": "Start Station Id",
    "Start_Station_Id": "Start Station Id",

    # Start Time
    "trip_start_time": "Start Time",
    "Start Time": "Start Time",
    "start_time": "Start Time",
    "Start_Time": "Start Time",

    # Start Station Name
    "from_station_name": "Start Station Name",
    "Start Station Name": "Start Station Name",
    "start_station_name": "Start Station Name",
    "Start_Station_Name": "Start Station Name",

    # End Station Id
    "to_station_id": "End Station Id",
    "End Station Id": "End Station Id",
    "end_station_id": "End Station Id",
    "End_Station_Id": "End Station Id",

    # End Time
    "trip_stop_time": "End Time",
    "End Time": "End Time",
    "end_time": "End Time",
    "End_Time": "End Time",

    # End Station Name
    "to_station_name": "End Station Name",
    "End Station Name": "End Station Name",
    "end_station_name": "End Station Name",
    "End_Station_Name": "End Station Name",

    # User Type
    "user_type": "User Type",
    "User Type": "User Type",
    "usertype": "User Type",
    "Member Type": "User Type",
    "User_Type": "User Type",

    # Bike Id (not in canonical schema but preserved if present)
    "bike_id": "Bike Id",
    "Bike Id": "Bike Id",
    "Bike_Id": "Bike Id",
}

# Canonical columns to keep (everything else is dropped after normalization)
_CANONICAL_COLUMNS = {
    "Trip Id", "Trip Duration", "Start Station Id", "Start Time",
    "Start Station Name", "End Station Id", "End Time",
    "End Station Name", "User Type", "Bike Id", "year",
}

# User Type label normalization
_USER_TYPE_MAP = {
    "Annual Member": "Annual Member",
    "Member": "Annual Member",
    "Subscriber": "Annual Member",
    "member": "Annual Member",
    "Casual Member": "Casual Member",
    "Casual": "Casual Member",
    "Customer": "Casual Member",
    "casual": "Casual Member",
}


def get_ridership_urls():
    """
    Query the Toronto Open Data API and return a dict of {year: download_url}
    for all available ridership ZIP files.
    """
    package_id = "bike-share-toronto-ridership-data"
    api_url = f"https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show?id={package_id}"

    response = requests.get(api_url)
    if response.status_code != 200:
        raise ConnectionError("Could not connect to Toronto Open Data API.")

    resources = response.json()["result"]["resources"]
    urls = {}

    for res in resources:
        name = res["name"].lower()
        if "ridership" in name and res["format"].lower() == "zip":
            for year in range(2015, 2027):
                if str(year) in name:
                    urls[year] = res["url"]

    return urls


def _normalize_columns(df):
    """
    Rename columns to the canonical schema using _COLUMN_MAP.
    Handles BOM characters and extra whitespace.
    """
    # Strip BOM and whitespace from column names
    df.columns = [col.replace("﻿", "").replace("ï»¿", "").strip() for col in df.columns]

    # Apply the mapping (only rename columns that exist in the map)
    rename_dict = {}
    for col in df.columns:
        if col in _COLUMN_MAP:
            rename_dict[col] = _COLUMN_MAP[col]

    df = df.rename(columns=rename_dict)
    return df


def _normalize_user_type(df):
    """
    Normalize User Type labels to 'Annual Member' or 'Casual Member'.
    Unknown labels are left unchanged.
    """
    if "User Type" not in df.columns:
        return df
    df["User Type"] = df["User Type"].map(_USER_TYPE_MAP).fillna(df["User Type"])
    return df


def _load_single_year(url, year, duration_range=None):
    """
    Download one year's ZIP, extract CSVs, normalize columns and User Type.
    Optionally filter by trip duration per-file to keep memory low.
    Returns a list of DataFrames (one per monthly CSV).
    """
    print(f"  Downloading {year}...")
    response = requests.get(url)
    dfs = []

    with ZipFile(BytesIO(response.content)) as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(".csv") and not file_info.filename.startswith("__MACOSX"):
                with z.open(file_info) as f:
                    df_temp = pd.read_csv(f, encoding="latin1")
                    df_temp = _normalize_columns(df_temp)
                    df_temp = _normalize_user_type(df_temp)
                    df_temp["year"] = year

                    # Drop columns not in the canonical schema (e.g. Bike_Model)
                    df_temp = df_temp[[c for c in df_temp.columns if c in _CANONICAL_COLUMNS]]

                    # Apply duration filter per-file if provided
                    if duration_range is not None and "Trip Duration" in df_temp.columns:
                        lo, hi = duration_range
                        df_temp = df_temp[
                            (df_temp["Trip Duration"] >= lo) & (df_temp["Trip Duration"] <= hi)
                        ]

                    dfs.append(df_temp)

    return dfs


def load_bike_ridership(years=None, duration_range=(60, 3600)):
    """
    Load Bike Share Toronto ridership data for one or more years.

    Parameters
    ----------
    years : list of int, optional
        Which years to load. Defaults to all available years from the API.
    duration_range : tuple of (int, int), optional
        (min_seconds, max_seconds) filter applied per-file during loading.
        Defaults to (60, 3600). Pass None to keep all rows.

    Returns
    -------
    pd.DataFrame
        Combined ridership data with normalized columns and a 'year' column.
    """
    urls = get_ridership_urls()

    if years is None:
        years = sorted(urls.keys())
    else:
        # Validate requested years
        missing = [y for y in years if y not in urls]
        if missing:
            print(f"Warning: no data found for years {missing}. Skipping.")
        years = [y for y in years if y in urls]

    if not years:
        print("No valid years to load.")
        return pd.DataFrame()

    all_dfs = []

    for year in years:
        year_dfs = _load_single_year(urls[year], year, duration_range=duration_range)
        all_dfs.extend(year_dfs)

    if not all_dfs:
        return pd.DataFrame()

    # Combine all monthly CSVs into one DataFrame
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")

    # Set Trip Id as index if present
    if "Trip Id" in df.columns:
        df = df.set_index("Trip Id")

    print(f"Successfully loaded {len(df):,} rows.")
    return df
