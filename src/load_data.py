"""
load_data.py
Loads and merges the 12 monthly Bike Share Toronto CSVs for 2023.
"""

import pandas as pd
from pathlib import Path


def load_bike_ridership_2023(data_folder=None):
    """
    Load all monthly Bike Share ridership CSVs for 2023.

    Args:
    data_folder : str or Path, optional
        Path to the folder containing the raw CSVs.
        Defaults to <project_root>/data/raw/.

    Returns:
    pd.DataFrame
        Combined dataframe indexed by Trip Id with columns:
        Trip Duration (seconds), Start Station Id, Start Time,
        Start Station Name, End Station Id, End Time,
        End Station Name, User Type.
    """
    if data_folder is None:
        project_root = Path(__file__).resolve().parent.parent
        data_folder = project_root / "data" / "raw"
    else:
        data_folder = Path(data_folder)

    months = [f"2023-{m:02d}" for m in range(1, 13)]

    # Expected column order after cleaning
    columns = [
        "Trip Id", "Trip Duration", "Start Station Id", "Start Time", 
        "Start Station Name", "End Station Id", "End Time",
        "End Station Name", "Bike Id", "User Type",
    ]

    all_dfs = []

    for month in months:
        file_path = data_folder / f"Bike share ridership {month}.csv"
        df_temp = pd.read_csv(file_path, encoding="latin1")

        # Clean BOM characters and whitespace from column names
        df_temp.columns = [col.replace("﻿", "").replace("ï»¿", "").strip()
                           for col in df_temp.columns]

        # Normalize inconsistent column name (double-space in some files)
        df_temp.rename(columns={"Trip  Duration": "Trip Duration"}, inplace=True)

        # Align all files to the same column set
        df_temp = df_temp.reindex(columns=columns)
        all_dfs.append(df_temp)

    df = pd.concat(all_dfs, ignore_index=True)

    # Drop columns that only appear in some years
    df = df.drop(columns=["Model", "Bike Id"], errors="ignore")

    # Set index
    df = df.set_index("Trip Id")

    return df
