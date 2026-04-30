# Bike Share Toronto Ridership Analysis (2023)
Originally developed as an academic team final project, this repository has since been independently revised and extended with improved code structure, modular source files, and a time-series forecasting pipeline.

## Overview
This project analyzes all 2023 trip data from [Bike Share Toronto](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/) — over 4 million rides across 12 months. The goal is to understand ridership patterns and build a daily demand forecasting model.

Key questions explored:

- When do people ride? (time of day, day of week, seasonality)
- Who rides? (annual members vs. casual users)
- What drives daily trip volume?
- Can we forecast demand accurately?

## Project Structure
```
Bikeshare-Ridership-Analysis/
│
├── data/
│   ├── raw/                         # Monthly CSVs from Bike Share Toronto (gitignored)
│   └── processed/                   # Cleaned and aggregated outputs (gitignored)
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb       # Load, merge and clean 12 monthly CSVs
│   ├── 02_EDA.ipynb                 # Exploratory data analysis
│   ├── 03_Feature_Engineering.ipynb # Build features for modeling
│   └── 04_Modeling.ipynb            # Time-series demand forecasting
│
├── src/
│   ├── __init__.py
│   ├── load_data.py                 # Load and merge raw monthly CSVs
│   ├── features.py                  # Feature engineering (timestamps, peak hours, encodings)
│   └── timeseries.py                # Daily aggregation, lag/rolling features, calendar flags
│
├── outputs/                         # Generated figures and model outputs
├── requirements.txt
└── .gitignore
```

## Setup
**1. Clone the repository**
```bash
git clone https://github.com/your-username/Bikeshare-Ridership-Analysis.git
cd Bikeshare-Ridership-Analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the raw data**
Download the 2023 monthly ridership CSVs from [Bike Share Toronto Open Data](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/) and place them in `data/raw/`. Files should follow the naming convention:

```
Bike share ridership 2023-01.csv
Bike share ridership 2023-02.csv
...
Bike share ridership 2023-12.csv
```

**4. Run the notebooks in order**
Start with `01_Data_Cleaning.ipynb` and work through to `04_Modeling.ipynb`.

## Key Features
- **Modular source code**: reusable functions in `src/` are imported across all notebooks, keeping analysis DRY and readable
- **Robust data loading**: handles encoding inconsistencies and column name variations across monthly files
- **IQR outlier removal**: removes docking errors and extreme trip durations
- **Time-series pipeline**: daily aggregation with lag features (1, 7, 14, 28 days), rolling statistics, cyclical calendar encodings, and Ontario public holiday flags
- **Binary encodings**: user type, peak hour, and weekday/weekend flags for model-ready features

## Dependencies

| Package | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| matplotlib, seaborn | Visualization |
| scikit-learn | Preprocessing and modeling |
| statsmodels | SARIMA / statistical models |
| holidays | Ontario public holiday detection |
