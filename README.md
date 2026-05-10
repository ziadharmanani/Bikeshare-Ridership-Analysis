# Bike Share Toronto Ridership Analysis (2020-2025)

An end-to-end data analysis and forecasting project using six years of Toronto Bike Share trip data. The project covers data cleaning, exploratory analysis,feature engineering, and a four-model forecasting comparison to predict daily ridership demand.

## The Story

Toronto Bike Share launched as a small pilot and grew into a major transit network over the past decade. COVID-19 caused a sharp ridership collapse in March 2020, followed by a gradual recovery that surpassed pre-pandemic levels by 2022. This project analyzes that arc and forecasts where demand is headed using ~28 million trip records.

## Project Structure

```
src/
  load_data_multi.py       Multi-year data loader with column normalization
  feature_engineering.py   Trip-level and daily-level feature pipelines

notebooks/
  01_Data_Cleaning.ipynb   Load, clean, filter, export to Parquet
  02_EDA.ipynb             Daily aggregation, temporal patterns, decomposition,
                           stationarity tests, ACF/PACF, correlations
  03_Feature_Engineering   Builds daily feature matrix (lags, rolling windows,
                           calendar/cyclical features)
  04_Modelling.ipynb       Four-model forecasting comparison

outputs/data/
  ridership.parquet        Daily aggregated features (~2,164 rows)
  ridership_clean.parquet  Trip-level cleaned data (~28M rows)
```

## Data

Source: [Toronto Open Data - Bike Share Ridership](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/)

Monthly CSVs covering January 2020 through December 2025. The data loader handles inconsistent column names, varying User Type labels (Member/Subscriber/Annual Member), encoding issues, and BOM characters across years. Trips are filtered to 60-3600 seconds to remove docking errors and extreme outliers.

## Feature Engineering

The pipeline builds features at two levels:

**Trip-level:** timestamp parsing (handles multiple date formats), trip duration in minutes, peak hour classification, user type binary encoding.

**Daily-level:** daily aggregation (trip count, mean/median duration, member percentage, peak hour percentage), calendar features (day of week, month, day of year with sine/cosine cyclical encoding, holidays, weekends), lag features (1, 7, 14, 28 days), and rolling window statistics (7, 14, 28-day mean and standard deviation).

## Modelling

Four models are compared on a 2025 hold-out test set (trained on 2020-2024):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA(1,1,1)(1,0,1,7) | 31,878 | 35,835 | 192.6% |
| Prophet | 5,711 | 6,693 | 110.7% |
| Holt-Winters (s=365) | 3,873 | 5,002 | 47.8% |
| XGBoost | 3,409 | 4,230 | 29.3% |

**SARIMA** with weekly seasonality (s=7) captured day-to-day autocorrelation but couldn't model the annual summer/winter cycle over a 365-day forecast horizon. Setting s=365 is computationally infeasible for SARIMAX.

**Prophet** captured the annual shape through automatic yearly/weekly decomposition and Canadian holiday regressors, but consistently underpredicted summer peaks due to COVID pulling the trend down.

**Holt-Winters** with annual seasonality (s=365) and multiplicative seasonal component delivered strong results by learning the full repeating annual pattern directly.

**XGBoost** performed best by treating forecasting as supervised regression over engineered features (lags, rolling averages, calendar variables), giving it access to both short-term autocorrelation and long-term seasonal patterns simultaneously.

## Setup

```bash
pip install -r requirements.txt
```

## Data Pipeline

1. Run `01_Data_Cleaning.ipynb` to download and clean raw data
2. Run `02_EDA.ipynb` for exploratory analysis
3. Run `03_Feature_Engineering.ipynb` to build the daily feature matrix
4. Run `04_Modelling.ipynb` for the forecasting comparison
