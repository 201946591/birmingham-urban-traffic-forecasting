# Data Acquisition and Provenance

This directory contains the preprocessed feature tables, spatial panels, and exogenous environmental datasets utilized in the dissertation.

## 1. Raw Department for Transport (DfT) Count Point Data
- **File**: `dft_rawcount_local_authority_id_141.csv`
- **Source**: UK Department for Transport Open Data Portal ([roadtraffic.dft.gov.uk](https://roadtraffic.dft.gov.uk))
- **Coverage**: Birmingham Local Authority (LA 141), 2000–2025
- **Observations**: 72,948 hourly daytime records across 23 attributes
- **Licence**: Open Government Licence (OGL) v3.0

To re-download the raw counts directly:
1. Navigate to [roadtraffic.dft.gov.uk/local-authorities/141](https://roadtraffic.dft.gov.uk/local-authorities/141).
2. Download the raw count point data CSV file.
3. Place the file in the project root or `data/` directory.

## 2. Processed Datasets Included in this Directory

| File Name | Rows | Columns | Description |
|:---|:---:|:---:|:---|
| `engineered_birmingham_traffic.csv` | 27,774 | 15 | City-wide aggregated hourly series with autoregressive lags ($t-1, t-2, t-3, t-24$) and cyclical temporal encodings. |
| `zonal_traffic.csv` | 27,774 | 15 | Road-type stratified panel separating Major arterial corridors and Minor residential streets. |
| `birmingham_weather.csv` | 227,928 | 3 | Hourly ERA5 reanalysis temperature (°C) and precipitation (mm) from Open-Meteo (2000–2025). |
| `uk_holidays.csv` | 208 | 2 | Official UK statutory bank holiday calendar dates (England & Wales). |
