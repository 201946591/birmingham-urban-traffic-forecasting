# ==============================================================================
# Spatiotemporal Data Fusion: Merging Traffic, Meteorological & Holiday Feeds
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Merges the road-type stratified traffic time series with:
# 1. Hourly Open-Meteo ERA5 weather variables (temperature, precipitation)
# 2. Official UK bank holiday calendar indicators
# Produces the spatiotemporal master feature table used in Section 3.6 & 4.4.

import os
import sys
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("PHASE 4 — SCRIPT 3: EXOGENOUS DATA FUSION")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load zonal traffic data
    zonal_path = os.path.join(script_dir, "zonal_traffic.csv")
    if not os.path.exists(zonal_path):
        print("[ERROR] zonal_traffic.csv not found! Run 02_spatial_data_prep.py first.")
        sys.exit(1)

    print("[INFO] Ingesting road-type stratified traffic series...")
    traffic_df = pd.read_csv(zonal_path)
    traffic_df["datetime"] = pd.to_datetime(traffic_df["datetime"])
    print(f"  - Traffic records: {traffic_df.shape[0]:,} rows x {traffic_df.shape[1]} columns")

    # 2. Ingest weather data
    weather_path = os.path.join(script_dir, "birmingham_weather.csv")
    if not os.path.exists(weather_path):
        print("[ERROR] birmingham_weather.csv not found! Run 01_exogenous_data_fetch.py first.")
        sys.exit(1)

    print("[INFO] Ingesting meteorological archive...")
    weather_df = pd.read_csv(weather_path)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    print(f"  - Weather records: {weather_df.shape[0]:,} rows x {weather_df.shape[1]} columns")

    # 3. Ingest UK statutory bank holidays
    holiday_path = os.path.join(script_dir, "uk_holidays.csv")
    if not os.path.exists(holiday_path):
        print("[ERROR] uk_holidays.csv not found! Run 01_exogenous_data_fetch.py first.")
        sys.exit(1)

    print("[INFO] Ingesting UK statutory holiday calendar...")
    hol_df = pd.read_csv(holiday_path)
    hol_df["date"] = pd.to_datetime(hol_df["date"])
    print(f"  - Bank holiday dates: {len(hol_df)} cataloged")

    # 4. Feature engineering: Calendar and proximity effects
    # DfT manual survey counts are intentionally conducted on typical neutral weekdays
    # (Tuesday-Thursday, 07:00-19:00) and avoid bank holidays. Hence is_holiday is largely 0.
    # We engineer a 'near_holiday' proximity flag to evaluate whether holiday eve or post-holiday
    # travel surges affect suburban and arterial traffic differently.
    if traffic_df["datetime"].dt.tz is not None:
        traffic_df["datetime"] = traffic_df["datetime"].dt.tz_convert(None)

    traffic_df["date_only"] = traffic_df["datetime"].dt.normalize()
    holiday_dates_set = set(hol_df["date"].dt.normalize())
    holiday_dates_list = sorted(hol_df["date"].dt.normalize().tolist())

    traffic_df["is_holiday"] = traffic_df["date_only"].apply(
        lambda d: 1 if d in holiday_dates_set else 0
    )

    def is_near_holiday(d):
        for h in holiday_dates_list:
            diff = abs((d - h).days)
            if diff <= 1:
                return 1
        return 0

    traffic_df["near_holiday"] = traffic_df["date_only"].apply(is_near_holiday)
    traffic_df = traffic_df.drop(columns=["date_only"])

    near_hol_count = traffic_df["near_holiday"].sum()
    print(f"  - Traffic observations adjacent to bank holidays (+/- 1 day): {near_hol_count:,} ({near_hol_count / len(traffic_df) * 100:.2f}%)")

    # 5. Spatiotemporal join: Merge weather onto traffic series
    print("\n[INFO] Joining hourly meteorological observations onto traffic records...")
    weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)
    merged_df = traffic_df.merge(weather_df, on="datetime", how="left")

    weather_nulls = merged_df["temperature_2m"].isnull().sum()
    if weather_nulls > 0:
        print(f"  - Imputing {weather_nulls:,} unmatched meteorological entries via forward/backward fill.")
        merged_df["temperature_2m"] = merged_df["temperature_2m"].ffill().bfill()
        merged_df["precipitation"] = merged_df["precipitation"].ffill().bfill()
    else:
        print("  - All traffic timestamps aligned with weather feeds without missing values.")

    # 6. Export spatiotemporal master feature table
    output_path = os.path.join(script_dir, "spatiotemporal_master.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("SPATIOTEMPORAL DATA FUSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  [SAVED] spatiotemporal_master.csv ({merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns)")
    print(f"  - Temperature range:   {merged_df['temperature_2m'].min():.1f}°C to {merged_df['temperature_2m'].max():.1f}°C")
    print(f"  - Precipitation range: 0.00 mm to {merged_df['precipitation'].max():.1f} mm/hr")
    print("=" * 60)


if __name__ == "__main__":
    main()
