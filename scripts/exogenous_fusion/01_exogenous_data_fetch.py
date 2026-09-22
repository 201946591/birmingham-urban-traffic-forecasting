# ==============================================================================
# Exogenous Meteorological & Calendar Data Acquisition
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Sourcing:
# 1. Weather: Open-Meteo Historical Weather API (ECMWF ERA5 reanalysis archive)
#    - Licence: CC BY 4.0 (Creative Commons Attribution 4.0 International)
#    - Citation: Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo.
#    - Coordinates: Birmingham, UK (52.4862°N, 1.8904°W)
#    - Features: Hourly 2-metre temperature (°C) and precipitation (mm)
# 2. Calendar: Official UK Bank Holidays (GOV.UK calendar via Python holidays library)

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import requests
import holidays


def fetch_weather_data(start_year=2000, end_year=2025):
    """
    Fetch hourly historical weather from Open-Meteo for Birmingham, UK.
    Fetches year-by-year blocks to remain within public rate limits.
    Returns DataFrame with columns: [datetime, temperature_2m, precipitation]
    """
    # Birmingham central coordinates corresponding to Birmingham Local Authority count centroid
    LAT = 52.4862
    LON = -1.8904

    all_frames = []

    print("=" * 60)
    print("PHASE 4 — SCRIPT 1: EXOGENOUS DATA ACQUISITION")
    print("=" * 60)
    print(f"\n[INFO] Querying Open-Meteo ERA5 archive for Birmingham ({LAT}°N, {LON}°W)")
    print(f"[INFO] Historical temporal range: {start_year}-01-01 to {end_year}-12-31")
    print("-" * 60)

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,precipitation"
            f"&timezone=Europe%2FLondon"
        )

        print(f"  - Ingesting {year}...", end=" ", flush=True)

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            precip = hourly.get("precipitation", [])

            if not times:
                print(f"[WARN] No records returned for {year}")
                continue

            year_df = pd.DataFrame({
                "datetime": pd.to_datetime(times),
                "temperature_2m": temps,
                "precipitation": precip
            })
            all_frames.append(year_df)
            print(f"OK ({len(year_df):,} hourly observations)")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            print(f"  Retrying {year} after brief delay...")
            time.sleep(5)
            try:
                resp = requests.get(url, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                hourly = data.get("hourly", {})
                year_df = pd.DataFrame({
                    "datetime": pd.to_datetime(hourly.get("time", [])),
                    "temperature_2m": hourly.get("temperature_2m", []),
                    "precipitation": hourly.get("precipitation", [])
                })
                all_frames.append(year_df)
                print(f"  - Retry successful ({len(year_df):,} hours)")
            except Exception as e2:
                print(f"  [FATAL] Unable to retrieve weather records for {year}: {e2}")
                sys.exit(1)

        # Respectful delay between successive API calls
        time.sleep(1)

    weather_df = pd.concat(all_frames, ignore_index=True)
    weather_df = weather_df.sort_values("datetime").reset_index(drop=True)

    # Impute rare isolated missing records via bidirectional fill
    null_count = weather_df.isnull().sum().sum()
    if null_count > 0:
        print(f"\n[INFO] Imputing {null_count} missing meteorological entries via forward/backward fill.")
        weather_df = weather_df.ffill().bfill()

    return weather_df


def generate_uk_holidays(start_year=2000, end_year=2025):
    """
    Generate official UK (England & Wales) statutory bank holidays for the project timeline.
    Mirrors official GOV.UK calendar schedules.
    """
    print(f"\n[INFO] Compiling UK statutory bank holidays ({start_year}-{end_year})...")

    uk_hols = holidays.UK(years=range(start_year, end_year + 1))

    records = []
    for date, name in sorted(uk_hols.items()):
        records.append({"date": pd.Timestamp(date), "holiday_name": name})

    hol_df = pd.DataFrame(records)
    print(f"  - Compiled {len(hol_df)} bank holidays across the study horizon.")

    return hol_df


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Fetch or load weather data
    weather_path = os.path.join(output_dir, "birmingham_weather.csv")
    if os.path.exists(weather_path):
        print(f"[INFO] Existing weather dataset detected at: {weather_path}")
        weather_df = pd.read_csv(weather_path)
    else:
        weather_df = fetch_weather_data(start_year=2000, end_year=2025)
        weather_df.to_csv(weather_path, index=False)

    print(f"\n{'=' * 60}")
    print("METEOROLOGICAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  - Output file: {weather_path}")
    print(f"  - Total records: {weather_df.shape[0]:,} rows x {weather_df.shape[1]} columns")
    print(f"  - Timeline: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
    print(f"  - Temperature: Mean={weather_df['temperature_2m'].mean():.1f}°C, Range=[{weather_df['temperature_2m'].min():.1f}°C, {weather_df['temperature_2m'].max():.1f}°C]")
    print(f"  - Precipitation: Max={weather_df['precipitation'].max():.1f} mm/hr")

    # 2. Compile UK bank holidays
    hol_path = os.path.join(output_dir, "uk_holidays.csv")
    if os.path.exists(hol_path):
        print(f"[INFO] Existing holiday schedule detected at: {hol_path}")
        hol_df = pd.read_csv(hol_path)
    else:
        hol_df = generate_uk_holidays(start_year=2000, end_year=2025)
        hol_df.to_csv(hol_path, index=False)

    print(f"\n{'=' * 60}")
    print("BANK HOLIDAY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  - Output file: {hol_path}")
    print(f"  - Total holidays: {len(hol_df)} dates cataloged")

    print(f"\n{'=' * 60}")
    print("EXOGENOUS DATA ACQUISITION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  [SAVED] birmingham_weather.csv ({weather_df.shape[0]:,} hourly records)")
    print(f"  [SAVED] uk_holidays.csv ({len(hol_df)} holidays)")
    print("=" * 60)


if __name__ == "__main__":
    main()

