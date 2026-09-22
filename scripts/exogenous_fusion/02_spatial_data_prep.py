# ==============================================================================
# Spatial Zonal Data Preparation: Major vs. Minor Road-Type Stratification
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Stratifies the Birmingham traffic dataset across two functional road classes:
# 1. Major Roads: Arterial corridors (A-roads, motorways) carrying heavy continuous baseline flow
# 2. Minor Roads: Residential and unclassified distributor streets (B, C, and U roads)
# Lags and rolling statistics are engineered strictly within each road type partition
# to prevent cross-zone signal contamination.

import os
import sys
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("PHASE 4 — SCRIPT 2: SPATIAL DATA PREPARATION (ZONING)")
    print("=" * 60)

    # 1. Locate and read the master preprocessed traffic dataset
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "master_birmingham_traffic.csv"),
        os.path.join(os.path.dirname(__file__), "..", "dft_rawcount_local_authority_id_141.csv"),
    ]

    input_path = None
    for p in possible_paths:
        if os.path.exists(p):
            input_path = os.path.abspath(p)
            break

    if input_path is None:
        print("[ERROR] Could not locate master_birmingham_traffic.csv or raw DfT survey file.")
        sys.exit(1)

    print(f"[INFO] Ingesting traffic dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  - Initial shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Parse timestamps
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "count_date" in df.columns and "hour" in df.columns:
        df["datetime"] = pd.to_datetime(df["count_date"]) + pd.to_timedelta(df["hour"], unit="h")
    else:
        print("[ERROR] Neither 'datetime' nor 'count_date' + 'hour' found in dataset.")
        sys.exit(1)

    # 3. Inspect functional road classification distribution
    road_types = df["road_type"].unique()
    print(f"\n[INFO] Identified road types: {road_types}")
    for rt in sorted(road_types):
        count = len(df[df["road_type"] == rt])
        print(f"  - {rt} roads: {count:,} records")

    # 4. Aggregate hourly traffic flow by (datetime, road_type)
    # Aggregating across individual sensors per road category builds separate continuous
    # time-series for Major arterial corridors and Minor residential distributors.
    print(f"\n[INFO] Aggregating hourly traffic volume by road type...")
    zonal_df = (
        df.groupby(["datetime", "road_type"])["all_motor_vehicles"]
        .sum()
        .reset_index()
    )
    zonal_df = zonal_df.sort_values("datetime").reset_index(drop=True)
    print(f"  - Aggregated zonal series shape: {zonal_df.shape[0]:,} rows x {zonal_df.shape[1]} columns")

    # 5. Extract temporal features and cyclical trigonometric projections
    zonal_df["hour"] = zonal_df["datetime"].dt.hour
    zonal_df["day_of_week"] = zonal_df["datetime"].dt.dayofweek
    zonal_df["month"] = zonal_df["datetime"].dt.month
    zonal_df["is_weekend"] = (zonal_df["day_of_week"] >= 5).astype(int)
    zonal_df["year"] = zonal_df["datetime"].dt.year

    zonal_df["hour_sin"] = np.sin(2 * np.pi * zonal_df["hour"] / 24.0)
    zonal_df["hour_cos"] = np.cos(2 * np.pi * zonal_df["hour"] / 24.0)
    zonal_df["dow_sin"] = np.sin(2 * np.pi * zonal_df["day_of_week"] / 7.0)
    zonal_df["dow_cos"] = np.cos(2 * np.pi * zonal_df["day_of_week"] / 7.0)

    # 6. Engineer autoregressive lag features per zone
    # Note: Lags must be computed independently within each road_type group.
    # Otherwise, Major road observations would bleed into the history of Minor roads.
    print(f"\n[INFO] Engineering independent autoregressive lags per road type...")
    lag_cols = []
    for lag in [1, 2, 3, 24]:
        col_name = f"lag_{lag}"
        zonal_df[col_name] = (
            zonal_df.groupby("road_type")["all_motor_vehicles"]
            .shift(lag)
        )
        lag_cols.append(col_name)

    # Rolling window statistics per road category
    zonal_df["rolling_mean_3"] = (
        zonal_df.groupby("road_type")["all_motor_vehicles"]
        .transform(lambda x: x.shift(1).rolling(window=3).mean())
    )
    zonal_df["rolling_std_3"] = (
        zonal_df.groupby("road_type")["all_motor_vehicles"]
        .transform(lambda x: x.shift(1).rolling(window=3).std())
    )

    # Drop boundary rows where lags are undefined
    initial_len = len(zonal_df)
    zonal_df = zonal_df.dropna().reset_index(drop=True)
    print(f"  - Removed {initial_len - len(zonal_df)} initial boundary rows with undefined lags.")

    # 7. Encode road_type as binary indicator for machine learning algorithms
    zonal_df["road_type_encoded"] = zonal_df["road_type"].map({"Major": 1, "Minor": 0})

    # 8. Summary statistics by functional classification
    print(f"\n{'=' * 60}")
    print("ROAD CLASSIFICATION PROFILE")
    print(f"{'=' * 60}")
    for rt in sorted(zonal_df["road_type"].unique()):
        sub = zonal_df[zonal_df["road_type"] == rt]
        print(f"  [{rt} Roads]")
        print(f"    - Observations: {len(sub):,}")
        print(f"    - Mean hourly volume: {sub['all_motor_vehicles'].mean():,.1f} veh/hr")
        print(f"    - Max hourly peak:    {sub['all_motor_vehicles'].max():,.0f} veh/hr")

    # 9. Save output dataset
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "zonal_traffic.csv")
    zonal_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("SPATIAL ZONAL DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  [SAVED] zonal_traffic.csv ({zonal_df.shape[0]:,} rows x {zonal_df.shape[1]} columns)")
    print("=" * 60)


if __name__ == "__main__":
    main()
