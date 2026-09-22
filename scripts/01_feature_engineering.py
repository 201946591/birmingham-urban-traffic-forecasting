# ==============================================================================
# Feature Engineering: Temporal Lags, Cyclical Encodings & Rolling Statistics
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd

def main():
    # 1. Load Preprocessed Data from Phase 1
    input_path = "../phase_1/master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("=" * 60)
    print("PHASE 3 — SCRIPT 1: FEATURE ENGINEERING")
    print("=" * 60)
    print(f"[INFO] Reading Phase 1 master data from: {input_path}")

    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 2. Aggregate Birmingham-wide Hourly Traffic
    traffic_df = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    traffic_df = traffic_df.sort_values('datetime').reset_index(drop=True)

    # Extract Datetime Features
    traffic_df['hour'] = traffic_df['datetime'].dt.hour
    traffic_df['day_of_week'] = traffic_df['datetime'].dt.dayofweek
    traffic_df['month'] = traffic_df['datetime'].dt.month
    traffic_df['is_weekend'] = traffic_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Cyclical Encodings (sine & cosine for smooth continuous cycles)
    traffic_df['hour_sin'] = np.sin(2 * np.pi * traffic_df['hour'] / 24.0)
    traffic_df['hour_cos'] = np.cos(2 * np.pi * traffic_df['hour'] / 24.0)
    traffic_df['dow_sin'] = np.sin(2 * np.pi * traffic_df['day_of_week'] / 7.0)
    traffic_df['dow_cos'] = np.cos(2 * np.pi * traffic_df['day_of_week'] / 7.0)

    # 3. Create Lag Features (Past Traffic History)
    print("[INFO] Engineering Lag Features (t-1, t-2, t-3, t-24)...")
    traffic_df['lag_1'] = traffic_df['all_motor_vehicles'].shift(1)   # 1 hour ago
    traffic_df['lag_2'] = traffic_df['all_motor_vehicles'].shift(2)   # 2 hours ago
    traffic_df['lag_3'] = traffic_df['all_motor_vehicles'].shift(3)   # 3 hours ago
    traffic_df['lag_24'] = traffic_df['all_motor_vehicles'].shift(24) # 24 hours ago (Yesterday)

    # Rolling window features
    traffic_df['rolling_mean_3'] = traffic_df['all_motor_vehicles'].shift(1).rolling(window=3).mean()
    traffic_df['rolling_std_3'] = traffic_df['all_motor_vehicles'].shift(1).rolling(window=3).std()

    # Drop NA rows created by shifting/lags
    clean_df = traffic_df.dropna().reset_index(drop=True)

    # 4. Save Output
    output_filename = "engineered_birmingham_traffic.csv"
    clean_df.to_csv(output_filename, index=False)

    print(f"\n[SUCCESS] Feature Engineering Complete!")
    print(f"  → Cleaned shape: {clean_df.shape[0]:,} rows × {clean_df.shape[1]} columns")
    print(f"  → Output saved to: {output_filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()

