# ==============================================================================
# Exploratory Deep Learning: LSTM 3D Sequence Tensor Preparation
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Converts sequential hourly traffic observations into 3D sliding window tensors
# [samples, time_steps = 24, features = 1] for exploratory PyTorch LSTM training.
# Note: As documented in Section 4.6, sequential models required synthetic night
# interpolation and were ultimately excluded from the final daytime benchmark.

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    # 1. Load Aggregated Traffic Data
    input_path = "../phase_1/master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "../phase_3/master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("=" * 60)
    print("PHASE 3.1 — SCRIPT 6: LSTM DATA PREPARATION")
    print("=" * 60)
    print(f"[INFO] Loading data from: {input_path}")

    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Aggregate Birmingham-wide hourly traffic
    traffic_df = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    traffic_df = traffic_df.sort_values('datetime').reset_index(drop=True)
    series = traffic_df['all_motor_vehicles'].values.astype(float)

    print(f"  -> Total hourly observations: {len(series):,}")

    # 2. Normalize data to 0-1 range (LSTM works best with scaled data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Save scaler min/max for later inverse transform
    scale_min = scaler.data_min_[0]
    scale_max = scaler.data_max_[0]

    # 3. Create Sliding Window Sequences
    WINDOW_SIZE = 24  # Use past 24 hours to predict next hour

    X_sequences = []
    y_targets = []

    for i in range(WINDOW_SIZE, len(scaled_series)):
        X_sequences.append(scaled_series[i - WINDOW_SIZE:i])  # Past 24 values
        y_targets.append(scaled_series[i])                      # Next value

    X = np.array(X_sequences)
    y = np.array(y_targets)

    # Reshape X to 3D: (samples, timesteps, features) — required by LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 4. Chronological Train/Test Split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  -> Window size: {WINDOW_SIZE} hours (past 24hrs -> predict next hour)")
    print(f"  -> Total sequences created: {len(X):,}")
    print(f"  -> X shape: {X.shape} (samples, timesteps, features)")
    print(f"  -> Training sequences: {len(X_train):,}")
    print(f"  -> Testing sequences:  {len(X_test):,}")

    # 5. Save to .npz file
    output_file = "lstm_sequences.npz"
    np.savez(output_file,
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test,
             scale_min=scale_min, scale_max=scale_max)

    print(f"\n[SUCCESS] LSTM sequences saved to: {output_file}")
    print("  -> Ready for 07_lstm_model_training.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

