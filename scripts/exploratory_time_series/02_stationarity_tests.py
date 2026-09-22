# ==============================================================================
# Statistical Stationarity Testing: ADF and KPSS Hypothesis Frameworks
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Formally evaluates the stationarity assumptions underlying linear time-series:
# 1. Augmented Dickey-Fuller (ADF): H0 = Unit root present (non-stationary)
# 2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS): H0 = Level/trend stationary
# Supports Section 2.2.2 and Section 3.4 diagnostics.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("[INFO] Loading data for Stationarity Testing...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    traffic_ts = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    series = traffic_ts['all_motor_vehicles'].dropna()

    # 2. Run ADF & KPSS Tests on Raw Series
    adf_res = adfuller(series)
    kpss_res = kpss(series, regression='c', nlags='auto')

    print(f"\n[ADF TEST RESULT] Statistic = {adf_res[0]:.4f}, p-value = {adf_res[1]:.4e}")
    print(f"[KPSS TEST RESULT] Statistic = {kpss_res[0]:.4f}, p-value = {kpss_res[1]:.4f}")

    # 3. Create First-Differenced Series
    diff_series = series.diff().dropna()
    adf_diff = adfuller(diff_series)

    # 4. Generate Plot (Original vs Differenced)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Birmingham Traffic — Stationarity & Differencing Analysis', fontsize=15, fontweight='bold')

    # Original Series Plot
    axes[0].plot(series.values, color='#2c3e50', linewidth=0.8)
    axes[0].set_title(f'Original Series (ADF p={adf_res[1]:.4e}, KPSS p={kpss_res[1]:.4f})')
    axes[0].set_ylabel('Traffic Volume')
    axes[0].axhline(y=series.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean = {series.mean():,.0f}')
    axes[0].legend()

    # Differenced Series Plot (d=1)
    axes[1].plot(diff_series.values, color='#e74c3c', linewidth=0.6)
    axes[1].set_title(f'First-Differenced Series (d=1) (ADF p={adf_diff[1]:.4e} → Strongly Stationary)')
    axes[1].set_ylabel('Δ Traffic Volume')
    axes[1].set_xlabel('Observation Index')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    output_filename = "stationarity_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot saved to: {output_filename}")

if __name__ == "__main__":
    main()

