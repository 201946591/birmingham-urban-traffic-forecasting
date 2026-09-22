# ==============================================================================
# STL Time-Series Decomposition (Trend, Seasonality, and Residuals)
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Decomposes aggregate hourly traffic volume into its constituent components:
# 1. Long-term trend
# 2. Recurring diurnal seasonality
# 3. Residual stochastic noise
# Informs the exploratory analysis discussed in Chapter 2 and Chapter 3.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("[INFO] Loading data for STL Decomposition...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Aggregate total traffic volume by datetime (hourly Birmingham traffic)
    traffic_ts = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    traffic_ts = traffic_ts.set_index('datetime').sort_index()

    series = traffic_ts['all_motor_vehicles']

    # 2. Perform STL Decomposition
    # period=24 means 24 hours per day (daily seasonality cycle)
    print("[INFO] Running STL Decomposition (period=24)...")
    stl = STL(series, period=24, robust=True)
    result = stl.fit()

    # 3. Create 4-Panel Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Birmingham Traffic Flow — STL Decomposition', fontsize=15, fontweight='bold')

    axes[0].plot(series.index, series.values, color='#2c3e50', linewidth=0.8)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('1. Original Observed Traffic Volume')

    axes[1].plot(result.trend.index, result.trend.values, color='#e74c3c', linewidth=1.2)
    axes[1].set_ylabel('Trend')
    axes[1].set_title('2. Long-term Trend Component')

    axes[2].plot(result.seasonal.index, result.seasonal.values, color='#3498db', linewidth=0.8)
    axes[2].set_ylabel('Seasonal')
    axes[2].set_title('3. Daily Seasonal Pattern (Period = 24 Hours)')

    axes[3].plot(result.resid.index, result.resid.values, color='#27ae60', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].set_title('4. Irregular Residual / Noise Component')
    axes[3].set_xlabel('Date')

    plt.tight_layout()
    output_filename = "decomposition_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot saved to: {output_filename}")

if __name__ == "__main__":
    main()

