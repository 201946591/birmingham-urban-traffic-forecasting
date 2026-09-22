# ==============================================================================
# Autocorrelation (ACF) & Partial Autocorrelation (PACF) Diagnostics
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Generates ACF and PACF correlograms across lags 1 to 48.
# Informs the autoregressive lag structure (t-1, t-2, t-3) utilized in feature
# engineering and identifies residual serial dependence.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("[INFO] Loading data for ACF/PACF Analysis...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    traffic_ts = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    series = traffic_ts['all_motor_vehicles'].dropna()

    # 2. Generate ACF & PACF Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Birmingham Traffic — Autocorrelation Diagnostics (ACF & PACF)', fontsize=15, fontweight='bold')

    # Plot ACF (identifies MA order q)
    plot_acf(series, lags=40, ax=axes[0], color='#3498db', vlines_kwargs={'colors': '#3498db'})
    axes[0].set_title('Autocorrelation Function (ACF) → Identifies MA(q) Order')
    axes[0].set_xlabel('Lag (Hours)')
    axes[0].set_ylabel('Autocorrelation')

    # Plot PACF (identifies AR order p)
    plot_pacf(series, lags=40, ax=axes[1], color='#e74c3c', vlines_kwargs={'colors': '#e74c3c'}, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF) → Identifies AR(p) Order')
    axes[1].set_xlabel('Lag (Hours)')
    axes[1].set_ylabel('Partial Autocorrelation')

    plt.tight_layout()
    output_filename = "acf_pacf_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot saved to: {output_filename}")

if __name__ == "__main__":
    main()

