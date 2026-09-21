# ==============================================================================
# Residual Diagnostic Analysis & Ljung-Box Portmanteau Test
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Evaluates whether time-series residuals satisfy white noise conditions:
# 1. Standardized residual time-series plot
# 2. Histogram with Gaussian KDE overlay
# 3. Normal Q-Q probability plot
# 4. Residual Autocorrelation Function (ACF)
# 5. Ljung-Box portmanteau test for residual independence (H0: White noise)

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("[INFO] Loading data & fitting ARIMA for Residual Diagnostics...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    traffic_ts = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    series = traffic_ts['all_motor_vehicles'].dropna()

    train_size = int(len(series) * 0.8)
    train_data = series.iloc[:train_size]

    # Fit ARIMA model
    model = pm.auto_arima(train_data, d=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True)
    residuals = model.resid()

    # 2. Run Ljung-Box Test
    lb_df = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\n[LJUNG-BOX TEST RESULTS]")
    print(lb_df)

    # 3. Create 4-Panel Diagnostic Plot
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('ARIMA Residual Diagnostics — Model Validation Suite', fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Residual Time Series
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(residuals, color='#2c3e50', linewidth=0.6, alpha=0.8)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('1. Residuals Over Time (Should hover around 0)')
    ax1.set_xlabel('Observation Index')
    ax1.set_ylabel('Residual Error')

    # Panel 2: Residual Histogram + Normal KDE Overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=50, density=True, alpha=0.7, color='#3498db', edgecolor='white', label='Residuals')
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_norm, stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals)), 'r-', linewidth=2, label='Normal Curve')
    ax2.set_title('2. Residual Distribution vs Normal Curve')
    ax2.set_xlabel('Residual Error')
    ax2.set_ylabel('Density')
    ax2.legend()

    # Panel 3: Normal Q-Q Plot
    ax3 = fig.add_subplot(gs[1, 0])
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('3. Q-Q Plot (Points should fall on red line if normal)')
    ax3.get_lines()[0].set_color('#3498db')
    ax3.get_lines()[0].set_markersize(3)
    ax3.get_lines()[1].set_color('#e74c3c')

    # Panel 4: Residual ACF Plot
    ax4 = fig.add_subplot(gs[1, 1])
    plot_acf(residuals, lags=30, ax=ax4, color='#27ae60', vlines_kwargs={'colors': '#27ae60'})
    ax4.set_title('4. Residual ACF (Spikes indicate remaining autocorrelation)')
    ax4.set_xlabel('Lag (Hours)')

    output_filename = "residual_diagnostics.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Plot saved to: {output_filename}")

if __name__ == "__main__":
    main()
