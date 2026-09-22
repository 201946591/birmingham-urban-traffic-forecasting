# ==============================================================================
# Prediction Error Distribution & Temporal Residual Heteroscedasticity
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Analyzes error spread (Actual - Predicted) to detect systematic model bias
# and heteroscedasticity across diurnal peak and off-peak hours.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("[INFO] Loading data for Error Distribution Analysis...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    traffic_ts = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    series = traffic_ts['all_motor_vehicles'].dropna()

    train_size = int(len(series) * 0.8)
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]

    model = pm.auto_arima(train_data, d=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True)
    predictions = model.predict(n_periods=len(test_data))

    errors = test_data.values - predictions

    # 2. Generate Error Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ARIMA Prediction Error & Bias Analysis', fontsize=15, fontweight='bold')

    # Histogram of errors
    axes[0].hist(errors, bins=50, density=True, alpha=0.7, color='#9b59b6', edgecolor='white')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label=f'Zero Bias (Mean = {np.mean(errors):,.2f})')
    axes[0].set_title('1. Error Distribution Histogram (Actual - Predicted)')
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Scatter of errors over time
    axes[1].scatter(range(len(errors)), errors, alpha=0.5, s=10, color='#9b59b6')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('2. Prediction Errors Across Test Set Timeline')
    axes[1].set_xlabel('Test Observation Index')
    axes[1].set_ylabel('Prediction Error (Vehicles)')

    plt.tight_layout()
    output_filename = "error_distribution.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot saved to: {output_filename}")

if __name__ == "__main__":
    main()

