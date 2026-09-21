# ==============================================================================
# Exploratory Statistical Baseline: ARIMA Model Estimation & Holdout Evaluation
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Evaluates an exploratory Auto-ARIMA baseline with 95% confidence intervals.
# Note: As documented in Section 4.6, ARIMA was excluded from the final benchmark
# due to structural incompatibility with discontinuous 12-hour DfT manual survey data.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"

    print("[INFO] Loading data & performing Temporal Train/Test split...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    traffic_ts = df.groupby('datetime')['all_motor_vehicles'].sum().reset_index()
    series = traffic_ts['all_motor_vehicles'].dropna()

    # 2. Chronological Split (80% Train, 20% Test)
    train_size = int(len(series) * 0.8)
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]

    print(f"  → Training set: {len(train_data):,} samples")
    print(f"  → Testing holdout: {len(test_data):,} samples")

    # 3. Fit ARIMA & Forecast
    print("[INFO] Fitting auto_arima model...")
    model = pm.auto_arima(train_data, d=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True)
    
    n_periods = len(test_data)
    predictions, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)

    # 4. Calculate Evaluation Metrics
    actual = test_data.values
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    print("\n" + "=" * 50)
    print("ARIMA BASELINE METRICS RESULT")
    print("=" * 50)
    print(f"  ARIMA Order: {model.order}")
    print(f"  RMSE: {rmse:,.4f}")
    print(f"  MAE:  {mae:,.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")

    # 5. Generate Forecast Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('ARIMA Forecast — Actual Traffic vs Baseline Predictions (20% Holdout)', fontsize=15, fontweight='bold')

    # Plot recent training data for context
    context = train_data.iloc[-200:]
    ax.plot(range(len(context)), context.values, color='#2c3e50', label='Training Data', alpha=0.7)

    # Plot test actual vs predicted
    test_x = range(len(context), len(context) + len(test_data))
    ax.plot(test_x, actual, color='#3498db', label='Actual Traffic (Test Set)')
    ax.plot(test_x, predictions, color='#e74c3c', linestyle='--', label='ARIMA Forecast')
    ax.fill_between(test_x, conf_int[:, 0], conf_int[:, 1], color='#e74c3c', alpha=0.15, label='95% Confidence Interval')

    ax.axvline(x=len(context), color='gray', linestyle=':', label='Train/Test Split Boundary')
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Total Traffic Volume (all_motor_vehicles)')
    ax.legend(loc='upper left')

    # Overlay metric text box
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.95, f"RMSE = {rmse:,.2f}\nMAE  = {mae:,.2f}\nMAPE = {mape:.2f}%\nR²   = {r2:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=bbox_props)

    output_filename = "arima_forecast_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Plot saved to: {output_filename}")

    # 6. Save Baseline Results Text File
    results_txt_path = "arima_baseline_results.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("PHASE 2: ARIMA STATISTICAL BENCHMARKING — BASELINE RESULTS\n")
        f.write("Birmingham Traffic Flow Prediction Using Machine Learning\n")
        f.write("======================================================================\n\n")
        f.write(f"ARIMA Order (p, d, q): {model.order}\n")
        f.write(f"RMSE:   {rmse:,.4f}\n")
        f.write(f"MAE:    {mae:,.4f}\n")
        f.write(f"MAPE:   {mape:.2f}%\n")
        f.write(f"R²:     {r2:.6f}\n\n")
        f.write("======================================================================\n")
        f.write(">>> DEFINITIVE BASELINE THRESHOLD:\n")
        f.write(">>> Phase 3 ML models (XGBoost) MUST achieve lower RMSE & MAE to succeed.\n")
        f.write("======================================================================\n")

    print(f"[SUCCESS] Baseline metrics report saved to: {results_txt_path}")

if __name__ == "__main__":
    main()
