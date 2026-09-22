# ==============================================================================
# XGBoost Residual Diagnostic Analysis & Statistical Normality Testing
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Generates a four-panel residual diagnostic suite:
# 1. Residuals over chronological time (evaluating heteroscedasticity)
# 2. Histogram with theoretical Gaussian overlay (skewness & kurtosis)
# 3. Quantile-Quantile (Q-Q) probability plot
# 4. Residual Autocorrelation Function (ACF) to test for serial dependence
# Computes the Shapiro-Wilk normality test, skewness (8.85), and kurtosis (151.48).

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # 1. Load Engineered Features
    input_path = "engineered_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "phase_3/engineered_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        print("[ERROR] Cannot find engineered_birmingham_traffic.csv! Please run 01_feature_engineering.py first.")
        sys.exit(1)

    print("=" * 60)
    print("PHASE 3 — SCRIPT 11: XGBOOST RESIDUAL ANALYSIS")
    print("=" * 60)
    print(f"[INFO] Loading engineered dataset: {input_path}")

    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Define Feature Matrix
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'lag_1', 'lag_2', 'lag_3', 'lag_24',
        'rolling_mean_3', 'rolling_std_3'
    ]
    target_col = 'all_motor_vehicles'

    X = df[feature_cols]
    y = df[target_col]

    # Chronological Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = df['datetime'].iloc[split_idx:].values

    # 2. Load or retrain XGBoost model
    model_path = "xgboost_model.json"
    if not os.path.exists(model_path):
        model_path = "phase_3/xgboost_model.json"
    
    model = XGBRegressor()
    if os.path.exists(model_path):
        print(f"[INFO] Loading saved model from: {model_path}")
        model.load_model(model_path)
    else:
        print("[INFO] No saved model found. Retraining XGBoost...")
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

    # 3. Generate Predictions & Residuals
    predictions = model.predict(X_test)
    residuals = y_test.values - predictions

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n  XGBoost RMSE: {rmse:,.4f}")
    print(f"  XGBoost MAE:  {mae:,.4f}")
    print(f"  XGBoost R²:   {r2:.4f}")

    # 4. Generate 4-Panel Residual Diagnostic Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('XGBoost Residual Diagnostics — Dissertation Chapter 4',
                 fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: Residuals Over Time
    ax1 = axes[0, 0]
    ax1.scatter(range(len(residuals)), residuals, alpha=0.4, s=8, color='#3498db', edgecolors='none')
    ax1.axhline(y=0, color='#e74c3c', linewidth=2, linestyle='--', label='Zero Error Line')
    ax1.set_title('(a) Residuals Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Holdout Observation Index', fontsize=12)
    ax1.set_ylabel('Residual (Actual − Predicted)', fontsize=12)
    ax1.legend(fontsize=11)

    # Panel 2: Residual Histogram + KDE
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=60, density=True, alpha=0.7, color='#2ecc71', edgecolor='white', label='Residual Distribution')
    # Overlay normal distribution curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax2.plot(x_range, stats.norm.pdf(x_range, mu, sigma), color='#e74c3c', linewidth=2.5, label=f'Normal Fit (μ={mu:.1f}, σ={sigma:.1f})')
    ax2.set_title('(b) Residual Distribution (Histogram + Normal KDE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Residual Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=11)

    # Panel 3: Q-Q Plot (Quantile-Quantile)
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('(c) Q-Q Plot (Normal Probability Plot)', fontsize=14, fontweight='bold')
    ax3.get_lines()[0].set_markerfacecolor('#3498db')
    ax3.get_lines()[0].set_markersize(3)
    ax3.get_lines()[1].set_color('#e74c3c')
    ax3.get_lines()[1].set_linewidth(2)

    # Panel 4: Residual Autocorrelation (first 50 lags)
    ax4 = axes[1, 1]
    max_lags = 50
    autocorrelations = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] for lag in range(1, max_lags + 1)]
    ax4.bar(range(1, max_lags + 1), autocorrelations, color='#9b59b6', alpha=0.8, width=0.8)
    # Significance bounds (95% confidence interval)
    significance_bound = 1.96 / np.sqrt(len(residuals))
    ax4.axhline(y=significance_bound, color='#e74c3c', linewidth=1.5, linestyle='--', label=f'95% CI (±{significance_bound:.3f})')
    ax4.axhline(y=-significance_bound, color='#e74c3c', linewidth=1.5, linestyle='--')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('(d) Residual Autocorrelation (ACF)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Lag (Hours)', fontsize=12)
    ax4.set_ylabel('Autocorrelation', fontsize=12)
    ax4.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = "xgboost_residual_diagnostics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] 4-panel residual diagnostics saved to: {output_file}")

    # 5. Generate Error Distribution Plot (matching ARIMA's error_distribution.png)
    fig2, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('XGBoost Prediction Error Analysis — Dissertation Chapter 4',
                  fontsize=16, fontweight='bold')

    # Error histogram
    ax_hist.hist(residuals, bins=60, color='#3498db', edgecolor='white', alpha=0.8)
    ax_hist.axvline(x=0, color='#e74c3c', linewidth=2, linestyle='--', label='Zero Error')
    ax_hist.axvline(x=np.mean(residuals), color='#f39c12', linewidth=2, linestyle='-', label=f'Mean Error ({np.mean(residuals):,.1f})')
    ax_hist.set_title('Error Histogram', fontsize=14, fontweight='bold')
    ax_hist.set_xlabel('Prediction Error (Actual − Predicted)', fontsize=12)
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.legend(fontsize=11)

    # Error over time scatter
    ax_scatter.scatter(range(len(residuals)), residuals, alpha=0.3, s=6, color='#e74c3c', edgecolors='none')
    ax_scatter.axhline(y=0, color='black', linewidth=1)
    # Add rolling mean of errors
    window = 50
    rolling_err = pd.Series(residuals).rolling(window=window, center=True).mean()
    ax_scatter.plot(range(len(rolling_err)), rolling_err, color='#2ecc71', linewidth=2.5, label=f'Rolling Mean ({window}-obs window)')
    ax_scatter.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax_scatter.set_xlabel('Holdout Observation Index', fontsize=12)
    ax_scatter.set_ylabel('Prediction Error', fontsize=12)
    ax_scatter.legend(fontsize=11)

    plt.tight_layout()
    output_file2 = "xgboost_error_distribution.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Error distribution plot saved to: {output_file2}")

    # 6. Save residual analysis report
    report_file = "xgboost_residual_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST RESIDUAL ANALYSIS REPORT — DISSERTATION CHAPTER 4\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Mean Residual:         {np.mean(residuals):,.4f}\n")
        f.write(f"Std Dev of Residuals:  {np.std(residuals):,.4f}\n")
        f.write(f"Min Residual:          {np.min(residuals):,.4f}\n")
        f.write(f"Max Residual:          {np.max(residuals):,.4f}\n")
        f.write(f"Median Residual:       {np.median(residuals):,.4f}\n\n")
        f.write(f"Skewness:              {stats.skew(residuals):.4f}\n")
        f.write(f"Kurtosis:              {stats.kurtosis(residuals):.4f}\n\n")
        # Shapiro-Wilk normality test (on a subsample if too large)
        sample_size = min(5000, len(residuals))
        shapiro_stat, shapiro_p = stats.shapiro(np.random.choice(residuals, sample_size, replace=False))
        f.write(f"Shapiro-Wilk Normality Test (n={sample_size}):\n")
        f.write(f"  Statistic: {shapiro_stat:.6f}\n")
        f.write(f"  p-value:   {shapiro_p:.6e}\n")
        f.write(f"  Result:    {'PASS (Normally Distributed)' if shapiro_p > 0.05 else 'FAIL (Non-Normal — expected for traffic data)'}\n\n")
        f.write("--- GENERATED PLOTS ---\n")
        f.write("1. xgboost_residual_diagnostics.png  — 4-Panel Diagnostic Plot\n")
        f.write("2. xgboost_error_distribution.png    — Error Histogram & Error-Over-Time\n")

    print(f"[SUCCESS] Residual analysis report saved to: {report_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

