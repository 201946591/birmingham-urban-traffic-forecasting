# ==============================================================================
# Exploratory Deep Learning: LSTM Residual Diagnostic Analysis
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Evaluates residual diagnostic metrics (normality, skewness, kurtosis) for the
# exploratory PyTorch LSTM network evaluated on the sequential holdout dataset.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn


# Define LSTM Model Architecture (must match training script)
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        return out


def main():
    # 1. Load Prepared Sequences & Saved Model
    data_path = "lstm_sequences.npz"
    model_path = "lstm_model.pth"

    if not os.path.exists(data_path):
        print("[ERROR] Cannot find lstm_sequences.npz!")
        print("  -> Please run 06_lstm_data_preparation.py first.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print("[ERROR] Cannot find lstm_model.pth!")
        print("  -> Please run 07_lstm_model_training.py first.")
        sys.exit(1)

    print("=" * 60)
    print("PHASE 3.2 — SCRIPT 11: LSTM RESIDUAL ANALYSIS")
    print("=" * 60)

    # Load data
    data = np.load(data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    scale_min = float(data['scale_min'])
    scale_max = float(data['scale_max'])

    print(f"[INFO] Test sequences loaded: {X_test.shape[0]:,}")

    # Load trained model
    device = torch.device('cpu')
    model = TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("[INFO] Trained LSTM model loaded successfully.")

    # 2. Generate Predictions
    X_test_t = torch.FloatTensor(X_test)
    with torch.no_grad():
        predictions_scaled = model(X_test_t.to(device)).cpu().numpy().flatten()

    # Inverse transform to real traffic counts
    y_test_real = y_test * (scale_max - scale_min) + scale_min
    predictions_real = predictions_scaled * (scale_max - scale_min) + scale_min
    residuals = y_test_real - predictions_real

    rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
    mae = mean_absolute_error(y_test_real, predictions_real)
    r2 = r2_score(y_test_real, predictions_real)

    print(f"\n  LSTM RMSE: {rmse:,.4f}")
    print(f"  LSTM MAE:  {mae:,.4f}")
    print(f"  LSTM R²:   {r2:.4f}")

    # 3. Generate 4-Panel Residual Diagnostic Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LSTM Residual Diagnostics — Dissertation Chapter 4',
                 fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: Residuals Over Time
    ax1 = axes[0, 0]
    ax1.scatter(range(len(residuals)), residuals, alpha=0.4, s=8, color='#e67e22', edgecolors='none')
    ax1.axhline(y=0, color='#e74c3c', linewidth=2, linestyle='--', label='Zero Error Line')
    ax1.set_title('(a) Residuals Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Holdout Observation Index', fontsize=12)
    ax1.set_ylabel('Residual (Actual − Predicted)', fontsize=12)
    ax1.legend(fontsize=11)

    # Panel 2: Residual Histogram + KDE
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=60, density=True, alpha=0.7, color='#f39c12', edgecolor='white', label='Residual Distribution')
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax2.plot(x_range, stats.norm.pdf(x_range, mu, sigma), color='#e74c3c', linewidth=2.5, label=f'Normal Fit (μ={mu:.1f}, σ={sigma:.1f})')
    ax2.set_title('(b) Residual Distribution (Histogram + Normal KDE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Residual Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=11)

    # Panel 3: Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('(c) Q-Q Plot (Normal Probability Plot)', fontsize=14, fontweight='bold')
    ax3.get_lines()[0].set_markerfacecolor('#e67e22')
    ax3.get_lines()[0].set_markersize(3)
    ax3.get_lines()[1].set_color('#e74c3c')
    ax3.get_lines()[1].set_linewidth(2)

    # Panel 4: Residual Autocorrelation
    ax4 = axes[1, 1]
    max_lags = 50
    autocorrelations = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] for lag in range(1, max_lags + 1)]
    ax4.bar(range(1, max_lags + 1), autocorrelations, color='#8e44ad', alpha=0.8, width=0.8)
    significance_bound = 1.96 / np.sqrt(len(residuals))
    ax4.axhline(y=significance_bound, color='#e74c3c', linewidth=1.5, linestyle='--', label=f'95% CI (±{significance_bound:.3f})')
    ax4.axhline(y=-significance_bound, color='#e74c3c', linewidth=1.5, linestyle='--')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('(d) Residual Autocorrelation (ACF)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Lag (Hours)', fontsize=12)
    ax4.set_ylabel('Autocorrelation', fontsize=12)
    ax4.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = "lstm_residual_diagnostics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] 4-panel residual diagnostics saved to: {output_file}")

    # 4. Error Distribution Plot
    fig2, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('LSTM Prediction Error Analysis — Dissertation Chapter 4',
                  fontsize=16, fontweight='bold')

    ax_hist.hist(residuals, bins=60, color='#e67e22', edgecolor='white', alpha=0.8)
    ax_hist.axvline(x=0, color='#e74c3c', linewidth=2, linestyle='--', label='Zero Error')
    ax_hist.axvline(x=np.mean(residuals), color='#3498db', linewidth=2, linestyle='-', label=f'Mean Error ({np.mean(residuals):,.1f})')
    ax_hist.set_title('Error Histogram', fontsize=14, fontweight='bold')
    ax_hist.set_xlabel('Prediction Error (Actual − Predicted)', fontsize=12)
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.legend(fontsize=11)

    ax_scatter.scatter(range(len(residuals)), residuals, alpha=0.3, s=6, color='#8e44ad', edgecolors='none')
    ax_scatter.axhline(y=0, color='black', linewidth=1)
    window = 50
    rolling_err = pd.Series(residuals).rolling(window=window, center=True).mean()
    ax_scatter.plot(range(len(rolling_err)), rolling_err, color='#2ecc71', linewidth=2.5, label=f'Rolling Mean ({window}-obs window)')
    ax_scatter.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax_scatter.set_xlabel('Holdout Observation Index', fontsize=12)
    ax_scatter.set_ylabel('Prediction Error', fontsize=12)
    ax_scatter.legend(fontsize=11)

    plt.tight_layout()
    output_file2 = "lstm_error_distribution.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Error distribution plot saved to: {output_file2}")

    # 5. Save Report
    report_file = "lstm_residual_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM RESIDUAL ANALYSIS REPORT — DISSERTATION CHAPTER 4\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Mean Residual:         {np.mean(residuals):,.4f}\n")
        f.write(f"Std Dev of Residuals:  {np.std(residuals):,.4f}\n")
        f.write(f"Min Residual:          {np.min(residuals):,.4f}\n")
        f.write(f"Max Residual:          {np.max(residuals):,.4f}\n")
        f.write(f"Median Residual:       {np.median(residuals):,.4f}\n\n")
        f.write(f"Skewness:              {stats.skew(residuals):.4f}\n")
        f.write(f"Kurtosis:              {stats.kurtosis(residuals):.4f}\n\n")
        sample_size = min(5000, len(residuals))
        shapiro_stat, shapiro_p = stats.shapiro(np.random.choice(residuals, sample_size, replace=False))
        f.write(f"Shapiro-Wilk Normality Test (n={sample_size}):\n")
        f.write(f"  Statistic: {shapiro_stat:.6f}\n")
        f.write(f"  p-value:   {shapiro_p:.6e}\n")
        f.write(f"  Result:    {'PASS (Normally Distributed)' if shapiro_p > 0.05 else 'FAIL (Non-Normal — expected for traffic data)'}\n\n")
        f.write("--- GENERATED PLOTS ---\n")
        f.write("1. lstm_residual_diagnostics.png  — 4-Panel Diagnostic Plot\n")
        f.write("2. lstm_error_distribution.png    — Error Histogram & Error-Over-Time\n")

    print(f"[SUCCESS] Residual analysis report saved to: {report_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

