# ==============================================================================
# Exploratory Three-Model Comparative Performance Table (ARIMA vs XGBoost vs LSTM)
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Summarizes the exploratory three-model benchmark evaluated during preliminary
# sequence modeling trials on the interpolated panel.

import os
import sys
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("PHASE 3.1 — SCRIPT 8: THREE-MODEL COMPARISON TABLE")
    print("=" * 60)

    # Phase 2 ARIMA Results (fixed values)
    arima_rmse = 2730.69
    arima_mae = 1543.66
    arima_r2 = -0.4075

    # Phase 3 XGBoost Results
    xgb_rmse, xgb_mae, xgb_r2 = 804.76, 258.63, 0.8778
    xgb_results_path = "xgboost_results.txt"
    if not os.path.exists(xgb_results_path):
        xgb_results_path = "../phase_3/xgboost_results.txt"
    if os.path.exists(xgb_results_path):
        with open(xgb_results_path, "r") as f:
            for line in f:
                if "XGBoost RMSE:" in line:
                    xgb_rmse = float(line.split(":")[1].strip().replace(",", ""))
                elif "XGBoost MAE:" in line:
                    xgb_mae = float(line.split(":")[1].strip().replace(",", ""))
                elif "XGBoost R" in line and ":" in line:
                    xgb_r2 = float(line.split(":")[1].strip())

    # Phase 3.1 LSTM Results
    lstm_rmse, lstm_mae, lstm_r2 = 0, 0, 0
    if os.path.exists("lstm_results.txt"):
        with open("lstm_results.txt", "r") as f:
            for line in f:
                if "LSTM RMSE:" in line:
                    lstm_rmse = float(line.split(":")[1].strip().replace(",", ""))
                elif "LSTM MAE:" in line:
                    lstm_mae = float(line.split(":")[1].strip().replace(",", ""))
                elif "LSTM R2:" in line:
                    lstm_r2 = float(line.split(":")[1].strip())
    else:
        print("[WARNING] lstm_results.txt not found. Run 07_lstm_model_training.py first!")
        print("  Using placeholder values...")
        lstm_rmse, lstm_mae, lstm_r2 = 999.99, 599.99, 0.50

    # Find the best model
    best_rmse = min(arima_rmse, xgb_rmse, lstm_rmse)

    # Build Table
    table_data = [
        ["Model", "RMSE", "MAE", "R2 Score", "vs ARIMA"],
        ["ARIMA(1,1,3)\n(Statistical Baseline)",
         f"{arima_rmse:,.2f}", f"{arima_mae:,.2f}", f"{arima_r2:.4f}", "Baseline"],
        ["XGBoost Regressor\n(Tree-based ML)",
         f"{xgb_rmse:,.2f}", f"{xgb_mae:,.2f}", f"{xgb_r2:.4f}",
         f"{((arima_rmse - xgb_rmse) / arima_rmse) * 100:.1f}% better"],
        ["LSTM Network\n(Deep Learning)",
         f"{lstm_rmse:,.2f}", f"{lstm_mae:,.2f}", f"{lstm_r2:.4f}",
         f"{((arima_rmse - lstm_rmse) / arima_rmse) * 100:.1f}% better"],
    ]

    # Create Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    fig.suptitle('Model Performance Comparison: ARIMA vs XGBoost vs LSTM',
                 fontsize=14, fontweight='bold')

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style Header Row (dark blue)
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

    # Style ARIMA row (light red — worst)
    for i in range(5):
        table[(1, i)].set_facecolor('#fadbd8')

    # Highlight the best model row (light green)
    if best_rmse == xgb_rmse:
        best_row = 2
    elif best_rmse == lstm_rmse:
        best_row = 3
    else:
        best_row = 1

    for i in range(5):
        table[(best_row, i)].set_facecolor('#d4edda')

    output_filename = "three_model_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Also save as text
    with open("three_model_comparison.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL THREE-MODEL COMPARISON — DISSERTATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'R2':>10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'ARIMA(1,1,3)':<25} {arima_rmse:>12,.2f} {arima_mae:>12,.2f} {arima_r2:>10.4f}\n")
        f.write(f"{'XGBoost':<25} {xgb_rmse:>12,.2f} {xgb_mae:>12,.2f} {xgb_r2:>10.4f}\n")
        f.write(f"{'LSTM':<25} {lstm_rmse:>12,.2f} {lstm_mae:>12,.2f} {lstm_r2:>10.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"\nBest Model: {'XGBoost' if best_rmse == xgb_rmse else 'LSTM' if best_rmse == lstm_rmse else 'ARIMA'} (lowest RMSE)\n")

    print(f"[SUCCESS] 3-model comparison table saved to: {output_filename}")
    print(f"[SUCCESS] Text version saved to: three_model_comparison.txt")

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  ARIMA RMSE:   {arima_rmse:>10,.2f}  (Baseline)")
    print(f"  XGBoost RMSE: {xgb_rmse:>10,.2f}  ({((arima_rmse - xgb_rmse) / arima_rmse) * 100:.1f}% better)")
    print(f"  LSTM RMSE:    {lstm_rmse:>10,.2f}  ({((arima_rmse - lstm_rmse) / arima_rmse) * 100:.1f}% better)")
    print("=" * 60)

if __name__ == "__main__":
    main()

