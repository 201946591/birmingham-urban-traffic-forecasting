# ==============================================================================
# Feature Importance Extraction & Ranking (Gain vs F-score)
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def main():
    # 1. Load Data
    input_path = "engineered_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "phase_3/engineered_birmingham_traffic.csv"

    print("=" * 60)
    print("PHASE 3 — SCRIPT 4: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(input_path)
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'lag_1', 'lag_2', 'lag_3', 'lag_24',
        'rolling_mean_3', 'rolling_std_3'
    ]
    X = df[feature_cols]
    y = df['all_motor_vehicles']

    # 2. Fit Model
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42)
    model.fit(X, y)

    # 3. Extract Feature Importances
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

    # 4. Generate Horizontal Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('XGBoost Model — Top Feature Importances', fontsize=14, fontweight='bold')

    importances.plot(kind='barh', color='#3498db', ax=ax, edgecolor='black', alpha=0.85)
    ax.set_xlabel('Relative Importance Score')
    ax.set_ylabel('Predictor Feature')

    for index, value in enumerate(importances):
        ax.text(value + 0.005, index, f"{value:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    output_filename = "feature_importance_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Feature Importance plot saved to: {output_filename}")

if __name__ == "__main__":
    main()
