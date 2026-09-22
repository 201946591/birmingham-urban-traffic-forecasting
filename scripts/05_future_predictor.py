# ==============================================================================
# Future Traffic Flow Inference & Multi-Modal Decomposition
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Inference utility: Loads the trained XGBoost model checkpoint and produces
# point predictions with multi-modal disaggregation across individual vehicle classes.

import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def predict_traffic(model, target_datetime, recent_traffic=None):
    """
    Predict traffic volume for a given datetime.
    """
    dt = pd.to_datetime(target_datetime)

    hour = dt.hour
    day_of_week = dt.dayofweek
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0

    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)

    # Defaults based on Birmingham hourly averages
    if 7 <= hour <= 9:       # Morning rush
        lag_1, lag_2, lag_3, lag_24 = 4500, 3800, 3000, 4500
    elif 16 <= hour <= 18:   # Evening rush
        lag_1, lag_2, lag_3, lag_24 = 5000, 4800, 4500, 5000
    elif 0 <= hour <= 5:     # Night time
        lag_1, lag_2, lag_3, lag_24 = 500, 600, 700, 500
    else:                    # Normal daytime
        lag_1, lag_2, lag_3, lag_24 = 3500, 3200, 3000, 3500

    if is_weekend:
        lag_1 *= 0.7
        lag_2 *= 0.7
        lag_3 *= 0.7
        lag_24 *= 0.7

    rolling_mean_3 = np.mean([lag_1, lag_2, lag_3])
    rolling_std_3 = np.std([lag_1, lag_2, lag_3])

    features = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'lag_24': lag_24,
        'rolling_mean_3': rolling_mean_3,
        'rolling_std_3': rolling_std_3,
    }])

    prediction = model.predict(features)[0]
    return max(0, prediction)


def main():
    model_path = "xgboost_model.json"
    if not os.path.exists(model_path):
        model_path = "phase_3/xgboost_model.json"
    if not os.path.exists(model_path):
        print("[ERROR] Cannot find xgboost_model.json!")
        print("  -> Please run 02_xgboost_model_training.py first to train & save the model.")
        sys.exit(1)

    print("=" * 65)
    print("PHASE 3 — SCRIPT 5: GRANULAR FUTURE TRAFFIC PREDICTOR")
    print("=" * 65)

    model = XGBRegressor()
    model.load_model(model_path)
    print(f"[INFO] Trained XGBoost model loaded from: {model_path}\n")

    print("Enter a future date and time to predict Birmingham traffic volume.")
    print("Format: YYYY-MM-DD HH:MM  (e.g. 2026-08-15 08:00)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input(">> Enter date & time: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n[INFO] Exiting predictor. Goodbye!")
            break

        try:
            dt = pd.to_datetime(user_input)
        except Exception:
            print("  [ERROR] Invalid format! Use: YYYY-MM-DD HH:MM (e.g. 2026-08-15 08:00)\n")
            continue

        predicted_total = predict_traffic(model, dt)

        day_name = dt.strftime('%A')
        date_str = dt.strftime('%d %B %Y, %H:%M')
        weekend_flag = "Weekend" if dt.dayofweek >= 5 else "Weekday"

        # Calculate empirical vehicle breakdown
        cars = predicted_total * 0.8057
        lgvs = predicted_total * 0.1243
        hgvs = predicted_total * 0.0453
        buses = predicted_total * 0.0155
        motorcycles = predicted_total * 0.0057
        cyclists = predicted_total * 0.0035

        print("\n" + "=" * 65)
        print(f"PREDICTION FOR: {date_str} ({day_name}, {weekend_flag})")
        print("=" * 65)
        print(f" TOTAL PREDICTED VOLUME: {predicted_total:,.0f} vehicles/hour (Birmingham-wide)\n")

        # Empirical modal split breakdown based on 72,948 observed DfT survey counts:
        # Cars & Taxis: 80.57%, LGVs: 12.43%, HGVs: 4.53%, Buses: 1.55%, Motorcycles: 0.57%, Cycles: 0.35%
        print(" MODAL SPLIT DISAGGREGATION (EMPIRICAL DfT RATIOS):")
        print(" " + "-" * 61)
        print(f"  [Passenger] Cars & Taxis:         {cars:>8,.0f} veh/hr (80.57%)")
        print(f"  [Light Freight] LGVs / Delivery:  {lgvs:>8,.0f} veh/hr (12.43%)")
        print(f"  [Heavy Freight] HGVs:             {hgvs:>8,.0f} veh/hr ( 4.53%)")
        print(f"  [Transit] Buses & Coaches:        {buses:>8,.0f} veh/hr ( 1.55%)")
        print(f"  [Two-Wheeled] Motorcycles:        {motorcycles:>8,.0f} veh/hr ( 0.57%)")
        print(" " + "-" * 61)
        print(f"  [Active Travel] Pedal Cycles:     {cyclists:>8,.0f} cyc/hr ( 0.35%)")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

