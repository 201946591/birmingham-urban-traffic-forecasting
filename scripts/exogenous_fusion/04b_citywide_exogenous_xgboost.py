# ==============================================================================
# City-Wide Exogenous Ablation Experiment: Evaluating Weather & Holiday Impact
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Methodological purpose:
# When evaluating Phase 4 spatial zoning (Major vs Minor roads), target volumes
# scale down from ~3,000 veh/hr (city-wide sum) to ~450-3,200 veh/hr per road type,
# which mathematically lowers raw RMSE.
#
# To isolate whether Open-Meteo ERA5 weather (temperature, precipitation) and
# bank holiday proximity truly improve prediction accuracy (rather than being an
# artifact of spatial disaggregation), this ablation script fits the identical
# exogenous feature space directly onto the unsegmented city-wide traffic series.
# This establishes a direct 1-to-1 comparison against the Phase 3 baseline (RMSE 804.76).

import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("=" * 70)
    print("CITY-WIDE EXOGENOUS ABLATION EXPERIMENT (WEATHER & CALENDAR EVALUATION)")
    print("=" * 70)

    # 1. Load city-wide engineered dataset from Phase 3
    script_dir = os.path.dirname(os.path.abspath(__file__))
    phase3_data = os.path.join(script_dir, "..", "phase_3", "engineered_birmingham_traffic.csv")
    
    if not os.path.exists(phase3_data):
        phase3_data = os.path.join(script_dir, "..", "engineered_birmingham_traffic.csv")
    if not os.path.exists(phase3_data):
        print(f"[ERROR] Could not locate engineered city-wide traffic data: {phase3_data}")
        sys.exit(1)

    print(f"[INFO] Reading city-wide engineered traffic dataset from: {phase3_data}")
    df = pd.read_csv(phase3_data)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    # 2. Ingest exogenous environmental and holiday feeds
    weather_file = os.path.join(script_dir, "birmingham_weather.csv")
    holidays_file = os.path.join(script_dir, "uk_holidays.csv")

    if not os.path.exists(weather_file) or not os.path.exists(holidays_file):
        print("[ERROR] Weather or holiday data missing. Please run 01_exogenous_data_fetch.py first.")
        sys.exit(1)

    weather_df = pd.read_csv(weather_file)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"]).dt.tz_localize(None)
    
    hol_df = pd.read_csv(holidays_file)
    hol_df["date"] = pd.to_datetime(hol_df["date"])

    # 3. Fuse exogenous features onto hourly city-wide timestamps
    print("[INFO] Merging hourly weather observations and holiday calendars...")
    df = df.merge(weather_df, on="datetime", how="left")
    
    # Impute isolated meteorological missing records via bidirectional fill
    df["temperature_2m"] = df["temperature_2m"].ffill().bfill()
    df["precipitation"] = df["precipitation"].ffill().bfill()

    # Engineer holiday proximity indicator (+/- 1 day window)
    df["date_only"] = df["datetime"].dt.normalize()
    holiday_dates_set = set(hol_df["date"].dt.normalize())
    holiday_dates_list = sorted(hol_df["date"].dt.normalize().tolist())
    
    df["is_holiday"] = df["date_only"].apply(lambda d: 1 if d in holiday_dates_set else 0)
    
    def is_near_holiday(d):
        for h in holiday_dates_list:
            if abs((d - h).days) <= 1:
                return 1
        return 0
    df["near_holiday"] = df["date_only"].apply(is_near_holiday)

    # 4. Define feature matrix and chronological train/test partition
    features = [
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "lag_1", "lag_2", "lag_3", "lag_24",
        "rolling_mean_3", "rolling_std_3",
        "temperature_2m", "precipitation", "is_holiday", "near_holiday"
    ]
    target = "all_motor_vehicles"
    
    X = df[features]
    y = df[target]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Train partition: {len(X_train):,} samples | Holdout: {len(X_test):,} samples")
    print(f"[INFO] Features evaluated ({len(features)}): {features}")

    # 5. Fit XGBoost regressor with identical hyperparameters
    print("\n[INFO] Fitting XGBoost regressor with exogenous features...")
    model = XGBRegressor(
        n_estimators=300, 
        learning_rate=0.03, 
        max_depth=5, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    phase3_baseline_rmse = 804.7589
    delta_rmse = phase3_baseline_rmse - rmse
    
    print("\n" + "=" * 70)
    print("ABLATION BENCHMARK RESULTS (CITY-WIDE SCALE)")
    print("=" * 70)
    print(f"  Phase 3 Baseline RMSE (Endogenous Only):    {phase3_baseline_rmse:.4f}")
    print(f"  Phase 4b Model RMSE (With Weather/Holidays): {rmse:.4f}")
    print(f"  Model MAE:                                  {mae:.4f}")
    print(f"  Model R² Score:                             {r2:.4f}")
    print("-" * 70)
    if delta_rmse > 0:
        print(f"  Empirical Outcome: Exogenous regressors yielded a marginal RMSE reduction of {delta_rmse:.2f} veh/hr.")
    else:
        print(f"  Empirical Outcome: Exogenous regressors did not improve accuracy (delta RMSE = {delta_rmse:+.2f} veh/hr).")
        print("  Findings support Section 4.4 discussion: local traffic inertia dominates weather effects in urban corridors.")
    print("=" * 70)

if __name__ == "__main__":
    main()

