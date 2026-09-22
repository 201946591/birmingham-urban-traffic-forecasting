# ==============================================================================
# Master Methodological Benchmark Pipeline
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# Module: EBUS621 Dissertation Project - University of Liverpool Management School
# ==============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("FINAL METHODOLOGICAL BENCHMARK (OBSERVED DAYTIME DATA)")
    print("=" * 60)

    filename = "dft_rawcount_local_authority_id_141.csv"
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "..", filename),
        os.path.join(os.path.dirname(__file__), "..", "data", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", filename),
    ]
    input_path = None
    for cand in candidates:
        if os.path.exists(cand):
            input_path = os.path.abspath(cand)
            break

    if input_path is None:
        print(f"[ERROR] Could not locate raw DfT count file: {filename}")
        sys.exit(1)
        
    print(f"[INFO] Reading DfT survey data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    
    # Construct timestamp from count_date and survey hour
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df["datetime"] = pd.to_datetime(df["count_date"]) + pd.to_timedelta(df["hour"], unit="h")
        
    # Department for Transport manual counts are recorded during active daytime windows
    # (typically 07:00 to 19:00 on neutral weekdays). We intentionally stick strictly to
    # genuinely observed hours and avoid synthetic night-time imputation.
    df = df.sort_values(["datetime"]).reset_index(drop=True)
    
    # 2. Road-Type Stratification (Major vs Minor)
    # Major roads (arterials) carry much higher volume baselines than Minor residential roads.
    # We aggregate hourly counts across each road class to preserve this structural distinction.
    print(f"[INFO] Building road-type stratified hourly series...")
    zonal_df = df.groupby(["datetime", "road_type"])["all_motor_vehicles"].sum().reset_index()
    zonal_df = zonal_df.sort_values(["road_type", "datetime"]).reset_index(drop=True)

    # 3. Temporal Feature Engineering (Strictly Past-Only)
    zonal_df["hour"] = zonal_df["datetime"].dt.hour
    zonal_df["day_of_week"] = zonal_df["datetime"].dt.dayofweek
    zonal_df["month"] = zonal_df["datetime"].dt.month
    
    # Cyclical trigonometric transformations for hour and weekday
    # Sine/cosine mapping avoids artificial jumps between hour 23 and hour 0.
    zonal_df["hour_sin"] = np.sin(2 * np.pi * zonal_df["hour"] / 24.0)
    zonal_df["hour_cos"] = np.cos(2 * np.pi * zonal_df["hour"] / 24.0)
    zonal_df["dow_sin"] = np.sin(2 * np.pi * zonal_df["day_of_week"] / 7.0)
    zonal_df["dow_cos"] = np.cos(2 * np.pi * zonal_df["day_of_week"] / 7.0)

    # Autoregressive lag features (t-1, t-2, t-3)
    # Group by road_type so lag shifts never bleed across different road categories.
    lag_cols = []
    for lag in [1, 2, 3]:
        col_name = f"lag_{lag}"
        zonal_df[col_name] = zonal_df.groupby("road_type")["all_motor_vehicles"].shift(lag)
        lag_cols.append(col_name)

    # Rolling window statistics (past 3-hour mean and std)
    # Shift by 1 first to strictly prevent target leakage from the current hour.
    zonal_df["rolling_mean_3"] = zonal_df.groupby("road_type")["all_motor_vehicles"].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    zonal_df["rolling_std_3"] = zonal_df.groupby("road_type")["all_motor_vehicles"].transform(
        lambda x: x.shift(1).rolling(window=3).std()
    )

    # Drop boundary rows where initial 3-hour lags are undefined
    zonal_df = zonal_df.dropna().reset_index(drop=True)
    
    # Binary encoding: Major = 1, Minor = 0
    zonal_df["road_type_encoded"] = zonal_df["road_type"].map({"Major": 1, "Minor": 0})
    
    # Sort chronologically before partitioning
    zonal_df = zonal_df.sort_values("datetime").reset_index(drop=True)

    # Feature space specification (10 predictors)
    features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", 
        "lag_1", "lag_2", "lag_3", 
        "rolling_mean_3", "rolling_std_3", 
        "road_type_encoded"
    ]
    target = "all_motor_vehicles"

    # 4. Strict Chronological 80/20 Train-Test Split
    # We never use random k-fold shuffling on time-series to avoid future data leakage.
    # The holdout set deliberately spans March 2019 to 2025 to test model robustness
    # against COVID-19 pandemic lockdowns and post-pandemic recovery.
    split_idx = int(len(zonal_df) * 0.8)
    train_df = zonal_df.iloc[:split_idx]
    test_df = zonal_df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"\n[INFO] Training samples: {len(X_train):,} | Testing holdout: {len(X_test):,}")
    print(f"[INFO] Features ({len(features)}): {features}")
    
    # 5. Model Specifications
    # Benchmark suite: Statistical Linear Baseline vs. Conventional Trees vs. Gradient Boosting
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=10, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=5, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
    }

    # 6. Model Training and Evaluation on Unseen Test Partition
    results = []
    
    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate standard evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 4)
        })
        
        # Road-type stratified error breakdown for XGBoost
        if name == "XGBoost Regressor":
            test_df["xgb_preds"] = preds
            print("  >> Stratified Error Breakdown (XGBoost):")
            for rt, encoded in [("Major", 1), ("Minor", 0)]:
                subset = test_df[test_df["road_type_encoded"] == encoded]
                sub_rmse = np.sqrt(mean_squared_error(subset[target], subset["xgb_preds"]))
                sub_r2 = r2_score(subset[target], subset["xgb_preds"])
                print(f"     - {rt} Roads: RMSE = {sub_rmse:.2f} | R2 = {sub_r2:.4f}")

    # 7. Print and Save Comparative Scoreboard
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("FINAL MODEL BENCHMARK SCOREBOARD (OBSERVED DAYTIME DATA)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    with open("final_benchmark_results.txt", "w") as f:
        f.write("FINAL MODEL BENCHMARK SCOREBOARD (OBSERVED DAYTIME DATA)\n")
        f.write("=" * 60 + "\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("Stratified Analysis (XGBoost):\n")
        for rt, encoded in [("Major", 1), ("Minor", 0)]:
            subset = test_df[test_df["road_type_encoded"] == encoded]
            sub_rmse = np.sqrt(mean_squared_error(subset[target], subset["xgb_preds"]))
            sub_r2 = r2_score(subset[target], subset["xgb_preds"])
            f.write(f" - {rt} Roads: RMSE = {sub_rmse:.2f} | R2 = {sub_r2:.4f}\n")
            
    print("\n[SUCCESS] Benchmark completed. Results saved to 'final_benchmark_results.txt'")

if __name__ == "__main__":
    main()

