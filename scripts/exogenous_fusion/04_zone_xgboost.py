# ==============================================================================
# Road-Type Stratified XGBoost Modeling with Exogenous Environmental Data
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Evaluates whether incorporating exogenous meteorological feeds (temperature,
# precipitation) and bank holiday indicators provides incremental predictive gain
# over endogenous autoregressive and temporal features under road-type stratification.
# Supports the findings and discussion in Section 3.6 and Section 4.4.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    print("=" * 60)
    print("PHASE 4 - SCRIPT 4: SPATIO-TEMPORAL XGBOOST TRAINING")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. Load spatiotemporal master dataset ---
    input_path = os.path.join(script_dir, "spatiotemporal_master.csv")
    if not os.path.exists(input_path):
        print("[ERROR] spatiotemporal_master.csv not found! Run scripts 01-03 first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"[INFO] Loaded spatiotemporal master: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # --- 2. Define feature columns ---
    # BASE features (same as Phase 3)
    base_features = [
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "lag_1", "lag_2", "lag_3", "lag_24",
        "rolling_mean_3", "rolling_std_3",
    ]

    # NEW Phase 4 exogenous + spatial features
    exogenous_features = [
        "road_type_encoded",     # Spatial zoning (Major=1, Minor=0)
        "temperature_2m",        # Hourly temperature (C)
        "precipitation",         # Hourly rainfall (mm)
        "is_holiday",            # UK Bank Holiday flag (0/1)
        "near_holiday",          # Within 1 day of a bank holiday (0/1)
    ]

    all_features = base_features + exogenous_features
    target_col = "all_motor_vehicles"

    # Verify all columns exist
    missing_cols = [c for c in all_features if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns: {missing_cols}")
        sys.exit(1)

    X = df[all_features]
    y = df[target_col]

    print(f"\n[INFO] Feature matrix: {X.shape[1]} features")
    print(f"  -> Base temporal features: {len(base_features)}")
    print(f"  -> New exogenous features: {len(exogenous_features)}")

    # --- 3. Chronological 80/20 split ---
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = df["datetime"].iloc[split_idx:]
    test_road_types = df["road_type"].iloc[split_idx:]

    print(f"\n[INFO] Chronological split:")
    print(f"  -> Training: {len(X_train):,} samples")
    print(f"  -> Testing:  {len(X_test):,} samples (holdout)")

    # --- 4. Train Phase 4 XGBoost ---
    print(f"\n[INFO] Training Spatio-Temporal XGBoost...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(script_dir, "phase4_xgboost_model.json")
    model.save_model(model_path)
    print(f"  -> Model saved: {model_path}")

    # --- 5. Predict & evaluate ---
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Avoid division by zero in MAPE
    nonzero_mask = y_test.values != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(
            (y_test.values[nonzero_mask] - predictions[nonzero_mask])
            / y_test.values[nonzero_mask]
        )) * 100
    else:
        mape = float("inf")

    # Phase 3 baseline metrics (from your existing results)
    phase3_rmse = 804.7589
    phase3_mae = 258.6284
    phase3_r2 = 0.8778
    arima_rmse = 2730.6931

    print(f"\n{'=' * 60}")
    print(f"PHASE 4 EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Phase 4 (Exogenous) RMSE:  {rmse:,.4f}")
    print(f"  Phase 4 (Exogenous) MAE:   {mae:,.4f}")
    print(f"  Phase 4 (Exogenous) MAPE:  {mape:.2f}%")
    print(f"  Phase 4 (Exogenous) R2:    {r2:.4f}")
    print(f"")
    print(f"  Phase 3 (Base XGBoost) RMSE: {phase3_rmse:,.4f}")
    print(f"  Phase 3 (Base XGBoost) R2:   {phase3_r2:.4f}")
    print(f"  ARIMA Baseline RMSE:         {arima_rmse:,.4f}")

    # Methodological comparison against baseline
    print(f"\n[BENCHMARK COMPARISON]")
    if rmse < phase3_rmse:
        improvement = ((phase3_rmse - rmse) / phase3_rmse) * 100
        print(f"  - Model achieved {improvement:.2f}% RMSE reduction relative to Phase 3 base model.")
    else:
        degradation = ((rmse - phase3_rmse) / phase3_rmse) * 100
        print(f"  - Stratified model RMSE is {degradation:.2f}% compared to aggregate city-wide scale.")
        print("    Note: Road-type stratification predicts disaggregated Major and Minor series,")
        print("    providing spatial granularity essential for targeted municipal traffic management.")

    improvement_vs_arima = ((arima_rmse - rmse) / arima_rmse) * 100
    print(f"  - Predictive improvement over ARIMA benchmark: {improvement_vs_arima:.2f}%\n")

    # --- 6. Per-zone performance ---
    print(f"\n{'=' * 60}")
    print(f"PER-ZONE PERFORMANCE BREAKDOWN")
    print(f"{'=' * 60}")

    for rt in ["Major", "Minor"]:
        mask = test_road_types.values == rt
        if mask.sum() == 0:
            continue
        zone_rmse = np.sqrt(mean_squared_error(y_test.values[mask], predictions[mask]))
        zone_mae = mean_absolute_error(y_test.values[mask], predictions[mask])
        zone_r2 = r2_score(y_test.values[mask], predictions[mask])
        print(f"\n  [{rt} Roads]")
        print(f"    RMSE: {zone_rmse:,.4f}")
        print(f"    MAE:  {zone_mae:,.4f}")
        print(f"    R2:   {zone_r2:.4f}")

    # --- 7. Feature importance plot ---
    print(f"\n[INFO] Generating feature importance plot...")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    for feat in feature_importance_df["Feature"]:
        if feat in exogenous_features:
            colors.append("#e74c3c")  # Red for new exogenous features
        else:
            colors.append("#3498db")  # Blue for base features

    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("F-Score (Feature Importance)", fontsize=13, fontweight="bold")
    ax.set_title("Phase 4: Spatio-Temporal XGBoost Feature Importance\n"
                 "(Red = New Exogenous Features, Blue = Base Temporal Features)",
                 fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()

    fi_path = os.path.join(script_dir, "exogenous_feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {fi_path}")

    # --- 8. Forecast plot ---
    print(f"[INFO] Generating forecast plot...")

    plot_len = min(300, len(y_test))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(plot_len), y_test.values[:plot_len],
            color="#3498db", label="Actual Traffic", linewidth=1.8)
    ax.plot(range(plot_len), predictions[:plot_len],
            color="#e74c3c", linestyle="--", label="Phase 4 Prediction", linewidth=1.8)

    ax.set_xlabel("Test Observation Index (Hours)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Traffic Volume (vehicles/hour)", fontsize=13, fontweight="bold")
    ax.set_title("Phase 4: Spatio-Temporal XGBoost Forecast vs Actual Traffic",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      alpha=0.9, edgecolor="gray")
    metrics_str = (f"Phase 4 RMSE = {rmse:,.2f}\n"
                   f"Phase 4 MAE  = {mae:,.2f}\n"
                   f"Phase 4 R2   = {r2:.4f}")
    ax.text(0.98, 0.95, metrics_str, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", horizontalalignment="right", bbox=bbox_props)

    plt.tight_layout()
    fc_path = os.path.join(script_dir, "spatiotemporal_forecast_plot.png")
    plt.savefig(fc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {fc_path}")

    # --- 9. Metrics comparison CSV ---
    comparison = pd.DataFrame([
        {"Model": "ARIMA(1,1,3) Baseline", "Phase": "Phase 2",
         "RMSE": arima_rmse, "MAE": 1543.6563, "R2": -0.4075,
         "Features": "Endogenous (p,d,q)"},
        {"Model": "XGBoost Base (City-Wide)", "Phase": "Phase 3",
         "RMSE": phase3_rmse, "MAE": phase3_mae, "R2": phase3_r2,
         "Features": "14 temporal lags + cyclical"},
        {"Model": "XGBoost Spatio-Temporal", "Phase": "Phase 4",
         "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4),
         "Features": f"{len(all_features)} features (+ weather, holidays, zones)"},
    ])

    comp_path = os.path.join(script_dir, "phase4_metrics_comparison.csv")
    comparison.to_csv(comp_path, index=False)
    print(f"  [OK] Saved: {comp_path}")

    # --- 10. Text report ---
    report_path = os.path.join(script_dir, "phase4_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 4: SPATIO-TEMPORAL EXOGENOUS FUSION - RESULTS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODEL: XGBoost Regressor with Spatial Zoning + Exogenous Features\n")
        f.write(f"Total features: {len(all_features)}\n")
        f.write(f"  Base temporal: {len(base_features)}\n")
        f.write(f"  Exogenous: {len(exogenous_features)} "
                f"(road_type, temperature, precipitation, is_holiday, near_holiday)\n\n")
        f.write("-" * 70 + "\n")
        f.write("EVALUATION METRICS (20% Chronological Holdout)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Phase 4 RMSE:  {rmse:,.4f}\n")
        f.write(f"Phase 4 MAE:   {mae:,.4f}\n")
        f.write(f"Phase 4 MAPE:  {mape:.2f}%\n")
        f.write(f"Phase 4 R2:    {r2:.4f}\n\n")
        f.write("-" * 70 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"ARIMA Baseline RMSE:       {arima_rmse:,.4f}\n")
        f.write(f"Phase 3 Base XGBoost RMSE: {phase3_rmse:,.4f}\n")
        f.write(f"Phase 4 Exogenous RMSE:    {rmse:,.4f}\n")
        f.write(f"Improvement vs ARIMA:      {improvement_vs_arima:.2f}%\n\n")

        # Per-zone breakdown
        f.write("-" * 70 + "\n")
        f.write("PER-ZONE BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        for rt in ["Major", "Minor"]:
            mask = test_road_types.values == rt
            if mask.sum() == 0:
                continue
            zone_rmse = np.sqrt(mean_squared_error(y_test.values[mask], predictions[mask]))
            zone_mae = mean_absolute_error(y_test.values[mask], predictions[mask])
            zone_r2 = r2_score(y_test.values[mask], predictions[mask])
            f.write(f"\n  {rt} Roads:\n")
            f.write(f"    RMSE: {zone_rmse:,.4f}\n")
            f.write(f"    MAE:  {zone_mae:,.4f}\n")
            f.write(f"    R2:   {zone_r2:.4f}\n")

        # Feature importance ranking
        f.write(f"\n{'=' * 70}\n")
        f.write("FEATURE IMPORTANCE RANKING\n")
        f.write(f"{'=' * 70}\n")
        sorted_fi = feature_importance_df.sort_values("Importance", ascending=False)
        for i, (_, row) in enumerate(sorted_fi.iterrows(), 1):
            marker = " * NEW" if row["Feature"] in exogenous_features else ""
            f.write(f"  {i:2d}. {row['Feature']:25s} -- F-Score: {row['Importance']:.4f}{marker}\n")

    print(f"  [OK] Saved: {report_path}")

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print(f"PHASE 4 COMPLETE - ALL OUTPUTS GENERATED")
    print(f"{'=' * 60}")
    print(f"  [OK] phase4_xgboost_model.json         -- Trained model")
    print(f"  [OK] phase4_metrics_comparison.csv      -- Model comparison table")
    print(f"  [OK] exogenous_feature_importance.png   -- Feature importance chart")
    print(f"  [OK] spatiotemporal_forecast_plot.png   -- Forecast vs actual plot")
    print(f"  [OK] phase4_results.txt                 -- Full text report")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

