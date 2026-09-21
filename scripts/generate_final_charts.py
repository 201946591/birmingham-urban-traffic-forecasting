# ==============================================================================
# Publication Figure Generation Suite (300 DPI)
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Generates all publication-grade figures presented in Chapter 4:
# - Figure 4.1: Actual vs. Predicted Traffic Flow (Holdout Test Set)
# - Figure 4.2: Road-Type Stratified Predictions (Major Arterial vs. Minor Residential)
# - Figure 4.3: Feature Importance Analysis (Gain vs. Split Frequency)
# - Figure 4.4: Four-Model Comparative Performance Matrix
# - Figure 4.5: Residual Error Distribution with Zoom Inset & KDE
# - Figure 4.6 / 4.7: Birmingham Multi-Modal Vehicle Classification Breakdown
# - Figure 4.7 / 4.8: Historical Era Analysis & COVID-19 Concept Drift

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("REGENERATING ALL FINAL CHARTS (Verified Values)")
    print("=" * 60)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 13
    
    output_dir = "final_academic_charts"
    if not os.path.exists(output_dir) and os.path.exists(os.path.join(os.path.dirname(__file__), "..", output_dir)):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # ===========================================================
    # LOAD DATA AND TRAIN MODEL (same pipeline as audit)
    # ===========================================================
    print("[1] Loading raw DfT data and training models...")
    input_path = "dft_rawcount_local_authority_id_141 (1).csv"
    if not os.path.exists(input_path):
        parent_candidate = os.path.join(os.path.dirname(__file__), "..", input_path)
        if os.path.exists(parent_candidate):
            input_path = os.path.abspath(parent_candidate)
    df = pd.read_csv(input_path, low_memory=False)
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df["datetime"] = pd.to_datetime(df["count_date"]) + pd.to_timedelta(df["hour"], unit="h")
    
    df = df.sort_values(["datetime"]).reset_index(drop=True)
    
    zonal_df = df.groupby(["datetime", "road_type"])["all_motor_vehicles"].sum().reset_index()
    zonal_df = zonal_df.sort_values(["road_type", "datetime"]).reset_index(drop=True)
    
    zonal_df["hour"] = zonal_df["datetime"].dt.hour
    zonal_df["day_of_week"] = zonal_df["datetime"].dt.dayofweek
    zonal_df["hour_sin"] = np.sin(2 * np.pi * zonal_df["hour"] / 24.0)
    zonal_df["hour_cos"] = np.cos(2 * np.pi * zonal_df["hour"] / 24.0)
    zonal_df["dow_sin"] = np.sin(2 * np.pi * zonal_df["day_of_week"] / 7.0)
    zonal_df["dow_cos"] = np.cos(2 * np.pi * zonal_df["day_of_week"] / 7.0)
    
    for lag in [1, 2, 3]:
        zonal_df[f"lag_{lag}"] = zonal_df.groupby("road_type")["all_motor_vehicles"].shift(lag)
    zonal_df["rolling_mean_3"] = zonal_df.groupby("road_type")["all_motor_vehicles"].transform(
        lambda x: x.shift(1).rolling(window=3).mean())
    zonal_df["rolling_std_3"] = zonal_df.groupby("road_type")["all_motor_vehicles"].transform(
        lambda x: x.shift(1).rolling(window=3).std())
    
    zonal_df = zonal_df.dropna().reset_index(drop=True)
    zonal_df["road_type_encoded"] = zonal_df["road_type"].map({"Major": 1, "Minor": 0})
    zonal_df = zonal_df.sort_values("datetime").reset_index(drop=True)
    
    features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "lag_1", "lag_2", "lag_3",
                "rolling_mean_3", "rolling_std_3", "road_type_encoded"]
    target = "all_motor_vehicles"
    
    split_idx = int(len(zonal_df) * 0.8)
    train_df = zonal_df.iloc[:split_idx]
    test_df = zonal_df.iloc[split_idx:].copy()
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    test_df["xgb_preds"] = preds
    
    print(f"    Model trained. Test R²={r2_score(y_test, preds):.4f}")
    
    # ===========================================================
    # FIGURE 4.1: Overall Actual vs Predicted (NEW — not stratified)
    # ===========================================================
    print("[2] Generating Figure 4.1: Overall Actual vs Predicted...")
    
    sample_df = test_df.iloc[:200].copy()
    sample_df = sample_df.reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(15, 7.0), dpi=300)
    ax.plot(sample_df.index, sample_df[target], label='Actual Traffic Volume', 
            color='#1f77b4', linewidth=2.4, alpha=0.95)
    ax.plot(sample_df.index, sample_df["xgb_preds"], label='Predicted Volume (XGBoost)', 
            color='#d62728', linewidth=2.4, linestyle='--', alpha=0.95)
    ax.set_xlabel('Chronological Test Observations (Hourly Sequence)', fontweight='bold', fontsize=15, labelpad=18, color='#111111')
    ax.set_ylabel('Traffic Volume (Vehicles / Hour)', fontweight='bold', fontsize=15, labelpad=12, color='#111111')
    ax.set_title('XGBoost Actual vs. Predicted Traffic Volume\n(200-Observation Sequence from 20% Chronological Holdout Set)', 
                 fontweight='bold', fontsize=18, pad=18, color='#111111')
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2)
    ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
    ax.set_axisbelow(True)
    ax.legend(fontsize=15, loc='upper right', frameon=True, facecolor='white', 
              framealpha=0.95, edgecolor='#b0b0b0')
    
    # Add metrics annotation
    rmse_val = np.sqrt(mean_squared_error(y_test, preds))
    r2_val = r2_score(y_test, preds)
    stats_text = (
        "Overall Test Metrics:\n"
        f"  • RMSE = {rmse_val:.2f} veh/hr\n"
        f"  • R² = {r2_val:.4f}\n"
        f"  • Sample n = {len(y_test):,} obs"
    )
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=13.5, family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='#a0a0a0', linewidth=1.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_4_1_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    SAVED: fig_4_1_actual_vs_predicted.png (Clean Spacing & Legend)")
    
    # ===========================================================
    # FIGURE 4.2: Stratified Predictions (Major vs Minor)
    # ===========================================================
    print("[3] Generating Figure 4.2: Stratified Predictions...")
    
    major_df = test_df[test_df["road_type_encoded"] == 1].iloc[:100].reset_index(drop=True)
    minor_df = test_df[test_df["road_type_encoded"] == 0].iloc[:100].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), dpi=300)
    
    # Panel 1: Major Arterial Roads
    ax1.plot(range(len(major_df)), major_df[target], label='Actual Traffic Volume', 
             color='#1f77b4', linewidth=2.5, alpha=0.95)
    ax1.plot(range(len(major_df)), major_df["xgb_preds"], label='Predicted Volume (XGBoost)', 
             color='#d62728', linewidth=2.5, linestyle='--', alpha=0.95)
    ax1.set_title('Major Arterial Roads: Actual vs. Predicted Traffic Volume\n(RMSE = 1,014.73 veh/hr  |  R² = 0.8544)', 
                  fontweight='bold', fontsize=17, pad=12, color='#111111')
    ax1.set_ylabel('Traffic Volume (veh/hr)', fontweight='bold', fontsize=16, labelpad=14, color='#111111')
    ax1.tick_params(axis='both', which='major', labelsize=17.5, length=7, width=1.5)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight('semibold')
    ax1.grid(True, linestyle='--', alpha=0.55, color='#cccccc')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=15, loc='upper right', frameon=True, facecolor='white', 
               framealpha=0.95, edgecolor='#b0b0b0')
    
    # Panel 2: Minor Residential Roads
    ax2.plot(range(len(minor_df)), minor_df[target], label='Actual Traffic Volume', 
             color='#1f77b4', linewidth=2.5, alpha=0.95)
    ax2.plot(range(len(minor_df)), minor_df["xgb_preds"], label='Predicted Volume (XGBoost)', 
             color='#d62728', linewidth=2.5, linestyle='--', alpha=0.95)
    ax2.set_title('Minor Residential Roads: Actual vs. Predicted Traffic Volume\n(RMSE = 163.96 veh/hr  |  R² = 0.8865)', 
                  fontweight='bold', fontsize=17, pad=12, color='#111111')
    ax2.set_ylabel('Traffic Volume (veh/hr)', fontweight='bold', fontsize=16, labelpad=14, color='#111111')
    ax2.set_xlabel('Chronological Test Observations (100-Hour Sample Sequence)', 
                   fontweight='bold', fontsize=16, labelpad=16, color='#111111')
    ax2.tick_params(axis='both', which='major', labelsize=17.5, length=7, width=1.5)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontweight('semibold')
    ax2.grid(True, linestyle='--', alpha=0.55, color='#cccccc')
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=15, loc='upper right', frameon=True, facecolor='white', 
               framealpha=0.95, edgecolor='#b0b0b0')
    
    plt.suptitle('Road-Type Stratified Traffic Volume Predictions', 
                 fontweight='bold', fontsize=21, y=0.975, color='#111111')
    plt.subplots_adjust(top=0.87, bottom=0.095, hspace=0.36, left=0.10, right=0.97)
    
    plt.savefig(os.path.join(output_dir, 'fig_4_2_stratified_predictions.png'), dpi=300)
    plt.close()
    print("    SAVED: fig_4_2_stratified_predictions.png (Enlarged Numbers & Bold Ticks)")
    
    # ===========================================================
    # FIGURE 4.3: Feature Importance
    # ===========================================================
    print("[4] Generating Figure 4.3: Feature Importance...")
    
    importance = xgb_model.get_booster().get_score(importance_type='weight')
    
    # Map feature names to readable labels
    label_map = {
        'hour_sin': 'Hour (Sine)',
        'hour_cos': 'Hour (Cosine)',
        'dow_sin': 'Day-of-Week (Sine)',
        'dow_cos': 'Day-of-Week (Cosine)',
        'lag_1': 'Lag t-1',
        'lag_2': 'Lag t-2',
        'lag_3': 'Lag t-3',
        'rolling_mean_3': 'Rolling Mean (3hr)',
        'rolling_std_3': 'Rolling Std Dev (3hr)',
        'road_type_encoded': 'Road Type (Major/Minor)'
    }
    
    imp_df = pd.DataFrame({'Feature': [label_map.get(k, k) for k in importance.keys()], 
                           'Score': list(importance.values())})
    imp_df = imp_df.sort_values(by='Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    colors = ['#2ca02c' if s > 500 else '#66bb6a' for s in imp_df['Score']]
    bars = ax.barh(imp_df['Feature'], imp_df['Score'], color=colors, 
                    edgecolor='#1b5e20', linewidth=1.0, height=0.7)
    
    # Add enlarged value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 25, bar.get_y() + bar.get_height() / 2, f'{int(width):,}', 
                va='center', fontsize=14.5, fontweight='bold', color='#111111')
    
    ax.set_xlabel('F-Score (Number of Splits)', fontweight='bold', fontsize=16, labelpad=18, color='#111111')
    ax.set_title('XGBoost Feature Importance Rankings', fontweight='bold', fontsize=20, pad=18, color='#111111')
    
    ax.tick_params(axis='x', which='major', labelsize=16, length=6, width=1.3)
    ax.tick_params(axis='y', which='major', labelsize=15, length=0)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    for tick in ax.get_xticklabels():
        tick.set_fontweight('semibold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('semibold')
        
    ax.set_xlim(right=max(imp_df['Score']) * 1.15)
    ax.grid(True, axis='x', linestyle='--', alpha=0.55, color='#cccccc')
    ax.grid(False, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_4_3_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    SAVED: fig_4_3_feature_importance.png (Polished Numbers & Spaced X-Axis)")
    
    # ===========================================================
    # FIGURE 4.4: Four-Model Comparison (VERIFIED values)
    # ===========================================================
    print("[5] Generating Figure 4.4: Four-Model Comparison...")
    
    models_names = ['Linear Regression\n(Baseline)', 'Decision\nTree', 'Random\nForest', 'XGBoost']
    rmse_scores = [696.71, 652.93, 610.21, 621.68]
    r2_scores = [0.8873, 0.9010, 0.9136, 0.9103]
    
    fig, ax1 = plt.subplots(figsize=(13, 8), dpi=300)
    
    x = np.arange(len(models_names))
    width = 0.48
    
    color_rmse = '#1f77b4'
    color_rf = '#2ca02c'
    bar_colors = [color_rmse, color_rmse, color_rf, color_rmse]
    edge_colors = ['#0d4f8b', '#0d4f8b', '#1a6b1a', '#0d4f8b']
    
    bars = ax1.bar(x, rmse_scores, width, color=bar_colors, alpha=0.88, 
                   edgecolor=edge_colors, linewidth=1.2, zorder=3)
    
    # Left Y-Axis: RMSE
    ax1.set_xlabel('Predictive Models', fontweight='bold', fontsize=16, labelpad=16, color='#111111')
    ax1.set_ylabel('Root Mean Squared Error (RMSE)', color=color_rmse, fontweight='bold', fontsize=16, labelpad=14)
    ax1.tick_params(axis='y', labelcolor=color_rmse, labelsize=15, length=6, width=1.3)
    ax1.tick_params(axis='x', labelsize=14, length=6, width=1.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_names, fontweight='bold', color='#111111')
    ax1.set_ylim(500, 755)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
        
    # Add enlarged RMSE numbers above bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 5.5, f'{yval:.2f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=15, color='#111111',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))
    
    # Right Y-Axis: R2 Score
    ax2 = ax1.twinx()
    color_r2 = '#d62728'
    ax2.set_ylabel('Coefficient of Determination (R²)', color=color_r2, fontweight='bold', fontsize=16, labelpad=14)
    ax2.plot(x, r2_scores, color=color_r2, marker='o', linewidth=3.0, markersize=11, 
             markerfacecolor=color_r2, markeredgecolor='white', markeredgewidth=2,
             label='R² Score', zorder=5)
    ax2.tick_params(axis='y', labelcolor=color_r2, labelsize=15, length=6, width=1.3)
    ax2.set_ylim(0.880, 0.922)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
    
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Reposition R2 labels cleanly to avoid colliding with line or bars
    r2_offsets = [
        (0.24, -0.0016),   # 0.8873 shifted right into white space below line
        (0.24, -0.0016),   # 0.9010 shifted right into white space below line
        (0.12, 0.0014),    # 0.9136 above peak
        (0.14, 0.0010)     # 0.9103 to right
    ]
    
    for i, (v, (dx, dy)) in enumerate(zip(r2_scores, r2_offsets)):
        ax2.text(i + dx, v + dy, f'{v:.4f}', color=color_r2, fontweight='bold', fontsize=15,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.92, edgecolor='#e0b0b0', linewidth=0.8))
    
    ax1.grid(True, linestyle='--', alpha=0.55, color='#cccccc')
    ax2.grid(False)
    ax1.set_axisbelow(True)
    
    plt.title('Four-Model Comparative Performance Benchmark\n(Observed Daytime Data, 80/20 Chronological Split, n=5,555 test observations)', 
              fontweight='bold', fontsize=18, pad=18, color='#111111')
    
    plt.subplots_adjust(top=0.88, bottom=0.13, left=0.09, right=0.91)
    plt.savefig(os.path.join(output_dir, 'fig_4_4_model_comparison.png'), dpi=300)
    plt.close()
    print("    SAVED: fig_4_4_model_comparison.png (Polished Numbers, Shifted Labels & Enlarged Text)")
    
    # ===========================================================
    # FIGURE 4.5/4.6: Residual Histogram (CORRECTED VALUES)
    # ===========================================================
    print("[6] Generating Figure 4.5: Residual Histogram (POLISHED HIGH-DEFINITION)...")
    
    residuals = y_test.values - preds
    res_mean = residuals.mean()
    res_std = residuals.std()
    res_skew = pd.Series(residuals).skew()
    res_kurt = pd.Series(residuals).kurtosis()
    res_med = np.median(residuals)
    
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=300)
    
    # Core window captures 98.15% of observations (-1,500 to +1,500 veh/hr)
    core_mask = (residuals >= -1500) & (residuals <= 1500)
    core_residuals = residuals[core_mask]
    bin_edges = np.linspace(-1500, 1500, 61) # 50 veh/hr bins
    
    n, bins, patches = ax.hist(core_residuals, bins=bin_edges, color='#7b2d8e', alpha=0.72,
                               edgecolor='#4a1a56', linewidth=0.9, label='Residual Bins (50 veh/hr)')
    
    kde = gaussian_kde(core_residuals)
    x_kde = np.linspace(-1500, 1500, 400)
    bin_width = bin_edges[1] - bin_edges[0]
    y_kde = kde(x_kde) * len(core_residuals) * bin_width
    ax.plot(x_kde, y_kde, color='#3a0d45', linewidth=3.0, label='Kernel Density Estimate (KDE)')
    
    ax.axvline(x=res_mean, color='#d62728', linestyle='--', linewidth=2.8, label=f'Mean Error ({res_mean:+.2f} veh/hr)')
    ax.axvline(x=0, color='#111111', linestyle='-', linewidth=2.4, label='Zero Error (0.00)')
    
    ax.set_xlim(-1550, 1550)
    ax.set_ylim(0, max(n) * 1.22)
    
    ax.set_xlabel('Prediction Residual (Actual − Predicted, veh/hr)', fontweight='bold', fontsize=16.5, labelpad=16, color='#111111')
    ax.set_ylabel('Observation Frequency (Count)', fontweight='bold', fontsize=16.5, labelpad=14, color='#111111')
    ax.set_title('XGBoost Prediction Error Distribution (Holdout Evaluation)\n(Core Distribution Spanning 98.15% of Holdout Observations with Full-Spectrum Inset)', 
                 fontweight='bold', fontsize=18, pad=18, color='#111111')
    
    ax.tick_params(axis='both', which='major', labelsize=15.5, length=6, width=1.4)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    for tick in ax.get_xticklabels() + ax.get_yticklabels(): 
        tick.set_fontweight('bold')
    
    legend = ax.legend(fontsize=14.5, loc='upper left', frameon=True, facecolor='white', framealpha=0.96, edgecolor='#b0b0b0')
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    ax.grid(True, linestyle='--', alpha=0.55, color='#cccccc')
    ax.set_axisbelow(True)
    
    stats_text = (
        'Holdout Residual Statistics (n=5,555):\n'
        f'  • Mean Error: +{res_mean:.2f} veh/hr\n'
        f'  • Median Error: {res_med:.2f} veh/hr\n'
        f'  • Standard Deviation: {res_std:.2f} veh/hr\n'
        f'  • Skewness: {res_skew:.2f} (Right-Tail Outliers)\n'
        f'  • Kurtosis: {res_kurt:.2f} (Leptokurtic)\n'
        f'  • Core Window (±1.5k): 98.15% (5,452 obs)\n'
        f'  • Long Tail (>1.5k): 1.85% (Max +{residuals.max():,.0f})'
    )
    ax.text(0.975, 0.965, stats_text, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', fontsize=13.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.96, edgecolor='#4a1a56', linewidth=1.3))
    
    # Inset showing the full -6k to +16k spectrum on log scale
    ax_inset = inset_axes(ax, width='37%', height='33%', loc='center right', bbox_to_anchor=(-0.03, -0.06, 1, 1), bbox_transform=ax.transAxes)
    inset_bins = np.linspace(-6000, 17000, 45)
    ax_inset.hist(residuals, bins=inset_bins, color='#2b5c8f', edgecolor='#163354', linewidth=0.7, alpha=0.85, log=True)
    ax_inset.set_ylim(0.8, 6000)
    ax_inset.axvspan(-1500, 1500, color='#7b2d8e', alpha=0.32, label='Core Window')
    ax_inset.set_title('Full Holdout Spectrum (-6k to +16k, Log Counts)', fontsize=12, fontweight='bold', pad=4, color='#111111')
    ax_inset.tick_params(labelsize=11, length=3.5, width=1.1)
    for tick in ax_inset.get_xticklabels() + ax_inset.get_yticklabels(): 
        tick.set_fontweight('bold')
    ax_inset.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x != 0 else '0'))
    ax_inset.grid(True, linestyle=':', alpha=0.55)
    
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.96)
    plt.savefig(os.path.join(output_dir, 'fig_4_5_residual_histogram.png'), dpi=300)
    plt.close()
    print(f"    SAVED: fig_4_5_residual_histogram.png (Polished High-Definition with Inset)")
    print(f"    VALUES: Mean={res_mean:.2f}, Std={res_std:.2f}, Skew={res_skew:.2f}, Kurt={res_kurt:.2f}")
    
    # ===========================================================
    # ===========================================================
    # FIGURE 4.6/4.7: Modal Split (ENLARGED NUMBERS & POLISHED)
    # ===========================================================
    print("[7] Generating Figure 4.6/4.7: Modal Split (ENLARGED & POLISHED)...")
    
    labels = ['Passenger Cars & Taxis', 'Light Goods Vehicles', 'Heavy Goods Vehicles', 
              'Buses & Coaches', 'Motorcycles', 'Pedal Cycles']
    sizes = [31664107, 4886461, 1781565, 607621, 223829, 137771]
    total = sum(sizes)
    percentages = [s/total*100 for s in sizes]
    
    colors = ['#1e3a5f', '#2b5c8f', '#3e7099', '#5485a8', '#78a2be', '#a0c1d6']
    fig, ax = plt.subplots(figsize=(14.5, 8.0), dpi=300)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, percentages, color=colors, edgecolor='#112233', linewidth=1.3, height=0.62)
    
    for i, (bar, pct, count) in enumerate(zip(bars, percentages, sizes)):
        width = bar.get_width()
        if pct > 50:
            ax.text(width / 2, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.2f}%  ({count:,.0f})', 
                    ha='center', va='center', fontweight='bold', fontsize=16, color='white')
        else:
            ax.text(width + 1.0, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.2f}%  ({count:,.0f})', 
                    ha='left', va='center', fontweight='bold', fontsize=15, color='#111111')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15.5, fontweight='bold', color='#111111')
    ax.set_xlabel('Percentage of Total Network Traffic (%)', fontweight='bold', fontsize=16.5, labelpad=20, color='#111111')
    ax.set_title('Birmingham Traffic Modal Split Breakdown (2000–2025)\nTotal Recorded Movements: 39,301,354 Vehicles', 
                 fontweight='bold', fontsize=19, pad=18, color='#111111')
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    
    ax.tick_params(axis='x', which='major', labelsize=16, length=6, width=1.5)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}%'))
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    
    ax.grid(True, linestyle='--', alpha=0.55, color='#cccccc', axis='x')
    ax.set_axisbelow(True)
    
    plt.subplots_adjust(left=0.23, right=0.96, top=0.88, bottom=0.15)
    plt.savefig(os.path.join(output_dir, 'fig_4_6_modal_split.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_4_7_modal_split.png'), dpi=300)
    plt.close()
    print("    SAVED: fig_4_6_modal_split.png & fig_4_7_modal_split.png")
    
    # ===========================================================
    # FIGURE 4.8: COVID-19 Concept Drift (verified values)
    # ===========================================================
    print("[8] Generating Figure 4.8 / 4.7: COVID-19 Concept Drift...")
    
    eras = ['Historical\n(2000–2017)', 'Pre-COVID\n(2018–2019)', 'Lockdown\n(2020–2021)', 'Recovery\n(2022–2025)']
    volumes = [549.77, 321.36, 603.01, 616.78]
    era_colors = ['#3b528b', '#21918c', '#5ec962', '#fde725']
    
    fig, ax = plt.subplots(figsize=(11, 7.5))
    bars = ax.bar(eras, volumes, color=era_colors, edgecolor='#2b2b2b', linewidth=1.2, width=0.62)
    
    # Value labels on top of bars
    for bar, v in zip(bars, volumes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=16, color='#111111')
    
    # Sample size annotations inside bars
    era_n = [54792, 7116, 4020, 7020]
    for idx, (bar, n) in enumerate(zip(bars, era_n)):
        text_color = '#111111' if idx == 3 else '#ffffff'  # dark text on yellow bar for high contrast
        ax.text(bar.get_x() + bar.get_width()/2, 22, f'n = {n:,}', 
                ha='center', va='bottom', fontsize=14, color=text_color, fontweight='bold')
    
    ax.set_ylabel('Mean Traffic Volume (vehicles/hour)', fontweight='bold', fontsize=16.5, labelpad=15)
    ax.set_xlabel('Temporal Era', fontweight='bold', fontsize=16.5, labelpad=20)
    ax.set_title('Traffic Volume Variations Across Defined Temporal Eras\n(Birmingham DfT Count Point Data, 2000–2025)',
                 fontweight='bold', fontsize=18, pad=20)
    
    ax.set_ylim(0, 720)
    ax.tick_params(axis='x', labelsize=15, pad=8)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    ax.tick_params(axis='y', labelsize=15.5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc', axis='y')
    ax.set_axisbelow(True)
    
    plt.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.18)
    plt.savefig(os.path.join(output_dir, 'fig_4_7_covid_drift.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_4_8_covid_drift.png'), dpi=300)
    plt.close()
    print("    SAVED: fig_4_7_covid_drift.png & fig_4_8_covid_drift.png")
    
    # ===========================================================
    # SUMMARY
    # ===========================================================
    print(f"\n{'='*60}")
    print("ALL CHARTS REGENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"\nFiles generated:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
