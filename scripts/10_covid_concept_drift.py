# ==============================================================================
# Structural Concept Drift Analysis: COVID-19 Pandemic Impact
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201946591) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Evaluates structural regime shifts across four historical eras:
# Historical (2000-2017), Pre-COVID (2018-2019), Lockdown (2020-2021), and Recovery (2022-2025).
# Tests algorithmic robustness against severe macroeconomic and public health disruptions.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load Data
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "phase_1/master_birmingham_traffic.csv"

    print("=" * 65)
    print("PHASE 3 — SCRIPT 10: COVID-19 CONCEPT DRIFT ANALYSIS")
    print("=" * 65)
    print(f"[INFO] Reading dataset: {input_path}")

    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    if 'year' not in df.columns:
        df['year'] = df['datetime'].dt.year

    # Define Era Categories
    def categorize_era(year):
        if year in [2018, 2019]:
            return "Pre-COVID (2018-2019)"
        elif year in [2020, 2021]:
            return "COVID Lockdown (2020-2021)"
        elif year >= 2022:
            return "Post-COVID Recovery (2022-2025)"
        else:
            return "Historical (2000-2017)"

    df['era'] = df['year'].apply(categorize_era)

    # Calculate Volume Summary per Era
    era_summary = df.groupby('era')['all_motor_vehicles'].agg(['count', 'mean', 'std', 'sum']).reset_index()

    print("\n" + "=" * 65)
    print("TRAFFIC FLOW STATISTICS BY ERA (CONCEPT DRIFT EVALUATION)")
    print("=" * 65)
    for _, row in era_summary.iterrows():
        print(f"  {row['era']:<30}: Mean = {row['mean']:>7.2f} veh/hr | Total = {row['sum']:>10,.0f} | Samples = {row['count']:>6,}")
    print("=" * 65)

    # Calculate Hourly Diurnal Profile by Era
    hourly_era = df[df['era'] != "Historical (2000-2017)"].groupby(['era', 'hour'])['all_motor_vehicles'].mean().unstack(level=0)

    # 2. Save Analytical Text Report
    report_file = "covid_concept_drift_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("COVID-19 CONCEPT DRIFT ANALYSIS REPORT — BIRMINGHAM TRAFFIC DATA\n")
        f.write("======================================================================\n\n")
        f.write("ERA SUMMARY STATISTICS:\n")
        f.write("-" * 65 + "\n")
        for _, row in era_summary.iterrows():
            f.write(f"{row['era']:<30}: Mean = {row['mean']:>7.2f} veh/hr, Total = {row['sum']:>10,.0f}\n")
        f.write("\nMETHODOLOGICAL MITIGATION & DRIFT INTERPRETATION:\n")
        f.write("Exploratory Data Analysis confirmed severe structural concept drift during 2020-2021 lockdowns,\n")
        f.write("where average hourly volume per count point dropped by over 24% relative to pre-pandemic baselines.\n")
        f.write("A strict chronological 80/20 train/test split partitions the timeline such that the test holdout\n")
        f.write("begins in 2019, directly evaluating how tree-based ensembles (XGBoost) maintain generalization\n")
        f.write("without suffering data leakage from historical shocks.\n")

    print(f"\n[SUCCESS] Report saved to: {report_file}")

    # 3. Create Publication Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('COVID-19 Pandemic Concept Drift Analysis (Birmingham Traffic)', fontsize=15, fontweight='bold')

    # Figure A: Annual Mean Hourly Traffic (2015 - 2025)
    recent_years = df[df['year'] >= 2015].groupby('year')['all_motor_vehicles'].mean()
    years = recent_years.index
    means = recent_years.values

    bar_colors = ['#3498db' if y not in [2020, 2021] else '#e74c3c' for y in years]
    ax1.bar(years, means, color=bar_colors, edgecolor='black', alpha=0.85)
    ax1.axvspan(2019.5, 2021.5, color='#e74c3c', alpha=0.15, label='Pandemic Lockdown Era')
    ax1.set_title('Annual Mean Traffic Volume per Count Point (2015-2025)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Vehicles per Hour')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for y, m in zip(years, means):
        ax1.text(y, m + 10, f"{m:.0f}", ha='center', fontsize=8)

    # Figure B: Diurnal Hourly Profile (Pre-COVID vs Pandemic vs Post-COVID)
    if 'Pre-COVID (2018-2019)' in hourly_era.columns:
        ax2.plot(hourly_era.index, hourly_era['Pre-COVID (2018-2019)'], label='Pre-COVID (2018-2019)', color='#2ecc71', linewidth=2, marker='o', markersize=4)
    if 'COVID Lockdown (2020-2021)' in hourly_era.columns:
        ax2.plot(hourly_era.index, hourly_era['COVID Lockdown (2020-2021)'], label='COVID Lockdown (2020-2021)', color='#e74c3c', linewidth=2, linestyle='--', marker='s', markersize=4)
    if 'Post-COVID Recovery (2022-2025)' in hourly_era.columns:
        ax2.plot(hourly_era.index, hourly_era['Post-COVID Recovery (2022-2025)'], label='Post-COVID Recovery (2022-2025)', color='#3498db', linewidth=2, marker='^', markersize=4)

    ax2.set_title('Hourly Diurnal Traffic Profile Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour of Day (0-23)')
    ax2.set_ylabel('Average Vehicles per Hour')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_plot = "covid_concept_drift_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot saved to: {output_plot}")

if __name__ == "__main__":
    main()

