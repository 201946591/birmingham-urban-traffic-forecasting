# ==============================================================================
# Granular Vehicle Classification & Modal Split Analysis
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================
# Disaggregates the Birmingham traffic panel across 6 vehicle classes:
# Cars/Taxis, Light Goods Vehicles (LGVs), Heavy Goods Vehicles (HGVs),
# Buses/Coaches, Two-Wheeled Motor Vehicles, and Pedal Cycles (Active Travel).
# Directly addresses Research Question 3 and Research Gap 4.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load Master Dataset
    input_path = "../master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "master_birmingham_traffic.csv"
    if not os.path.exists(input_path):
        input_path = "phase_1/master_birmingham_traffic.csv"

    print("=" * 60)
    print("PHASE 3 — SCRIPT 9: GRANULAR VEHICLE BREAKDOWN ANALYSIS")
    print("=" * 60)
    print(f"[INFO] Reading master data from: {input_path}")

    df = pd.read_csv(input_path)

    vehicle_cols = [
        'cars_and_taxis',
        'lgvs',
        'all_hgvs',
        'buses_and_coaches',
        'two_wheeled_motor_vehicles',
        'pedal_cycles'
    ]

    vehicle_labels = [
        'Cars & Taxis (Passenger)',
        'Light Goods (LGVs / Vans)',
        'Heavy Freight (HGVs)',
        'Buses & Coaches',
        'Motorcycles',
        'Pedal Cycles (Active Travel)'
    ]

    totals = df[vehicle_cols].sum()
    grand_total = totals.sum()
    percentages = (totals / grand_total) * 100

    print("\n" + "=" * 60)
    print("BIRMINGHAM TRAFFIC COMPOSITION BREAKDOWN")
    print("=" * 60)
    for col, label, total, pct in zip(vehicle_cols, vehicle_labels, totals, percentages):
        print(f"  {label:<30}: {total:>12,.0f} counts  ({pct:>5.2f}%)")
    print("-" * 60)
    print(f"  {'TOTAL MOTOR & ACTIVE VEHICLES':<30}: {grand_total:>12,.0f} counts  (100.00%)")
    print("=" * 60)

    # 2. Save Vehicle Breakdown Report
    report_file = "granular_vehicle_breakdown.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("GRANULAR VEHICLE BREAKDOWN ANALYSIS REPORT — BIRMINGHAM DfT DATA\n")
        f.write("======================================================================\n\n")
        for label, total, pct in zip(vehicle_labels, totals, percentages):
            f.write(f"{label:<30}: {total:>12,.0f} ({pct:>5.2f}%)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Recorded Vehicle Movement: {grand_total:,.0f}\n")

    print(f"\n[SUCCESS] Report saved to: {report_file}")

    # 3. Create Publication Pie Chart & Bar Chart
    colors = ['#3498db', '#e67e22', '#e74c3c', '#9b59b6', '#34495e', '#2ecc71']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Birmingham Granular Traffic Classification & Active Travel Share', fontsize=18, fontweight='bold')

    # Pie Chart
    # Hide percentages for tiny slices (<2%) on the pie itself to prevent visual overlap.
    def clean_autopct(pct):
        return ('%1.1f%%' % pct) if pct > 2 else ''

    wedges, texts, autotexts = ax1.pie(
        percentages,
        labels=None, # Removed direct labels to completely stop any overlap
        autopct=clean_autopct,
        startangle=140,
        colors=colors,
        pctdistance=0.75,
        textprops=dict(color="black", fontsize=12, fontweight='bold')
    )
    ax1.set_title('Vehicle Composition Distribution (%)', fontsize=14, fontweight='bold')
    
    # Explicit Legend Box to show ALL exact percentages clearly
    legend_labels = [f"{l} ({p:.2f}%)" for l, p in zip(vehicle_labels, percentages)]
    ax1.legend(wedges, legend_labels, title="Vehicle Categories", loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=12, title_fontsize=13)

    # Horizontal Bar Chart for Log Scale Visibility
    y_pos = np.arange(len(vehicle_labels))
    ax2.barh(y_pos, totals.values, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(vehicle_labels, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel('Total Recorded Count (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_title('Total Volume by Vehicle Category (Logarithmic)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=12)

    for i, v in enumerate(totals.values):
        ax2.text(v * 1.15, i, f"{v:,.0f}", va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_plot = "granular_vehicle_breakdown_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Plot saved to: {output_plot}")

if __name__ == "__main__":
    main()
