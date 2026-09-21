# ==============================================================================
# Methodology Flowchart Generation (Figure 3.1)
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool - EBUS621 Dissertation Project
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

def create_methodology_flowchart():
    """Create a professional academic methodology flowchart for Chapter 3."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Colors
    phase_colors = {
        'P1': {'bg': '#e8f4fd', 'border': '#2196F3', 'header': '#1565C0'},
        'P2': {'bg': '#fff8e1', 'border': '#FF9800', 'header': '#E65100'},
        'P3': {'bg': '#e8f5e9', 'border': '#4CAF50', 'header': '#2E7D32'},
        'P4': {'bg': '#fce4ec', 'border': '#E91E63', 'header': '#AD1457'},
        'Eval': {'bg': '#f3e5f5', 'border': '#9C27B0', 'header': '#6A1B9A'},
    }
    
    box_text_size = 11
    header_text_size = 12.5
    
    # ===== PHASE 1: Data Engineering =====
    y_p1 = 19.0
    # Phase box
    phase_box = FancyBboxPatch((1.0, y_p1 - 0.3), 12.0, 2.6, 
                                boxstyle="round,pad=0.15", 
                                facecolor=phase_colors['P1']['bg'], 
                                edgecolor=phase_colors['P1']['border'], linewidth=2.0)
    ax.add_patch(phase_box)
    ax.text(7.0, y_p1 + 2.0, 'Phase 1: Data Engineering', fontsize=header_text_size, 
            fontweight='bold', ha='center', va='center', color=phase_colors['P1']['header'])
    
    # Box A: DfT Raw Daytime Counts
    box_a = FancyBboxPatch((1.8, y_p1 + 0.2), 3.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P1']['border'], linewidth=1.5)
    ax.add_patch(box_a)
    ax.text(3.55, y_p1 + 0.7, 'DfT Raw Daytime\nCounts (72,948 rows)', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # Arrow A -> C
    ax.annotate('', xy=(5.8, y_p1 + 0.7), xytext=(5.3, y_p1 + 0.7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    
    # Box C: Trigonometric Time Encoding
    box_c = FancyBboxPatch((5.8, y_p1 + 0.2), 3.2, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P1']['border'], linewidth=1.5)
    ax.add_patch(box_c)
    ax.text(7.4, y_p1 + 0.7, 'Trigonometric\nTime Encoding', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # Arrow C -> D
    ax.annotate('', xy=(9.5, y_p1 + 0.7), xytext=(9.0, y_p1 + 0.7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    
    # Box D: Lag Feature Extraction
    box_d = FancyBboxPatch((9.5, y_p1 + 0.2), 3.2, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P1']['border'], linewidth=1.5)
    ax.add_patch(box_d)
    ax.text(11.1, y_p1 + 0.7, 'Lag Feature\nExtraction', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # ===== Down arrow from P1 to P2 =====
    ax.annotate('', xy=(7.0, y_p1 - 0.7), xytext=(7.0, y_p1 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))
    
    # ===== PHASE 2: Methodological Benchmarking =====
    y_p2 = 15.2
    phase_box2 = FancyBboxPatch((1.0, y_p2 - 0.3), 12.0, 2.6, 
                                 boxstyle="round,pad=0.15",
                                 facecolor=phase_colors['P2']['bg'], 
                                 edgecolor=phase_colors['P2']['border'], linewidth=2.0)
    ax.add_patch(phase_box2)
    ax.text(7.0, y_p2 + 2.0, 'Phase 2: Methodological Benchmarking', fontsize=header_text_size, 
            fontweight='bold', ha='center', va='center', color=phase_colors['P2']['header'])
    
    # Box E: Linear Regression
    box_e = FancyBboxPatch((2.0, y_p2 + 0.2), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P2']['border'], linewidth=1.5)
    ax.add_patch(box_e)
    ax.text(4.25, y_p2 + 0.7, 'Train Linear Regression\n(Statistical Baseline)', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # Box F: Decision Tree & Random Forest
    box_f = FancyBboxPatch((7.5, y_p2 + 0.2), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P2']['border'], linewidth=1.5)
    ax.add_patch(box_f)
    ax.text(9.75, y_p2 + 0.7, 'Train Decision Tree\n& Random Forest', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # ===== Down arrow from P2 to P3 =====
    ax.annotate('', xy=(7.0, y_p2 - 0.7), xytext=(7.0, y_p2 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))
    
    # ===== PHASE 3: Machine Learning Core =====
    y_p3 = 11.4
    phase_box3 = FancyBboxPatch((1.0, y_p3 - 0.3), 12.0, 2.6, 
                                 boxstyle="round,pad=0.15",
                                 facecolor=phase_colors['P3']['bg'], 
                                 edgecolor=phase_colors['P3']['border'], linewidth=2.0)
    ax.add_patch(phase_box3)
    ax.text(7.0, y_p3 + 2.0, 'Phase 3: Machine Learning Core', fontsize=header_text_size, 
            fontweight='bold', ha='center', va='center', color=phase_colors['P3']['header'])
    
    # Box G: XGBoost
    box_g = FancyBboxPatch((2.0, y_p3 + 0.2), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P3']['border'], linewidth=1.5)
    ax.add_patch(box_g)
    ax.text(4.25, y_p3 + 0.7, 'Train XGBoost\nEnsemble (n=300)', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # Arrow G -> H
    ax.annotate('', xy=(7.5, y_p3 + 0.7), xytext=(6.5, y_p3 + 0.7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    
    # Box H: Hyperparameter Tuning
    box_h = FancyBboxPatch((7.5, y_p3 + 0.2), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P3']['border'], linewidth=1.5)
    ax.add_patch(box_h)
    ax.text(9.75, y_p3 + 0.7, 'Hyperparameter Tuning\n& Regularisation', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # ===== Down arrow from P3 to P4 =====
    ax.annotate('', xy=(7.0, y_p3 - 0.7), xytext=(7.0, y_p3 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))
    
    # ===== PHASE 4: Exogenous Fusion & Stratification =====
    y_p4 = 7.6
    phase_box4 = FancyBboxPatch((1.0, y_p4 - 0.3), 12.0, 2.6, 
                                 boxstyle="round,pad=0.15",
                                 facecolor=phase_colors['P4']['bg'], 
                                 edgecolor=phase_colors['P4']['border'], linewidth=2.0)
    ax.add_patch(phase_box4)
    ax.text(7.0, y_p4 + 2.0, 'Phase 4: Exogenous Fusion & Stratification', fontsize=header_text_size, 
            fontweight='bold', ha='center', va='center', color=phase_colors['P4']['header'])
    
    # Box I: Stratify by Road Type
    box_i = FancyBboxPatch((2.0, y_p4 + 0.2), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P4']['border'], linewidth=1.5)
    ax.add_patch(box_i)
    ax.text(4.25, y_p4 + 0.7, 'Stratify by Road Type:\nMajor vs Minor', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # Arrow I -> J
    ax.annotate('', xy=(7.5, y_p4 + 0.7), xytext=(6.5, y_p4 + 0.7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    
    # Box J: Integrate Weather
    box_j = FancyBboxPatch((7.5, y_p4 + 0.2), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['P4']['border'], linewidth=1.5)
    ax.add_patch(box_j)
    ax.text(9.75, y_p4 + 0.7, 'Integrate Exogenous\nWeather Variables', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # ===== Down arrow from P4 to Eval =====
    ax.annotate('', xy=(7.0, y_p4 - 0.7), xytext=(7.0, y_p4 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))
    
    # ===== EVALUATION FRAMEWORK =====
    y_eval = 3.5
    eval_box = FancyBboxPatch((1.0, y_eval - 0.3), 12.0, 3.0, 
                               boxstyle="round,pad=0.15",
                               facecolor=phase_colors['Eval']['bg'], 
                               edgecolor=phase_colors['Eval']['border'], linewidth=2.5)
    ax.add_patch(eval_box)
    ax.text(7.0, y_eval + 2.4, 'Evaluation Framework', fontsize=header_text_size, 
            fontweight='bold', ha='center', va='center', color=phase_colors['Eval']['header'])
    
    # Box K: Chronological 80/20 Holdout
    box_k = FancyBboxPatch((2.5, y_eval + 0.7), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=phase_colors['Eval']['border'], linewidth=1.5)
    ax.add_patch(box_k)
    ax.text(4.75, y_eval + 1.2, 'Chronological 80/20\nHoldout Testing', fontsize=box_text_size, 
            ha='center', va='center', fontweight='bold')
    
    # Arrow K -> L
    ax.annotate('', xy=(7.5, y_eval + 1.2), xytext=(7.0, y_eval + 1.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    
    # Box L: Evaluate via RMSE, MAE, R²
    box_l = FancyBboxPatch((7.5, y_eval + 0.7), 4.5, 1.0, boxstyle="round,pad=0.1",
                           facecolor='#f3e5f5', edgecolor=phase_colors['Eval']['border'], linewidth=2.0)
    ax.add_patch(box_l)
    ax.text(9.75, y_eval + 1.2, 'Evaluate via\nRMSE, MAE, R\u00b2', fontsize=box_text_size + 1, 
            ha='center', va='center', fontweight='bold', color=phase_colors['Eval']['header'])
    
    # Side arrows indicating all phases feed into evaluation
    # P2 -> Eval (left side)
    ax.annotate('', xy=(1.0, y_eval + 2.0), xytext=(1.0, y_p2 + 0.7),
                arrowprops=dict(arrowstyle='->', color=phase_colors['P2']['border'], lw=1.5, 
                               linestyle='dashed'))
    # P3 -> Eval (left side, slightly offset)
    ax.annotate('', xy=(0.6, y_eval + 2.0), xytext=(0.6, y_p3 + 0.7),
                arrowprops=dict(arrowstyle='->', color=phase_colors['P3']['border'], lw=1.5, 
                               linestyle='dashed'))
    
    plt.tight_layout()
    
    output_dir = "final_academic_charts"
    if not os.path.exists(output_dir) and os.path.exists(os.path.join(os.path.dirname(__file__), "..", output_dir)):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_3_1_methodology_flowchart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved: {output_path}")

if __name__ == "__main__":
    create_methodology_flowchart()
