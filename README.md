# Urban Traffic Flow Forecasting in Birmingham: A Machine Learning and Econometric Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![R 4.2+](https://img.shields.io/badge/R-4.2%2B-blue.svg)](https://www.r-project.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Status: Complete](https://img.shields.io/badge/Status-MSc%20Dissertation%20Complete-success.svg)]()

This repository contains the empirical code, datasets, figure generation routines, and reproduction pipelines for the MSc Business Analytics and Big Data dissertation:

> **Dissertation Title**: *Urban Traffic Flow Forecasting in Birmingham: A Machine Learning and Econometric Benchmark Using Department for Transport Open Data*  
> **Author**: Pretty Jayaraj (Student ID: 201936238)  
> **Programme**: MSc Business Analytics and Big Data  
> **Institution**: University of Liverpool Management School  
> **Module**: EBUS621 – MSc Dissertation (2025/2026)  
> **Academic Supervisor**: Dr. Ehsan Khajeh  

---

## Executive Summary

Accurate short-term urban traffic forecasting is essential for intelligent transportation systems (ITS), adaptive traffic signal coordination, congestion charging enforcement, and municipal carbon abatement strategies. However, real-world municipal sensor networks are frequently constrained by discontinuous sampling schedules, sensor downtime, and non-continuous observation windows. 

This research investigates the predictive efficacy and operational trade-offs of statistical baselines versus machine learning ensembles using official Department for Transport (DfT) count records from Birmingham, UK (Local Authority 141; 72,948 hourly records spanning 2000–2025 and 39,301,354 total recorded vehicle movements).

By enforcing a strict chronological 80/20 train/test partition (evaluating holdout performance from March 2019 through 2025 across the COVID-19 pandemic shock and subsequent recovery), this study benchmarked four primary paradigms on strictly observed daytime traffic data (07:00–19:00):
1. **Linear Regression (Statistical Baseline)**: $R^2 = 0.8873$, $\text{RMSE} = 696.71$, $\text{MAE} = 245.23$
2. **Decision Tree Regressor**: $R^2 = 0.9010$, $\text{RMSE} = 652.93$, $\text{MAE} = 204.26$
3. **Random Forest Regressor**: $R^2 = 0.9136$, $\text{RMSE} = 610.21$, $\text{MAE} = 182.45$
4. **XGBoost Regressor (Recommended Architecture)**: $R^2 = 0.9103$, $\text{RMSE} = 621.68$, $\text{MAE} = 187.51$

While Random Forest marginally outperformed XGBoost by $0.0033$ in $R^2$, XGBoost demonstrated a **$3.75\times$ computational throughput advantage** during training and sub-millisecond inference latency, making it the optimal candidate for operational municipal deployment.

---

## Final Methodological Benchmark Scoreboard

Evaluated strictly on the unseen holdout partition ($N = 5,555$ daytime hours, 2019–2025) using 10 engineered past-only features:

| Model Architecture | Parameters / Depth | RMSE (veh/hr) | MAE (veh/hr) | $R^2$ Score | Operational Feasibility |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression (Baseline)** | Ordinary Least Squares | 696.71 | 245.23 | 0.8873 | Statistical Reference |
| **Decision Tree Regressor** | $\text{max\_depth}=10$ | 652.93 | 204.26 | 0.9010 | Interpretable Rules |
| **Random Forest Regressor** | 100 trees, $\text{depth}=10$ | 610.21 | 182.45 | 0.9136 | Accurate, High Training Cost |
| **XGBoost Regressor** | 300 trees, $\eta=0.03$ | 621.68 | 187.51 | 0.9103 | **Optimal ITS Trade-Off** |

### Stratified Breakdown by Functional Road Classification (XGBoost)
- **Major Arterial Corridors (A-roads & Motorways)**: $\text{RMSE} = 1,014.73$ veh/hr, $R^2 = 0.8544$ (High baseline flow, average $\sim 3,200$ veh/hr)
- **Minor Residential & Distributor Streets**: $\text{RMSE} = 163.96$ veh/hr, $R^2 = 0.8865$ (Localized feeder networks, average $\sim 450$ veh/hr)

---

## Repository Structure

```
birmingham-urban-traffic-forecasting/
├── data/                                 # Processed feature panels and exogenous datasets
│   ├── README.md                         # Data dictionary and DfT acquisition instructions
│   ├── engineered_birmingham_traffic.csv # Feature-engineered city-wide hourly dataset
│   ├── zonal_traffic.csv                 # Road-type stratified (Major vs Minor) hourly panel
│   ├── birmingham_weather.csv            # Open-Meteo ERA5 hourly temperature and precipitation
│   └── uk_holidays.csv                   # Official UK statutory bank holiday calendar
├── figures/                              # High-resolution (300 DPI) publication-grade figures
│   ├── fig_3_1_methodology_flowchart.png # Methodological architecture workflow
│   ├── fig_4_1_actual_vs_predicted.png   # XGBoost actual vs predicted holdout evaluation
│   ├── fig_4_2_stratified_predictions.png# Major vs Minor road-type holdout tracking
│   ├── fig_4_3_feature_importance.png    # Top predictor ranking (gain vs split frequency)
│   ├── fig_4_4_model_comparison.png      # Comparative scoreboard matrix
│   ├── fig_4_5_residual_histogram.png    # Diagnostic residual distribution with zoom inset
│   ├── fig_4_6_modal_split.png           # 6-class vehicle composition distribution
│   └── fig_4_7_covid_drift.png           # COVID-19 pandemic concept drift & diurnal shifts
├── models/                               # Serialized model checkpoints
│   ├── xgboost_model.json                # Fitted city-wide XGBoost regressor
│   └── phase4_xgboost_model.json         # Fitted spatiotemporal road-type XGBoost regressor
├── scripts/                              # Core empirical pipelines
│   ├── run_final_methodological_benchmark.py # Master 4-model evaluation pipeline
│   ├── generate_final_charts.py          # 300 DPI figure generation suite
│   ├── generate_flowchart.py             # Methodology flowchart routine
│   ├── phase1_preprocessing.R            # RStudio data engineering & continuity audit
│   ├── 01_feature_engineering.py         # Temporal lags and trigonometric encodings
│   ├── 02_xgboost_model_training.py      # XGBoost training and model checkpointing
│   ├── 03_model_comparison_table.py      # Comparative scoreboard table generation
│   ├── 04_feature_importance.py          # Feature gain importance extraction
│   ├── 05_future_predictor.py            # CLI inference tool with modal disaggregation
│   ├── 09_granular_vehicle_breakdown.py  # Multi-modal vehicle classification
│   ├── 10_covid_concept_drift.py         # Pandemic structural concept drift analysis
│   ├── 11_xgboost_residual_analysis.py   # Statistical normality & diagnostic tests
│   ├── exploratory_time_series/          # Phase 2 classical econometric routines
│   │   ├── 01_stl_decomposition.py       # STL decomposition (trend, season, resid)
│   │   ├── 02_stationarity_tests.py      # ADF & KPSS hypothesis testing
│   │   ├── 03_acf_pacf_analysis.py       # ACF/PACF correlograms (lags 1-40)
│   │   ├── 04_residual_diagnostics.py    # Ljung-Box portmanteau test & Q-Q plots
│   │   ├── 05_arima_forecasting.py       # Auto-ARIMA baseline with 95% CIs
│   │   └── 06_error_distribution.py      # Temporal heteroscedasticity scatter
│   ├── exploratory_deep_learning/        # Phase 3.2 exploratory PyTorch LSTM trials
│   │   ├── 06_lstm_data_preparation.py   # 3D sliding window tensor construction
│   │   ├── 07_lstm_model_training.py     # 2-layer stacked LSTM training
│   │   ├── 08_three_model_comparison.py  # ARIMA vs XGBoost vs LSTM comparison
│   │   └── 11_lstm_residual_analysis.py  # Deep learning residual diagnostics
│   └── exogenous_fusion/                 # Phase 4 environmental & spatial integration
│       ├── 01_exogenous_data_fetch.py    # Open-Meteo ERA5 API ingestion
│       ├── 02_spatial_data_prep.py       # Major vs Minor road-type stratification
│       ├── 03_exogenous_fusion.py        # Spatiotemporal table fusion
│       ├── 04_zone_xgboost.py            # Stratified XGBoost with weather features
│       └── 04b_citywide_exogenous_xgboost.py # City-wide exogenous ablation test
├── .gitignore
├── environment.yml                       # Conda environment definition
├── LICENSE                               # MIT License
├── README.md                             # Project overview and reproduction guide
└── requirements.txt                      # Pinned Python package dependencies
```

---

## Reproduction Protocol

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/birmingham-urban-traffic-forecasting.git
cd birmingham-urban-traffic-forecasting
```

### 2. Environment Setup
#### Option A: Python Virtual Environment (`venv`)
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Conda
```bash
conda env create -f environment.yml
conda activate birmingham-traffic
```

### 3. Replicate Final Methodological Benchmark
Run the single self-contained benchmark script:
```bash
py scripts/run_final_methodological_benchmark.py
```
This script ingests the raw DfT survey data, engineers the exact 10 past-only features, partitions the 80/20 chronological train/test split, trains all four models, and prints the verified benchmark scoreboard to the console and `final_benchmark_results.txt`.

### 4. Regenerate Publication Figures
To regenerate all 300 DPI figures presented in Chapter 4:
```bash
py scripts/generate_final_charts.py
py scripts/generate_flowchart.py
```

### 5. Interactive Future Traffic Inference
To run point predictions with empirical vehicle classification disaggregation:
```bash
py scripts/05_future_predictor.py
```

---

## Data Sourcing & Licensing

1. **Department for Transport (DfT) Traffic Counts**:
   - Source: UK Department for Transport Open Data Portal ([roadtraffic.dft.gov.uk](https://roadtraffic.dft.gov.uk)).
   - Local Authority: Birmingham (ID 141).
   - Licence: [Open Government Licence (OGL) v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Contains public sector information licensed under the Open Government Licence v3.0.

2. **Meteorological Reanalysis Data**:
   - Source: Open-Meteo Historical Weather API (ECMWF ERA5 reanalysis archive).
   - Coordinates: Birmingham, UK (52.4862°N, 1.8904°W).
   - Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Attribution: Zippenfenig, P. (2023).

---

## Citation

If you utilise this code or empirical benchmark in your research, please cite:

```bibtex
@mastersthesis{jayaraj2026birmingham,
  author       = {Jayaraj, Pretty},
  title        = {Urban Traffic Flow Forecasting in Birmingham: A Machine Learning and Econometric Benchmark Using Department for Transport Open Data},
  school       = {University of Liverpool Management School},
  year         = {2026},
  type         = {MSc Dissertation},
  address      = {Liverpool, United Kingdom},
  note         = {Module: EBUS621 Business Analytics and Big Data Dissertation. Supervised by Dr. Ehsan Khajeh.}
}
```

Harvard citation format (*Cite Them Right*):
> Jayaraj, P. (2026) *Urban Traffic Flow Forecasting in Birmingham: A Machine Learning and Econometric Benchmark Using Department for Transport Open Data*. MSc Dissertation. University of Liverpool Management School.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
