# solar-power-analytics

## Team

- Ali İhsan Çevik
- Furkan Kopan
- Mesut Altun
- Cem Akan
- Veysel Genç
- Yunus Emre Erten
- Zeliha İnan
- Gülse Ogultegin
- Sinem Durmaz
- Berat Erhan Şekeröz (Representative)
- Zeynep Elif Göksu
- Ebrar Kalfaoğlu
- Miray Balıkoğlu
- Arda Yenisaraç
- Kerem Nalçabasmaz

## Project Overview

Solar Power Analytics is a Python-based data analytics and visualization project developed to analyze solar power generation data from two different solar plants. The project focuses on cleaning raw generation and weather sensor data, producing meaningful visualizations, detecting abnormal production behavior, and preparing a machine learning-based power prediction module.

The main goal of this project is to support better understanding of solar plant performance by combining data preprocessing, exploratory data analysis, interactive dashboards, anomaly detection, and power forecasting in a single workflow.

 ### Key Features
 - Interactive Visualizations: Built with Plotly and Streamlit to display AC/DC power time series, hourly production heatmaps, and irradiation vs. power scatter plots.  
-  Threshold-Based Anomaly Detection: Identifies underperforming inverters by flagging daytime readings where AC power drops below a configurable threshold (default: 200 kW).  
- Comparative Analytics: Evaluates and compares daily energy production between different plants and provides a top-10 inverter comparison by energy output.  
- Environmental Impact Analysis: Analyzes the effect of ambient and module temperatures, as well as solar irradiation on power generation.

## Machine Learning Module

This project transitions from descriptive analytics to predictive modeling by implementing a dedicated machine learning pipeline to forecast future energy production.

### Implemented Models
- **Linear Regression:** Used as a baseline model to establish a linear relationship between irradiation and power output.
- **Random Forest Regressor:** An ensemble learning method applied to capture non-linear patterns and complex interactions between environmental variables.

### Feature Engineering & Inputs
To achieve high prediction accuracy, the following features are utilized:
- **Environmental Data:** Irradiation, Ambient Temperature, Module Temperature.
- **Operational Data:** DC Power (as a lead indicator for AC conversion).
- **Time-based Features:** Hourly and daily seasonal components extracted from timestamps.

### Prediction Target
- **AC Power Output:** The primary metric for evaluating the real-world efficiency and revenue of the solar plant.

### Evaluation Metrics
The models are rigorously evaluated using standard statistical error metrics to ensure reliability:
- **MAE (Mean Absolute Error):** To measure the average magnitude of errors.
- **RMSE (Root Mean Squared Error):** To penalize larger prediction outliers.
- **R² Score:** To determine how well the model explains the variance in the generation data.

## Project Structure

```
solar-power-analytics/
├── data/
│   ├── raw/                  # Raw CSV files (not tracked by git)
│   └── processed/            # Cleaned output files (not tracked by git)
├── reports/
│   └── images/               # Exported PNG charts
├── saved_models/             # Trained ML models
├── scripts/
│   ├── viz_demo.py           # Demo charts with synthetic data
│   ├── save_visuals.py       # Export charts as HTML
│   └── save_images.py        # Export charts as PNG
├── src/
│   ├── data_loader.py        # Data cleaning and preprocessing
│   └── visualization.py      # Plot functions
├── app.py                    # Streamlit dashboard entry point
├── config.py                 # Project-wide settings
├── requirements.txt          # Full environment dependencies
└── requirements-images.txt   # Additional deps for PNG export
```

## Data source

- Dataset: Solar Power Generation Data
- URL: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

### Download and place the data

**Option 1: KaggleHub (recommended)**

```zsh
python -m pip install kagglehub
python - <<'PY'
import shutil
import os
import kagglehub

path = kagglehub.dataset_download("anikannal/solar-power-generation-data")
raw_dir = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

for name in [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]:
    shutil.copy(os.path.join(path, name), os.path.join(raw_dir, name))

print("Copied files to", raw_dir)
PY
```

**Option 2: Manual download**

1) Download the zip from Kaggle.
2) Place these 4 CSV files under `data/raw/`:
   - `Plant_1_Generation_Data.csv`
   - `Plant_1_Weather_Sensor_Data.csv`
   - `Plant_2_Generation_Data.csv`
   - `Plant_2_Weather_Sensor_Data.csv`

Then generate cleaned data:

```zsh
python src/data_loader.py
```

## Visualization quick start

1) Generate cleaned data (once):
   - Run `src/data_loader.py` to create `data/processed/clean_plant0.csv`.
2) Build plots in `src/visualization.py` and call the functions from `app.py`.
3) Optional: run the demo script to verify charts without raw data.

## Demo charts (synthetic data)

Run the demo script to create HTML charts in `reports/`:

```zsh
python scripts/viz_demo.py
```

## Save all charts from cleaned data (HTML)

Create HTML reports using the real cleaned dataset:

```zsh
python scripts/save_visuals.py --plant 0
```

Files are saved to `reports/` with a timestamp. Open any HTML file in your browser:

```zsh
open reports/viz_daily_energy_plant0_YYYYMMDD_HHMMSS.html
```

## Save all charts as images (PNG)

Install the image backend once:

```zsh
python -m pip install -r requirements-images.txt
```

Then export PNG files:

```zsh
python scripts/save_images.py --plant 0
```

PNG files are saved under `reports/images/`.

## Configuration

Project settings live in `config.py`:
- Paths for data, reports, and saved models
- Night-hour filters
- `ANOMALY_THRESHOLD` used by anomaly charts and tables

## Requirements

`requirements.txt` contains pinned versions for the full environment (Streamlit, Plotly, Pandas, etc.).
If you use Python 3.13, make sure `numpy==2.4.4` (already updated) so installation succeeds.

For image export, install `requirements-images.txt` (Kaleido + Chrome requirement).

## Run the Streamlit app

Generate cleaned data, then launch the dashboard:

```zsh
python src/data_loader.py
streamlit run app.py
```

If you do not have Streamlit installed yet:

```zsh
python -m pip install -r requirements.txt
```

The app includes:
- Visualization tab (daily energy, inverter comparison, irradiation vs power, heatmap, temperature, power timeseries)
- Anomaly Detection tab (scatter + anomaly list table)
- Power Prediction tab (placeholder for ML integration)

## Notes

- Cleaned CSV files are ignored by git; everyone generates them locally.
- `requirements.txt` already includes plotly, pandas, numpy, and streamlit.
  
