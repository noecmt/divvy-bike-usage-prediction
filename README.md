# Divvy Bike Usage Prediction

Predict Chicago bike-sharing usage patterns using weather and temporal data.

## 📊 Data Setup

### Automated Download
```bash
python scripts/download_divvy_data.py
```

Downloads all Divvy trip data (2024 & 2025) from the official source.

### Manual Download
See `data/DATA_ACQUISITION_GUIDE.md` for manual instructions.

## 🚀 Quick Start

1. **Setup environment**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download data**
```bash
python scripts/download_divvy_data.py
```

3. **Run analysis**
```bash
jupyter lab
```

## 🎯 Project Goals

- Train ML models on 2024 data
- Validate predictions on 2025 data
- Compare Linear Regression, Random Forest, and XGBoost
- Identify key factors driving bike usage

## 📦 Data Sources

- **Divvy Trip Data**: https://divvybikes.com/system-data
- **Weather Data**: NOAA / Open-Meteo
- **Holidays**: US Federal holidays

## 🛠️ Tech Stack

Python 3.12+ • pandas • numpy • scikit-learn • xgboost • matplotlib • seaborn • plotly • jupyter

