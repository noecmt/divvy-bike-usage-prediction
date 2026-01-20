# Divvy Bike Usage Prediction

Predict Chicago bike-sharing usage patterns using weather and temporal data.

## 📊 Data Setup

**⚠️ Important**: Data files are **NOT included** in the Git repository (excluded via `.gitignore`).

### Docker (Automatic)
When using Docker, data is **automatically downloaded** on first container startup:
- ✅ Divvy trip data (2024-2025) - Downloaded from official S3 bucket
- ✅ US holidays data - Auto-generated
- ⚠️ Weather data - Must be provided manually in `data/raw/weather/`

### Local Development (Manual)
```bash
python scripts/download_divvy_data.py      # Downloads ~2GB of trip data
python scripts/generate_holidays.py         # Generates holidays CSV
```

### Weather Data (Required)
Obtain historical weather data and place in `data/raw/weather/`:
- `2024_weather_chicago.csv`
- `2025_weather_chicago.csv`

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Prerequisites**: Docker Desktop installed and running

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **First run**: Container will automatically download ~2GB of Divvy trip data (10-20 minutes). Subsequent runs are instant.

3. **Access the applications**
   - Streamlit Dashboard: http://localhost:8501
   - Jupyter Lab: http://localhost:8888

4. **Stop containers**
```bash
docker-compose down
```

**Benefits**:
- ✅ No data in Git repository
- ✅ Automatic data download on first run
- ✅ Data persisted in Docker volumes
- ✅ Fully portable - clone and run anywhere

### Option 2: Local Development

1. **Setup environment**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
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

4. **Launch dashboard**
```bash
streamlit run app/streamlit_app.py
```

## 🐳 Docker Configuration

### Build Options

**Build specific service:**
```bash
docker-compose build divvy-app
```

**Rebuild without cache:**
```bash
docker-compose build --no-cache
```

**Run only dashboard:**
```bash
docker-compose up divvy-app
```

**Run only Jupyter:**
```bash
docker-compose up jupyter
```

### Volume Management

Data and models are mounted as volumes for persistence:
- `./data` → Container `/app/data` (data downloaded into this volume)
- `./models` → Container `/app/models`
- `./notebooks` → Container `/app/notebooks`

### Clean Restart (Removes All Data)
```bash
docker-compose down -v  # Deletes volumes
docker-compose up -d    # Rebuilds and re-downloads data
```

## 🎯 Project Goals

- Train ML models on 2024 data
- Validate predictions on 2025 data
- Compare Linear Regression, Random Forest, and XGBoost
- Identify key factors driving bike usage

## 📦 Data Sources

- **Divvy Trip Data**: https://divvybikes.com/system-data
- **Weather Data**: Meteostat
- **Holidays**: US Federal holidays

## 🛠️ Tech Stack

Python 3.12+ • pandas • numpy • scikit-learn • xgboost • matplotlib • seaborn • plotly • jupyter • streamlit • Docker

## 📁 Project Structure

```
divvy-bike-usage-prediction/
├── app/                    # Streamlit dashboard
├── data/                   # Raw and processed data
├── models/                 # Trained ML models
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
└── requirements.txt        # Python dependencies
```
