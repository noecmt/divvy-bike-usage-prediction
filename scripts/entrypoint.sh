#!/bin/bash
# Entrypoint script for Divvy Bike Prediction container

set -e

echo " Starting Divvy Bike Prediction container..."

# Check if data exists, if not download it
if [ ! -f "data/raw/divvy/2024/202401-divvy-tripdata.csv" ]; then
    echo " Data not found. Downloading Divvy trip data..."
    python scripts/download_divvy_data.py
else
    echo "✓ Data already exists, skipping download"
fi

# Check if weather data exists
if [ ! -f "data/raw/weather/2024_weather_chicago.csv" ]; then
    echo "⚠️  Weather data not found in data/raw/weather/"
    echo "   Please ensure weather data is available for full functionality"
fi

# Check if holidays data exists
if [ ! -f "data/raw/holidays/us_holidays_2024_2025.csv" ]; then
    echo "📅 Holidays data not found. Generating..."
    python scripts/generate_holidays.py || echo "⚠️  Could not generate holidays data"
fi

echo "✅ Initialization complete!"
echo ""

# Execute the main command (Streamlit, Jupyter, etc.)
exec "$@"
