"""
Download Divvy bike trip data for 2024 and 2025
"""

import os
import urllib.request
from pathlib import Path

BASE_URL = "https://divvy-tripdata.s3.amazonaws.com"
YEARS = [2024, 2025]
DATA_DIR = Path("data/raw/divvy")

def download_file(url, dest_path):
    """Download file with progress"""
    print(f"Downloading {dest_path.name}...", end=" ")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"✓ ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False

def main():
    print("📥 Downloading Divvy trip data...\n")
    
    for year in YEARS:
        year_dir = DATA_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Year {year}:")
        for month in range(1, 13):
            filename = f"{year}{month:02d}-divvy-tripdata.csv"
            csv_path = year_dir / filename
            
            if csv_path.exists():
                print(f"  {filename} - Already exists, skipping")
                continue
            
            zip_url = f"{BASE_URL}/{year}{month:02d}-divvy-tripdata.zip"
            zip_path = year_dir / f"{year}{month:02d}-divvy-tripdata.zip"
            
            # Download ZIP
            if download_file(zip_url, zip_path):
                # Extract CSV
                print(f"  Extracting...", end=" ")
                import zipfile
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(year_dir)
                    zip_path.unlink()  # Delete ZIP after extraction
                    print("✓")
                except Exception as e:
                    print(f"✗ ({e})")
        
        print()
    
    print("✅ Download complete!")

if __name__ == "__main__":
    main()
