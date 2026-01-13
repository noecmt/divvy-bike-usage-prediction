"""
Generate US Federal Holidays CSV for 2024-2025
Used for Divvy bike usage prediction feature engineering
"""

import pandas as pd
import holidays

# Generate US federal holidays for 2024 and 2025
us_holidays = holidays.US(years=[2024, 2025])

# Convert to DataFrame
holiday_data = []
for date, name in sorted(us_holidays.items()):
    holiday_data.append({
        'date': date,
        'holiday_name': name,
        'type': 'federal'
    })

df = pd.DataFrame(holiday_data)

# Save to CSV
output_path = 'raw/holidays/us_holidays_2024_2025.csv'
df.to_csv(output_path, index=False)

print(f"✅ Holidays file created: {output_path}")
print(f"📊 Total holidays: {len(df)}")
print("\nPreview:")
print(df.head(10))
print("\n...")
print(df.tail(5))
