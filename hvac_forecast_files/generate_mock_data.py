import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(days=90):
    print(f"Generating {days} days of realistic synthetic data...")
    
    # 1. Generate Daily Weather (daily_weather.csv)
    # Use naive dates to completely bypass Daylight Saving Time (DST) overlap crashes
    today = datetime.now().date()
    end_date = today + timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42) # For reproducible mock data
    
    # Create a gentle seasonal temperature wave
    base_temp = 20 + 5 * np.sin(np.arange(len(dates)) / 30.0) 
    
    weather_df = pd.DataFrame({
        'date': dates.date,
        'temperature_2m_min': np.round(base_temp - np.random.uniform(4, 7, len(dates)), 1),
        'temperature_2m_max': np.round(base_temp + np.random.uniform(4, 10, len(dates)), 1)
    })
    
    weather_df.to_csv('daily_weather.csv', index=False)
    print(f"✅ Created daily_weather.csv ({len(weather_df)} rows)")

    # 2. Generate HVAC History (hvac_history.csv)
    meters = ['QG AVAC 1 (E)', 'QG AVAC 2', 'QG AVAC 3']
    
    # Create 10-minute strictly UTC timestamps (UTC has no DST, guaranteeing no crashes)
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    timestamps = pd.date_range(start=start_ts, end=end_ts, freq='10min')[:-1] 
    
    hvac_records = []
    
    for i, meter in enumerate(meters):
        base_power = 10 + (i * 5)
        
        for ts in timestamps:
            # Convert to local time safely for building schedule logic
            local_ts = ts.tz_convert('Europe/Lisbon')
            minute_of_day = local_ts.hour * 60 + local_ts.minute
            
            is_active = 450 <= minute_of_day <= 1380
            
            if is_active:
                peak_curve = np.sin(np.pi * (minute_of_day - 450) / (1380 - 450))
                active_load = (base_power * 2) * peak_curve 
                power = base_power + active_load + np.random.normal(0, 1.5)
            else:
                power = base_power * 0.2 + np.random.normal(0, 0.5)
                
            power = max(0.0, power)
            
            hvac_records.append({
                'meter_name': meter,
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S%z'),
                'active_power': round(power, 2)
            })

    hvac_df = pd.DataFrame(hvac_records)
    hvac_df.to_csv('hvac_history.csv', index=False)
    print(f"✅ Created hvac_history.csv ({len(hvac_df)} rows)")
    print("\nReady! You can now run the forecasting script.")

if __name__ == "__main__":
    generate_mock_data(days=90)