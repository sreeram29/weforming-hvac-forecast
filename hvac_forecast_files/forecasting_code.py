import os
import math
import logging
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import holidays

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hvac-forecaster")

METERS = ['QG AVAC 1 (E)', 'QG AVAC 2', 'QG AVAC 3']
PORTUGAL_HOLIDAYS = holidays.Portugal()

LGBM_PARAMS = {
    'n_estimators': 150, 'learning_rate': 0.06, 'num_leaves': 31,
    'max_depth': 6, 'min_child_samples': 20, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1
}

# =====================================================
# PHYSICS & FEATURE ENGINEERING
# =====================================================

def generate_diurnal_weather(weather_daily_df, start_dt, end_dt):
    """Expands daily T_min / T_max into a 10-minute continuous sine curve."""
    grid = pd.DataFrame({
        'timestamp': pd.date_range(start=start_dt, end=end_dt, freq='10min', tz='UTC')
    })
    
    grid['date'] = grid['timestamp'].dt.tz_convert('Europe/Lisbon').dt.date

    weather_daily_df['date'] = pd.to_datetime(weather_daily_df['date']).dt.date
    merged = grid.merge(weather_daily_df, on='date', how='left')
    merged[['temperature_2m_min', 'temperature_2m_max']] = merged[['temperature_2m_min', 'temperature_2m_max']].ffill()

    hour_float = merged['timestamp'].dt.tz_convert('Europe/Lisbon').dt.hour + (merged['timestamp'].dt.minute / 60.0)
    t_mean = (merged['temperature_2m_max'] + merged['temperature_2m_min']) / 2.0
    t_amp = (merged['temperature_2m_max'] - merged['temperature_2m_min']) / 2.0

    merged['temp_2m'] = t_mean + t_amp * np.sin(2 * np.pi * (hour_float - 9.0) / 24.0)
    merged['relative_humidity'] = 60.0 # Default/Placeholder if granular data is missing

    return merged[['timestamp', 'temp_2m', 'relative_humidity']]

def engineer_features(df):
    """Transforms raw timestamps into cyclic matrices and thermal inertia lags."""
    df = df.sort_values('timestamp').copy()
    local_dt = df['timestamp'].dt.tz_convert('Europe/Lisbon')

    minute_of_day = local_dt.dt.hour * 60 + local_dt.dt.minute
    slot_idx = minute_of_day / 10.0
    df['time_sin'] = np.sin(2 * np.pi * slot_idx / 144.0)
    df['time_cos'] = np.cos(2 * np.pi * slot_idx / 144.0)

    df['month'] = local_dt.dt.month
    df['day_of_week'] = local_dt.dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = local_dt.dt.date.apply(lambda d: 1 if d in PORTUGAL_HOLIDAYS else 0)

    # BMS Mall Schedule (07:30 to 23:00)
    df['is_mall_active'] = ((minute_of_day >= 450) & (minute_of_day <= 1380)).astype(int)

    df['temp_roll_1h'] = df['temp_2m'].rolling(window=6, min_periods=1).mean()
    df['temp_roll_3h'] = df['temp_2m'].rolling(window=18, min_periods=1).mean()

    return df

# =====================================================
# THE 3-TIER FORECASTING CORE
# =====================================================

def execute_tier_cascade(history_df, weather_10min_df, target_date):
    """Evaluates ML validity. Cascades down to seasonal profiles or naive persistence."""
    full_df = weather_10min_df.merge(history_df, on='timestamp', how='left')
    full_df = engineer_features(full_df)

    feature_cols = [
        'time_sin', 'time_cos', 'month', 'day_of_week', 
        'is_weekend', 'is_holiday', 'is_mall_active',
        'temp_2m', 'relative_humidity', 'temp_roll_1h', 'temp_roll_3h'
    ]

    local_ts_date = full_df['timestamp'].dt.tz_convert('Europe/Lisbon').dt.date
    train_mask = local_ts_date < target_date
    target_mask = local_ts_date == target_date

    train_data = full_df[train_mask].dropna(subset=['active_power']).copy()
    target_data = full_df[target_mask].copy()

    if len(target_data) != 144:
        raise ValueError(f"Grid mismatch: Target day contains {len(target_data)} rows instead of 144.")

    # TIER 1: LightGBM Machine Learning
    if len(train_data) >= 1000: 
        val_cutoff = target_date - timedelta(days=7)
        train_local_date = train_data['timestamp'].dt.tz_convert('Europe/Lisbon').dt.date
        
        X_train = train_data[train_local_date < val_cutoff][feature_cols]
        y_train = train_data[train_local_date < val_cutoff]['active_power']
        X_val = train_data[train_local_date >= val_cutoff][feature_cols]
        y_val = train_data[train_local_date >= val_cutoff]['active_power']

        if not X_val.empty:
            model = LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)

            if val_r2 >= 0.65:
                model.fit(train_data[feature_cols], train_data['active_power'])
                preds = model.predict(target_data[feature_cols])
                p_max = train_data['active_power'].max() * 1.15
                return np.clip(preds, 0.0, p_max), "TIER_1_LIGHTGBM", round(float(val_r2), 4)

    # TIER 2: Same-Weekday Historical Median Profile
    logger.warning("Tier 1 rejected. Escalating to Tier 2 (Weekday Profile).")
    target_weekday = target_date.weekday()
    train_data['local_weekday'] = train_data['timestamp'].dt.tz_convert('Europe/Lisbon').dt.weekday
    same_weekdays = train_data[train_data['local_weekday'] == target_weekday].copy()
    
    if not same_weekdays.empty:
        same_weekdays['slot'] = same_weekdays['timestamp'].dt.tz_convert('Europe/Lisbon').dt.strftime('%H:%M')
        target_data['slot'] = target_data['timestamp'].dt.tz_convert('Europe/Lisbon').dt.strftime('%H:%M')
        profile = same_weekdays.groupby('slot')['active_power'].median().reset_index()
        merged_profile = target_data[['slot']].merge(profile, on='slot', how='left')
        preds = merged_profile['active_power'].fillna(train_data['active_power'].median()).values
        return preds, "TIER_2_WEEKDAY_MEDIAN", 0.5500

    # TIER 3: Naive Persistence (Yesterday's Data)
    logger.error("Tier 2 failed. Escalating to Tier 3 (Yesterday Persistence).")
    yesterday_date = target_date - timedelta(days=1)
    train_local_date = train_data['timestamp'].dt.tz_convert('Europe/Lisbon').dt.date
    yesterday_data = train_data[train_local_date == yesterday_date]

    if len(yesterday_data) == 144:
        return yesterday_data['active_power'].values, "TIER_3_PERSISTENCE", 0.3000

    return np.zeros(144), "FAILSAFE_ZERO", 0.0000

# =====================================================
# LOCAL EXECUTION ORCHESTRATOR
# =====================================================

def run_forecast(target_date_str=None):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date() if target_date_str else datetime.now(ZoneInfo("Europe/Lisbon")).date() + timedelta(days=1)
    logger.info(f"Target Forecast Date: {target_date}")

    # Load local CSVs instead of DB
    try:
        daily_weather = pd.read_csv("daily_weather.csv")
        hvac_history_full = pd.read_csv("hvac_history.csv")
        hvac_history_full['timestamp'] = pd.to_datetime(hvac_history_full['timestamp'], utc=True)
    except FileNotFoundError as e:
        logger.critical(f"Missing required data file: {e}")
        return

    start_history_date = pd.to_datetime(hvac_history_full['timestamp'].min()).date()
    start_dt = pd.to_datetime(start_history_date)
    end_dt = pd.to_datetime(target_date) + timedelta(days=1) - timedelta(minutes=10)
    
    weather_10min_grid = generate_diurnal_weather(daily_weather, start_dt, end_dt)
    
    output_records = []
    
    for meter in METERS:
        meter_history = hvac_history_full[hvac_history_full['meter_name'] == meter]
        preds, tier, conf = execute_tier_cascade(meter_history, weather_10min_grid, target_date)
        
        target_timestamps = pd.date_range(start=pd.to_datetime(target_date), periods=144, freq='10min', tz='Europe/Lisbon').tz_convert('UTC')
        now_str = datetime.now(ZoneInfo("UTC")).isoformat()

        for ts, p_val in zip(target_timestamps, preds):
            output_records.append({
                "meter_name": meter,
                "timestamp": ts,
                "forecasted_active_power": round(p_val, 4),
                "strategy_tier": tier,
                "confidence_score": conf,
                "generated_at": now_str
            })

    output_df = pd.DataFrame(output_records)
    
    # --- NEW RESILIENT SAVING LOGIC ---
    output_filename = "forecast_output.csv"
    try:
        output_df.to_csv(output_filename, index=False)
        logger.info(f"Forecast successfully saved to {output_filename} with {len(output_df)} records.")
    except PermissionError:
        # If the file is open in Excel, Windows locks it. Save with a timestamp instead.
        fallback_filename = f"forecast_output_{datetime.now(ZoneInfo('UTC')).strftime('%H%M%S')}.csv"
        output_df.to_csv(fallback_filename, index=False)
        logger.warning(f"File '{output_filename}' is locked (likely open in Excel). Saved as '{fallback_filename}' instead.")

if __name__ == "__main__":
    # Example execution for tomorrow
    run_forecast()