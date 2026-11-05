# HVAC Day-Ahead Power Forecasting

This repository contains a highly resilient, automated machine learning pipeline designed to predict the power consumption of HVAC (Heating, Ventilation, and Air Conditioning) units at 10-minute intervals. 

It uses a **3-Tier Cascading Strategy** to ensure high reliability. If advanced machine learning fails or data is missing, the system gracefully falls back to historical averages or naive persistence to guarantee a forecast is always generated.

## How It Works (The Methodology)

The system looks at tomorrow's weather forecast, combines it with historical energy usage, and predicts exactly how much power the HVAC systems will draw every 10 minutes.

### Feature Engineering & Physics
* **Thermal Inertia:** Buildings don't heat up instantly. The model calculates 1-hour and 3-hour rolling temperature averages to understand how the building absorbs and retains heat.
  
* **Resilient Weather Processing:** Natively supports high-resolution weather data, but if your provider only supplies daily min/max temperatures, the pipeline mathematically synthesizes a realistic 10-minute diurnal temperature curve.
  
* **Human Context:** Automatically incorporates operating schedules (e.g., 07:30 to 23:00 active hours), weekends, and Portuguese public holidays.

### The 3-Tier Cascade
1. **Tier 1 (LightGBM):** The primary engine. Looks for complex, non-linear patterns. It self-validates on recent data; if the accuracy ($R^2$) is $\ge$ 0.65, the forecast is approved.
   
2. **Tier 2 (Historical Median):** If Tier 1 fails (e.g., erratic recent data), the system falls back to the median load profile of the same weekday (e.g., the average of all past Tuesdays).
   
3. **Tier 3 (Naive Persistence):** If Tier 2 fails (e.g., a brand new meter with no historical data), it outputs the exact power curve from the previous day.

---

## ⚙️ Setup & Installation

**Prerequisites:** 
* Python 3.9 or newer

**1. Clone the repository:**
```bash
git clone https://github.com/sreeram29/weforming-hvac-forecast.git
cd hvac_forecast_files

```

**2. Install required dependencies:**

```bash
pip install pandas numpy lightgbm scikit-learn holidays

```

---

## 🚀 How to Run the Pipeline

You can run this project either by generating a synthetic test dataset, or by bringing your own real-world data.

### Option A: Test with Mock Data (Recommended for first run)

If you want to test the pipeline immediately, you can use the included mock data generator. This creates 90 days of realistic, seasonally-adjusted HVAC usage and weather data.

**Step 1:** Generate the synthetic datasets

```bash
python generate_mock_data.py

```

*(This will create `hvac_history.csv` and `daily_weather.csv` in your folder.)*

**Step 2:** Run the forecasting engine

```bash
python forecasting_code.py

```

### Option B: Use Your Own Data

If you already have your own `hvac_history.csv` and `daily_weather.csv` files, simply place them in the root directory of the project.

**Required CSV Formats:**

* `hvac_history.csv`
* `meter_name` (String: e.g., "QG AVAC 1 (E)")
* `timestamp` (Datetime UTC: e.g., "2026-01-01 14:30:00")
* `active_power` (Float: kW load)


* `daily_weather.csv`
* `date` (Date: e.g., "2026-11-06")
* `temperature_2m_min` (Float: °C)
* `temperature_2m_max` (Float: °C)



Once your files are in place, simply execute:

```bash
python forecasting_code.py

```

---

## 📊 Expected Output

The script generates a `forecast_output.csv` file containing 432 rows (144 ten-minute intervals $\times$ 3 meters) for the target day.

| Column | Description |
| --- | --- |
| `meter_name` | The target HVAC unit. |
| `timestamp` | The 10-minute prediction window (UTC). |
| `forecasted_active_power` | Predicted kW load for that interval. |
| `strategy_tier` | Indicates which cascade tier generated the number (e.g., `TIER_1_LIGHTGBM`). |
| `confidence_score` | The $R^2$ validation score of the tier used. |
| `generated_at` | System execution timestamp. |
