# ==============================================================
# 🌧️ Flood Risk Predictor - Real Data Trainer
# Author: Kanav Chhabra
# --------------------------------------------------------------
# This script automatically downloads Mumbai's real weather data
# from the Open-Meteo API, processes it into daily features,
# builds a Random Forest model, and saves flood_model.pkl
# ==============================================================

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# --------------------------------------------------------------
# 📍 CONFIGURATION
# --------------------------------------------------------------
CITY_NAME = "Mumbai"
LAT, LON = 19.075984, 72.877656  # Mumbai coordinates
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WEATHER_CSV = os.path.join(DATA_DIR, "mumbai_hourly_weather.csv")
MODEL_FILE = "flood_model.pkl"

# --------------------------------------------------------------
# ☁️ STEP 1: Download real hourly weather data
# --------------------------------------------------------------
def fetch_weather_data(lat, lon, start_date, end_date, file_path):
    print(f"📡 Fetching real weather data for {CITY_NAME} ({start_date} → {end_date})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation",
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature_2m": data["hourly"]["temperature_2m"],
        "relativehumidity_2m": data["hourly"]["relativehumidity_2m"],
        "precipitation": data["hourly"]["precipitation"]
    })
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(file_path, index=False)
    print(f"✅ Data saved to {file_path}")
    return df

if not os.path.exists(WEATHER_CSV):
    weather = fetch_weather_data(LAT, LON, START_DATE, END_DATE, WEATHER_CSV)
else:
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["time"])

# --------------------------------------------------------------
# 🌤️ STEP 2: Convert hourly → daily features
# --------------------------------------------------------------
weather["date"] = weather["time"].dt.date
daily = weather.groupby("date").agg(
    Rainfall=("precipitation", "sum"),
    Temperature=("temperature_2m", "mean"),
    Humidity=("relativehumidity_2m", "mean")
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])

# --------------------------------------------------------------
# 🌱 STEP 3: Soil Moisture Proxy (if real not available)
# --------------------------------------------------------------
daily["Soil Moisture"] = (
    daily["Rainfall"].rolling(7, min_periods=1).mean()
)
# Normalize to 0–100
daily["Soil Moisture"] = 100 * (
    daily["Soil Moisture"] - daily["Soil Moisture"].min()
) / (daily["Soil Moisture"].max() - daily["Soil Moisture"].min() + 1e-9)

# --------------------------------------------------------------
# 🌊 STEP 4: Create Flood Risk Label (temporary proxy)
# --------------------------------------------------------------
# This will be replaced by real flood incident data later.
daily["Flood Risk (%)"] = (
    0.6 * (daily["Rainfall"] / (daily["Rainfall"].max() + 1e-9))
    + 0.3 * (daily["Soil Moisture"] / 100)
    + 0.1 * (daily["Humidity"] / 100)
) * 100

# --------------------------------------------------------------
# 🧠 STEP 5: Train Random Forest Model
# --------------------------------------------------------------
X = daily[["Rainfall", "Temperature", "Humidity", "Soil Moisture"]]
y = daily["Flood Risk (%)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Model Evaluation:")
print(f"   R² Score: {r2:.3f}")
print(f"   RMSE: {rmse:.3f}")

# --------------------------------------------------------------
# 💾 STEP 6: Save Model and Processed Data
# --------------------------------------------------------------
joblib.dump(model, MODEL_FILE)
daily.to_csv(os.path.join(DATA_DIR, "mumbai_daily_features.csv"), index=False)

print(f"\n✅ Model saved as '{MODEL_FILE}'")
print(f"✅ Daily dataset saved to '{DATA_DIR}/mumbai_daily_features.csv'")

# --------------------------------------------------------------
# 🧭 STEP 7: Feature Importance (Optional Insight)
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(
    x=model.feature_importances_,
    y=X.columns,
    palette="viridis"
)
plt.title("Feature Importance in Flood Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "feature_importance.png"))
plt.close()

print(f"📈 Feature importance chart saved as 'data/feature_importance.png'")

