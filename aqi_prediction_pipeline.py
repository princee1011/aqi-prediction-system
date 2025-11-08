import os
import json
import joblib
import numpy as np
import requests
from datetime import datetime
from tensorflow.keras.models import load_model

from data_loading import get_city_pollutant_data
from aqi_calculator import calculate_aqi_from_pollutants, calculate_daily_aqi_from_predictions
from config import WAQI_TOKEN, INDIAN_CITIES, SEQ_LENGTH, POLLUTANTS


# ============================================================
# Robust Real-Time Data Fetch (with fallback station search)
# ============================================================
def get_real_time_air_quality(city_name):

    def fetch_feed(path):
        url = f"https://api.waqi.info/feed/{path}/?token={WAQI_TOKEN}"
        return requests.get(url, timeout=12).json()

    def search_uid(keyword):
        url = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={keyword}"
        r = requests.get(url, timeout=12).json()
        if r.get("status") == "ok":
            for item in r.get("data", []):
                uid = item.get("uid")
                if uid:
                    return uid
        return None

    city_code = INDIAN_CITIES.get(city_name)
    if not city_code:
        return None

    data = fetch_feed(city_code)

    # Fallback if station is unknown
    if data.get("status") != "ok" and "Unknown station" in str(data.get("data", "")):
        uid = search_uid(city_code)
        if uid:
            data = fetch_feed(f"@{uid}")

    if data.get("status") != "ok":
        return None

    iaqi = data["data"].get("iaqi", {})

    pollutant_values = {
        'pm25': iaqi.get('pm25', {}).get('v', None),
        'pm10': iaqi.get('pm10', {}).get('v', None),
        'o3': iaqi.get('o3', {}).get('v', None),
        'no2': iaqi.get('no2', {}).get('v', None),
        'so2': iaqi.get('so2', {}).get('v', None)
    }

    aqi = calculate_aqi_from_pollutants(pollutant_values)
    station_name = data["data"].get("city", {}).get("name", city_name)

    return {
        "city": city_name,
        **pollutant_values,
        "aqi": aqi['aqi'],
        "category": aqi['category'],
        "color": aqi['color'],
        "emoji": aqi['emoji'],
        "dominant_pollutant": aqi['dominant_pollutant'],
        "sub_indices": aqi['sub_indices'],
        "health_implications": aqi['health_implications'],
        "precautionary_actions": aqi['precautionary_actions'],
        "station": station_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# Model Loader
# ============================================================
def load_model_and_scaler(city, pollutant):
    model = load_model(f"models/{city}/{pollutant}_model.h5", compile=False)
    scaler = joblib.load(f"models/{city}/{pollutant}_scaler.pkl")
    return model, scaler


# ============================================================
# Explainability Helper
# ============================================================
def build_explanation(pollutant, current_value, first_pred):
    current = current_value or 0
    change = first_pred - current
    change_pct = round((change / current * 100), 1) if current > 0 else 0

    if abs(change_pct) < 5:
        trend = "stable"
        factors = ["Trend is mostly unchanged.", "Recent historical signal is smooth."]
    elif change > 0:
        trend = "increasing"
        factors = ["Expected upward movement", "Recent values show rising pattern"]
    else:
        trend = "decreasing"
        factors = ["Expected decline", "Recent values show downward pattern"]

    return {
        "pollutant": pollutant.upper(),
        "current": round(current, 2),
        "predicted": round(first_pred, 2),
        "change": round(change, 2),
        "change_percent": change_pct,
        "trend": trend,
        "factors": factors,
    }


# ============================================================
# Pollutant Forecast
# ============================================================
def predict_pollutants(city, pollutant, days=7):
    df = get_city_pollutant_data(city, pollutant)
    if df.empty:
        return []

    df = df.set_index("date").resample("D").mean().interpolate()
    values = df[pollutant].values[-SEQ_LENGTH:].reshape(-1, 1)

    model, scaler = load_model_and_scaler(city, pollutant)
    scaled = scaler.transform(values).flatten().tolist()

    preds = []
    for _ in range(days):
        seq = np.array(scaled[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 1)
        pred_scaled = model.predict(seq, verbose=0)[0][0]
        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
        preds.append(max(pred, 0))
        scaled.append(pred_scaled)

    return preds


# ============================================================
# Full Multi-Pollutant + AQI Prediction Pipeline
# ============================================================
def predict_all_pollutants(city, days=7, pollutants=None):
    pollutants = pollutants or POLLUTANTS

    live = get_real_time_air_quality(city)
    if not live:
        return None

    final = {"city": city, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "current_data": live, "predictions": {}}

    pred_for_aqi = {}

    for p in pollutants:
        preds = predict_pollutants(city, p, days)
        pred_for_aqi[p] = preds

        explanation = build_explanation(p, live.get(p), preds[0] if preds else 0)

        final["predictions"][p] = {
            "predictions": preds,
            "explanation": explanation
        }

    final["daily_aqi"] = calculate_daily_aqi_from_predictions(pred_for_aqi)

    return final


# ============================================================
# API Test
# ============================================================
def test_api_connection():
    return get_real_time_air_quality(list(INDIAN_CITIES.keys())[0]) is not None


# ============================================================
# Run Test 
# ============================================================
if __name__ == "__main__":
    for city in INDIAN_CITIES.keys():
        print(f"\n🌍 {city}")
        print(json.dumps(predict_all_pollutants(city, days=3), indent=2))