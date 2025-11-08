import os
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
from config import WAQI_TOKEN, INDIAN_CITIES, SEQ_LENGTH
from aqi_calculator import calculate_aqi_from_pollutants, calculate_daily_aqi_from_predictions

# Project-wide pollutants (CO removed)
POLLUTANTS = ['pm25', 'pm10', 'o3', 'no2', 'so2']

# Project-wide cities (Bangalore removed)
PROJECT_CITIES = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata']


# ============================================================
# Fetch Real-Time Air Quality Data (CO removed)
# ============================================================
def get_real_time_air_quality(city_name):
    city_code = INDIAN_CITIES.get(city_name)
    url = f"https://api.waqi.info/feed/{city_code}/?token={WAQI_TOKEN}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data["status"] != "ok":
            return None

        iaqi = data["data"]["iaqi"]
        pollutant_values = {
            'pm25': iaqi.get('pm25', {}).get('v', None),
            'pm10': iaqi.get('pm10', {}).get('v', None),
            'o3': iaqi.get('o3', {}).get('v', None),
            'no2': iaqi.get('no2', {}).get('v', None),
            'so2': iaqi.get('so2', {}).get('v', None)
        }

        # Compute AQI + Advice
        aqi_result = calculate_aqi_from_pollutants(pollutant_values)

        return {
            "city": city_name,
            **pollutant_values,
            "aqi": aqi_result['aqi'],
            "aqi_category": aqi_result['category'],
            "aqi_color": aqi_result['color'],
            "aqi_emoji": aqi_result['emoji'],
            "dominant_pollutant": aqi_result['dominant_pollutant'],
            "sub_indices": aqi_result['sub_indices'],
            "health_implications": aqi_result['health_implications'],
            "precautionary_actions": aqi_result['precautionary_actions'],
            "station": data['data']['city']['name'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None


# ============================================================
# Load Trained LSTM Model + Scaler (per city)
# ============================================================
def load_model_and_scaler(city: str, pollutant: str):
    """Load trained model and scaler for a given city + pollutant."""
    try:
        model_path = os.path.join("models", city, f"{pollutant}_model.h5")
        scaler_path = os.path.join("models", city, f"{pollutant}_scaler.pkl")

        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"⚠️ Could not load model/scaler for {city} - {pollutant}: {e}")
        return None, None


# ============================================================
# Lightweight Explainability (textual)
# ============================================================
def explain_prediction(pollutant: str, current_value: float, predicted_value: float, trend_hint: str):
    """
    Simple, robust explanation:
    - change and change_percent compared to current_value
    - qualitative trend with short factor hints
    """
    safe_curr = current_value if (current_value is not None) else 0
    change = predicted_value - safe_curr
    change_percent = 0.0
    if safe_curr and safe_curr > 0:
        change_percent = round((change / safe_curr) * 100, 1)

    explanation = {
        'pollutant': pollutant.upper(),
        'current': round(safe_curr, 2),
        'predicted': round(predicted_value, 2),
        'change': round(change, 2),
        'change_percent': change_percent,
        'trend': trend_hint,
        'factors': []
    }

    # Heuristic messaging
    if abs(change_percent) < 5:
        explanation['trend'] = 'stable'
        explanation['factors'] = [
            f"{pollutant.upper()} levels stable versus current",
            "No sharp pattern shifts in recent window"
        ]
    elif change > 0:
        explanation['trend'] = 'increasing'
        explanation['factors'] = [
            f"Predicted increase in {pollutant.upper()}",
            "Recent lag values contribute positively"
        ]
    else:
        explanation['trend'] = 'decreasing'
        explanation['factors'] = [
            f"Predicted decrease in {pollutant.upper()}",
            "Recent lag values contribute negatively"
        ]
    return explanation


# ============================================================
#  Predict Single Pollutant (per city)
# ============================================================
def predict_single_pollutant(city: str, pollutant: str, days: int = 7):
    try:
        from data_loader import get_city_pollutant_data  # local import to avoid circulars

        # Load model/scaler
        model, scaler = load_model_and_scaler(city, pollutant)
        if model is None or scaler is None:
            return None

        # Current real-time value (for XAI comparison)
        current = get_real_time_air_quality(city)
        if not current:
            return None
        current_value = current.get(pollutant, 0) or 0

        # Historical data → daily resample → last SEQ_LENGTH
        hist = get_city_pollutant_data(city, pollutant)
        if hist is None or hist.empty:
            print(f"No historical data for {city} - {pollutant}")
            return None

        hist = hist.set_index('date').resample('D').mean().interpolate()
        if len(hist) < SEQ_LENGTH:
            print(f"Insufficient history for {city} - {pollutant}")
            return None

        values = hist[pollutant].tail(SEQ_LENGTH).values.reshape(-1, 1)
        seq_scaled = scaler.transform(values).flatten().tolist()
        seq = np.array(seq_scaled[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 1)

        preds = []
        for day in range(days):
            next_scaled = model.predict(seq, verbose=0)[0, 0]
            next_val = scaler.inverse_transform([[next_scaled]])[0, 0]
            preds.append(max(0, float(next_val)))
            # roll window
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, 0] = next_scaled

        # Basic explanation versus "tomorrow"
        first_pred = preds[0] if preds else 0
        trend_hint = 'increasing' if first_pred > current_value else 'decreasing'
        explanation = explain_prediction(pollutant, current_value, first_pred, trend_hint)

        return {
            'pollutant': pollutant,
            'current_value': current_value,
            'predictions': preds,
            'explanation': explanation
        }

    except Exception as e:
        print(f" Error predicting {city} - {pollutant}: {e}")
        return None


# ============================================================
# Predict All Pollutants + Daily AQI
# ============================================================
def predict_all_pollutants(city: str, days: int = 7, pollutants=None):
    pollutants = pollutants or POLLUTANTS
    print(f"\n{'='*60}\nPredicting {pollutants} & AQI for {city}\n{'='*60}")

    current = get_real_time_air_quality(city)
    if not current:
        print(" Cannot fetch real-time data.")
        return None

    all_preds = {}
    pred_series_for_aqi = {}

    for p in pollutants:
        print(f" {city} - {p.upper()}")
        res = predict_single_pollutant(city, p, days=days)
        if res:
            all_preds[p] = res
            pred_series_for_aqi[p] = res['predictions']
        else:
            print(f" Skip {p} (no model/data)")

    daily_aqi = calculate_daily_aqi_from_predictions(pred_series_for_aqi) if pred_series_for_aqi else []

    return {
        'city': city,
        'timestamp': datetime.now(),
        'current_data': current,
        'predictions': all_preds,
        'daily_aqi': daily_aqi
    }


# ============================================================
#  Quick API Smoke Test
# ============================================================
def test_api_connection():
    try:
        test_city = PROJECT_CITIES[0]
        code = INDIAN_CITIES.get(test_city)
        url = f"https://api.waqi.info/feed/{code}/?token={WAQI_TOKEN}"
        r = requests.get(url, timeout=10)
        data = r.json()
        ok = (data.get('status') == 'ok')
        print("API OK" if ok else f"API Error: {data.get('data')}")
        return ok
    except Exception as e:
        print(f" API test failed: {e}")
        return False
