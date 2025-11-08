import os
import sys
import time
import numpy as np
import joblib
from datetime import datetime

# Reduce TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loading import get_city_pollutant_data 

POLLUTANTS = ['pm25', 'pm10', 'o3', 'no2', 'so2']
SEQ_LENGTH = 30



def train_model_for_pollutant(city, pollutant):
    print(f"\n [{datetime.now().strftime('%H:%M:%S')}] {city} • {pollutant.upper()} → loading data ...", flush=True)
    try:
        df = get_city_pollutant_data(city, pollutant)
    except Exception as e:
        print(f"Failed to load data for {city}-{pollutant}: {e}", flush=True)
        return

    if df is None or df.empty:
        print(f" No data for {city}-{pollutant}. Skipping.", flush=True)
        return

    df = df.set_index('date').resample('D').mean().interpolate()
    values = df[pollutant].values.reshape(-1, 1)

    if len(values) <= SEQ_LENGTH + 10:
        print(f" Not enough rows after resample for {city}-{pollutant} (have {len(values)}). Skipping.", flush=True)
        return

    scaler = RobustScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaled) - SEQ_LENGTH):
        X.append(scaled[i:i + SEQ_LENGTH])
        y.append(scaled[i + SEQ_LENGTH])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f" Train shape: {X_train.shape}, Test shape: {X_test.shape}", flush=True)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f" Training {city}-{pollutant} ...", flush=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=45,
        batch_size=32,
        verbose=1,
        callbacks=[es]
    )

    y_pred = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_arr = np.abs((y_test_inv - y_pred_inv) / np.where(y_test_inv == 0, 1, y_test_inv)) * 100
    mape = float(np.mean(mape_arr))
    r2 = float(r2_score(y_test_inv, y_pred_inv))

    print(f" {city}-{pollutant.upper()} metrics | RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}% | R²: {r2:.3f}", flush=True)

    # Save model and scaler
    out_dir = os.path.join("models", city)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{pollutant}_model.h5"))
    joblib.dump(scaler, os.path.join(out_dir, f"{pollutant}_scaler.pkl"))

    # NEW: Save metrics to CSV
    metrics_file = "training_metrics.csv"
    header = "timestamp,city,pollutant,rmse,mae,mape,r2\n"
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{city},{pollutant},{rmse},{mae},{mape},{r2}\n"

    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as f:
            f.write(header)

    with open(metrics_file, "a") as f:
        f.write(line)

    print(f"Saved model & scaler → {out_dir}", flush=True)
    print(f"Metrics logged to → {metrics_file}", flush=True)

def train_all_models(cities):
    print(f" Starting training for cities: {cities}\n", flush=True)
    start = time.time()
    for city in cities:
        print(f"\n============================\n City: {city}\n============================", flush=True)
        for pollutant in POLLUTANTS:
            try:
                train_model_for_pollutant(city, pollutant)
            except Exception as e:
                print(f"Error in {city}-{pollutant}: {e}", flush=True)
        print(f" Completed: {city}\n", flush=True)
    print(f"Total time: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    # Your four cities
    cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata']
    train_all_models(cities)
