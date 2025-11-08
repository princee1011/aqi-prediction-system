# 🌫️ AQI Prediction System

This project predicts the Air Quality Index (AQI) for multiple Indian cities using historical pollutant data and LSTM-based time-series forecasting. The predicted pollutant values are converted into AQI scores using CPCB standards, along with category classification and health risk interpretation.
## 📂 Project Structure

NOSQL_FINAL/
│
├── app.py # Optional UI or API to run the system
├── aqi_calculator.py # CPCB AQI computation logic
├── aqi_prediction_pipeline.py # End-to-end AQI prediction pipeline
├── config.py # Configuration settings and constants
├── data_loading.py # Data preprocessing and loading functions
├── explainable_ai.py # Model explainability (SHAP / feature impact)
├── model_training.py # LSTM training script for city models

## 🧠 Workflow Overview

1. **Load Data** using `data_loading.py`
2. **Train LSTM Models** with `model_training.py`
3. **Predict Pollutant Levels** using the trained models
4. **Calculate AQI** via `aqi_calculator.py` based on CPCB breakpoints
5. **Run Full Prediction Pipeline** using `aqi_prediction_pipeline.py`
6. **(Optional)** Visualize explainability using `explainable_ai.py`

## 🏗️ Tech Stack

| Component            | Technology              |
|---------------------|-------------------------|
| Language            | Python                  |
| Machine Learning     | LSTM (Keras/TensorFlow) |
| Data Processing     | Pandas, NumPy           |
| Explainability      | SHAP                    |
| Deployment (Optional) | Flask / Streamlit     |


## 👤 Author
Princee  
If you find this useful, please ⭐ the repository!


