# Rawalpindi AQI Prediction System
## 🌐 Live Demo

🚀 **Check out the live dashboard here:**  
🔗 [Rawalpindi AQI Forecast (Streamlit App)](https://itsas3-rawalpinidi-aqi-predictor-aqi-dashboard-clean-7k2ygl.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://itsas3-rawalpinidi-aqi-predictor-aqi-dashboard-clean-7k2ygl.streamlit.app/)

## Overview

This project predicts the Air Quality Index (AQI) for Rawalpindi for the next 3 days using a fully automated, serverless-friendly, machine learning pipeline.

---

## Pipeline Steps

1. **Data Collection**
    - Source: OpenWeatherMap Air Pollution API
    - Real-time and historical AQI & pollutant data for Rawalpindi

2. **Data Processing & Feature Engineering**
    - Time features, pollutant lags, rolling averages, ratios
    - Data cleaning and handling of missing values

3. **Model Training & Evaluation**
    - Compared: Random Forest, Ridge Regression, XGBoost
    - Chosen: Ridge Regression (highest R², lowest MAE/RMSE)
    - Feature importance and performance metrics reported

4. **3-Day Rolling Forecast**
    - Uses latest feature row for recursive, step-by-step prediction
    - Outputs hourly AQI forecasts for 72 hours

5. **Interactive Dashboard**
    - Built in Streamlit
    - Displays rolling AQI forecast, alerts for hazardous AQI
    - Easy to extend with interpretability and user input

---

## Results

- **Best Model:** Ridge Regression
    - R²: 0.99999, RMSE: 0.00141, MAE: 0.00071 (on test set)
- **3-Day Forecast:** Very stable, accurate predictions
- **Dashboard:** Real-time, user-friendly, code available

---

## How to Run

1. Clone repo, install requirements (`pip install -r requirements.txt`)
2. Download and update `processed_aqi_features.csv` with latest data
3. Run dashboard: `streamlit run aqi_dashboard.py`
4. (Optional) Deploy online via Streamlit Cloud

---

## Further Improvements

- Automate data refresh & retraining via serverless functions
- Add SHAP/LIME for model interpretability
- Collect and plot actual vs. predicted AQI over time
- Alert system for hazardous air days

---

## Contact

Project by Amina Saeed. For questions, contact [aminasaeed080@email.com].
