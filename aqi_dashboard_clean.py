# aqi.py - Cleaned version for Streamlit deployment

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Page title
st.title("Rawalpindi AQI 3-Day Forecast Dashboard")
st.write("This dashboard generates and displays predicted AQI for the next 3 days using a trained Ridge Regression model.")

# Load trained model and processed features
df = pd.read_csv("processed_aqi_features.csv")
ridge = joblib.load("ridge_model.pkl")

# Feature columns (as used during training)
feature_cols = [
    'components.co', 'components.no', 'components.no2', 'components.o3',
    'components.so2', 'components.pm2_5', 'components.pm10', 'components.nh3',
    'hour', 'day', 'weekday', 'month', 'aqi_change', 'aqi_roll3', 'pm_ratio',
    'pm2_5_roll3', 'pm10_roll3', 'aqi_lag1', 'pm2_5_lag1'
]

# Get the latest known row for forecasting
last_known = df.iloc[[-1]].copy()

# Rolling forecast loop
future_steps = 72
forecast = []
last_row = last_known.copy()

# Initialize rolling feature history
aqi_hist = [float(last_row['aqi_lag1'].iloc[0]), float(last_row['main.aqi'].iloc[0])]
pm2_5_hist = [float(last_row['pm2_5_lag1'].iloc[0]), float(last_row['components.pm2_5'].iloc[0])]
pm10_hist = [float(last_row['components.pm10'].iloc[0]), float(last_row['components.pm10'].iloc[0])]

for step in range(future_steps):
    pred_aqi = ridge.predict(last_row[feature_cols])[0]
    forecast.append(pred_aqi)

    next_row = last_row.copy()
    next_row['aqi_lag1'] = last_row['main.aqi']
    next_row['pm2_5_lag1'] = last_row['components.pm2_5']

    aqi_hist.append(pred_aqi)
    pm2_5_hist.append(float(last_row['components.pm2_5'].iloc[0]))
    pm10_hist.append(float(last_row['components.pm10'].iloc[0]))

    next_row['aqi_roll3'] = np.mean(aqi_hist[-3:])
    next_row['pm2_5_roll3'] = np.mean(pm2_5_hist[-3:])
    next_row['pm10_roll3'] = np.mean(pm10_hist[-3:])

    next_row['main.aqi'] = pred_aqi
    next_row['components.pm2_5'] = next_row['pm2_5_roll3']
    next_row['components.pm10'] = next_row['pm10_roll3']
    next_row['aqi_change'] = pred_aqi - float(last_row['main.aqi'].iloc[0])

    next_hour = int(last_row['hour'].iloc[0]) + 1
    next_row['hour'] = next_hour % 24
    if next_row['hour'].iloc[0] == 0:
        next_row['day'] = int(last_row['day'].iloc[0]) + 1
        next_row['weekday'] = (int(last_row['weekday'].iloc[0]) + 1) % 7
    else:
        next_row['day'] = int(last_row['day'].iloc[0])
        next_row['weekday'] = int(last_row['weekday'].iloc[0])

    last_row = next_row.copy()

# Generate datetime index for forecast
future_dates = pd.date_range(last_known['datetime'].values[0], periods=future_steps+1, freq='h')[1:]
forecast_df = pd.DataFrame({'datetime': future_dates, 'predicted_aqi': forecast})

# Plot forecast
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(forecast_df['datetime'], forecast_df['predicted_aqi'], marker='o', label='Predicted AQI')
ax.set_xlabel('Datetime')
ax.set_ylabel('Predicted AQI')
ax.set_title('3-Day AQI Forecast')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Display forecast data table
st.subheader("Forecast Data")
st.dataframe(forecast_df)

# AQI alert if threshold exceeded
hazardous = forecast_df['predicted_aqi'] > 4
if hazardous.any():
    st.error("Warning: Hazardous AQI predicted in next 3 days!")
else:
    st.success("No hazardous AQI predicted in next 3 days.")
