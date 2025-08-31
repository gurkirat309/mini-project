# SkyGuard Setup Guide

## 🚀 Quick Start (Demo Mode)

The app works immediately in demo mode with mock data:

```bash
python app.py
```

Visit: http://127.0.0.1:5000

## 🔑 Get Real-Time Data (Optional)

To get real-time AQI data from OpenWeatherMap:

1. **Get Free API Key:**
   - Go to: https://openweathermap.org/api
   - Sign up for a free account
   - Get your API key from the dashboard

2. **Add API Key:**
   - Create a file named `.env` in the project root
   - Add: `OWM_API_KEY=your_actual_api_key_here`

3. **Restart the app:**
   ```bash
   python app.py
   ```

## 📊 What You Get

- **Home Page:** Current AQI for 5 major Indian cities
- **Prediction Page:** Next-day AQI forecasts with health advisories
- **7-Day Trend:** Visual trend of AQI predictions
- **Categories:** Good, Moderate, Unhealthy, etc. with health advice

## 🎯 Features

✅ **Working Now:**
- ML model trained on historical data (R² = 0.963)
- Realistic mock data for demo
- Beautiful UI with responsive design
- AQI category mapping and health advisories

✅ **With API Key:**
- Real-time AQI data from OpenWeatherMap
- Live pollutant measurements
- Accurate current conditions

## 🛠️ Technical Details

- **Model:** RandomForest Regressor
- **Features:** Lag AQI, rolling means, date features, pollutants
- **Performance:** MAE: ~20, RMSE: ~21, R²: 0.963
- **Framework:** Flask + HTML/CSS/JS
- **Data:** Historical CSV + Live API

---

**Enjoy SkyGuard! 🌤️** 