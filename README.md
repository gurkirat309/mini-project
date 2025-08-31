# SkyGuard: AI System for City-Wise Air Quality Forecasting

## Overview
SkyGuard is a web-based AI system that forecasts next-day Air Quality Index (AQI) for multiple cities using real-time data from the OpenWeatherMap Air Pollution API and a machine learning model trained on historical data.

## Features
- Fetches current AQI and pollutant data for major cities
- Predicts next-day AQI using a trained RandomForestRegressor
- Displays AQI categories and health advisories
- Visualizes 7-day AQI trends

## Project Structure
```
project/
├── app.py
├── models/
│   ├── aqi_model.joblib
│   ├── scaler.joblib
├── data/
│   ├── city_day.csv
├── templates/
│   ├── index.html
│   ├── predict.html
├── static/
│   ├── style.css
│   ├── script.js
├── notebooks/
│   ├── train_and_eval.ipynb
├── README.md
├── .env
```

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib requests joblib python-dotenv
   ```
3. **Get OpenWeatherMap API Key**
   - Sign up at https://openweathermap.org/api
   - Add your API key to a `.env` file:
     ```
     OWM_API_KEY=your_api_key_here
     ```
4. **Prepare Data**
   - Place `city_day.csv` in the `data/` directory (already included if using provided files).
5. **Train the Model**
   - Open `notebooks/train_and_eval.ipynb` and follow the steps to preprocess data, train, evaluate, and save the model and scaler to `models/`.
6. **Run the Flask App**
   ```bash
   python app.py
   ```
   - Visit `http://127.0.0.1:5000/` in your browser.

## AQI Categories & Advisories
| AQI Range | Category                          | Advisory                                    |
|-----------|-----------------------------------|---------------------------------------------|
| 0–50      | Good                              | Air quality is satisfactory.                |
| 51–100    | Moderate                          | Acceptable, but some pollutants may affect sensitive individuals. |
| 101–150   | Unhealthy for Sensitive Groups    | Sensitive groups should reduce outdoor exertion. |
| 151–200   | Unhealthy                         | Everyone may begin to experience health effects. |
| 201–300   | Very Unhealthy                    | Health warnings of emergency conditions.    |
| 301+      | Hazardous                         | Serious health effects. Avoid all outdoor activity. |

## Notes
- The app uses live API data for inference and historical data for training.
- All code is modular and ready to run.

---

**SkyGuard: Breathe Easy, Stay Informed!** 