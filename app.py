from flask import Flask, render_template, request, jsonify
import os
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Predefined list of cities with their lat/lon
CITIES = [
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639}
]

# Load trained model and scaler
try:
    model = joblib.load('models/aqi_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    city_encoder = joblib.load('models/city_encoder.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None
    city_encoder = None

# OpenWeatherMap API key
API_KEY = os.getenv("OWM_API_KEY", "YOUR_API_KEY_HERE")

def get_aqi_category(aqi):
    """Map AQI values to categories and advisories"""
    if aqi <= 50:
        return "Good", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "Acceptable, but some pollutants may affect sensitive individuals."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Sensitive groups should reduce outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health warnings of emergency conditions."
    else:
        return "Hazardous", "Serious health effects. Avoid all outdoor activity."

def fetch_aqi_from_api(lat, lon):
    """Fetch current AQI data from OpenWeatherMap API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            aqi = data['list'][0]['main']['aqi']
            components = data['list'][0]['components']
            
            # Convert AQI from 1-5 scale to 0-500 scale
            aqi_mapped = aqi * 100
            
            return aqi_mapped, components
        else:
            print(f"API Error: {response.status_code}")
            return None, None
            
    except Exception as e:
        print(f"Error fetching AQI: {e}")
        return None, None

def prepare_features_for_prediction(city_name, current_aqi, components=None):
    """Prepare features for model prediction"""
    if model is None or scaler is None:
        return None
    
    # Get current date features
    today = datetime.now()
    day_of_year = today.timetuple().tm_yday
    month = today.month
    day_of_week = today.weekday()
    
    # Encode city
    city_encoded = city_encoder.transform([city_name])[0]
    
    # Create feature vector
    features = [city_encoded, day_of_year, month, day_of_week]
    
    # Add lag features (using current AQI as lag1, estimate others)
    features.extend([current_aqi, current_aqi * 0.95, current_aqi * 0.9])  # lag1, lag2, lag3
    
    # Add rolling means (estimate based on current AQI)
    features.extend([current_aqi, current_aqi * 0.98])  # rolling_mean_3, rolling_mean_7
    
    # Add pollutant features if available
    if components:
        pollutant_mapping = {
            'pm2_5': 'PM2.5',
            'pm10': 'PM10', 
            'no2': 'NO2',
            'co': 'CO',
            'so2': 'SO2',
            'o3': 'O3'
        }
        
        for api_name, feature_name in pollutant_mapping.items():
            if api_name in components and feature_name in feature_names:
                features.append(components[api_name])
            else:
                features.append(0)  # Default value if not available
    else:
        # Add zeros for missing pollutant data
        features.extend([0] * 6)
    
    # Ensure we have the right number of features
    if len(features) != len(feature_names):
        # Pad or truncate to match expected features
        while len(features) < len(feature_names):
            features.append(0)
        features = features[:len(feature_names)]
    
    return np.array(features).reshape(1, -1)

def predict_next_day_aqi(city_name, current_aqi, components=None):
    """Predict next-day AQI for a given city"""
    if model is None:
        return None, None, None
    
    # Prepare features
    features = prepare_features_for_prediction(city_name, current_aqi, components)
    if features is None:
        return None, None, None
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Get category and advisory
    category, advisory = get_aqi_category(prediction)
    
    return prediction, category, advisory

@app.route("/")
def index():
    """Home page - display current AQI for all cities"""
    city_aqi = []
    
    for city in CITIES:
        aqi, components = fetch_aqi_from_api(city["lat"], city["lon"])
        
        if aqi is not None:
            category, advisory = get_aqi_category(aqi)
            city_aqi.append({
                "name": city["name"],
                "aqi": round(aqi, 1),
                "category": category
            })
        else:
            city_aqi.append({
                "name": city["name"],
                "aqi": "N/A",
                "category": "N/A"
            })
    
    return render_template("index.html", city_aqi=city_aqi)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction page - predict next-day AQI for selected city"""
    if request.method == "POST":
        city_name = request.form.get("city")
        
        # Find city coordinates
        city_data = next((city for city in CITIES if city["name"] == city_name), None)
        
        if city_data:
            # Fetch current AQI
            current_aqi, components = fetch_aqi_from_api(city_data["lat"], city_data["lon"])
            
            if current_aqi is not None:
                # Predict next-day AQI
                prediction, category, advisory = predict_next_day_aqi(city_name, current_aqi, components)
                
                if prediction is not None:
                    # Generate 7-day trend (simplified - using current AQI as base)
                    trend = []
                    for i in range(7):
                        trend.append({
                            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                            "aqi": round(current_aqi + (prediction - current_aqi) * (i/6), 1)
                        })
                    
                    return render_template("predict.html", 
                                         city=city_name, 
                                         current_aqi=round(current_aqi, 1),
                                         prediction=round(prediction, 1), 
                                         category=category, 
                                         advisory=advisory, 
                                         trend=trend)
                else:
                    return render_template("predict.html", 
                                         city=city_name, 
                                         error="Model prediction failed")
            else:
                return render_template("predict.html", 
                                     city=city_name, 
                                     error="Failed to fetch current AQI data")
        else:
            return render_template("predict.html", 
                                 city=None, 
                                 error="City not found")
    else:
        return render_template("predict.html", 
                             city=None, 
                             prediction=None, 
                             category=None, 
                             advisory=None, 
                             trend=None)

if __name__ == "__main__":
    app.run(debug=True) 