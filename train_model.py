import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the AQI data"""
    print("Loading data...")
    df = pd.read_csv('data/city_day.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by city and date
    df = df.sort_values(['City', 'Date'])
    
    # Create lag features for AQI
    df['AQI_lag1'] = df.groupby('City')['AQI'].shift(1)
    df['AQI_lag2'] = df.groupby('City')['AQI'].shift(2)
    df['AQI_lag3'] = df.groupby('City')['AQI'].shift(3)
    
    # Create rolling mean features
    df['AQI_rolling_mean_3'] = df.groupby('City')['AQI'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['AQI_rolling_mean_7'] = df.groupby('City')['AQI'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    
    # Extract date features
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    # Encode city
    le = LabelEncoder()
    df['city_encoded'] = le.fit_transform(df['City'])
    
    # Select features for training
    feature_columns = [
        'city_encoded', 'day_of_year', 'month', 'day_of_week',
        'AQI_lag1', 'AQI_lag2', 'AQI_lag3',
        'AQI_rolling_mean_3', 'AQI_rolling_mean_7'
    ]
    
    # Add pollutant features if available
    pollutant_columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    for col in pollutant_columns:
        if col in df.columns:
            feature_columns.append(col)
    
    # Remove rows with missing target values
    df = df.dropna(subset=['AQI'])
    
    # Remove rows with too many missing features
    df = df.dropna(subset=feature_columns)
    
    X = df[feature_columns]
    y = df['AQI']
    
    # Save the label encoder for later use
    joblib.dump(le, 'models/city_encoder.joblib')
    
    return X, y, le

def train_model(X, y):
    """Train the RandomForest model"""
    print("Training model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")
    
    return model, scaler, X_train.columns.tolist()

def save_model_and_scaler(model, scaler, feature_names):
    """Save the trained model and scaler"""
    print("Saving model and scaler...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, 'models/aqi_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    print("Model and scaler saved successfully!")

def get_aqi_category(aqi):
    """Map AQI values to categories"""
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

if __name__ == "__main__":
    print("Starting SkyGuard Model Training...")
    
    # Load and preprocess data
    X, y, le = load_and_preprocess_data()
    
    # Train model
    model, scaler, feature_names = train_model(X, y)
    
    # Save model and scaler
    save_model_and_scaler(model, scaler, feature_names)
    
    print("Training completed successfully!") 