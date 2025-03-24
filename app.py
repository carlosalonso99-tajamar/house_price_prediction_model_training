from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import os

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Housing.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

# Load and prepare data
def prepare_data():
    df = pd.read_csv(DATA_PATH)
    
    # Create new features
    df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
    df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 1)
    df['area_per_story'] = df['area'] / (df['stories'] + 1)
    
    # Convert categorical variables to numeric
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                         'airconditioning', 'prefarea']
    
    for col in categorical_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # One-hot encode furnishingstatus
    furnishing_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnishing')
    df = pd.concat([df, furnishing_dummies], axis=1)
    df = df.drop('furnishingstatus', axis=1)
    
    return df

# Train model
def train_model():
    df = prepare_data()
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler

# Load or train model
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except:
    model, scaler = train_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create input array with all features
        input_data = {
            'area': float(data['area']),
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'stories': int(data['stories']),
            'mainroad': 1 if data['mainroad'].lower() == 'yes' else 0,
            'guestroom': 1 if data['guestroom'].lower() == 'yes' else 0,
            'basement': 1 if data['basement'].lower() == 'yes' else 0,
            'hotwaterheating': 1 if data['hotwaterheating'].lower() == 'yes' else 0,
            'airconditioning': 1 if data['airconditioning'].lower() == 'yes' else 0,
            'parking': int(data['parking']),
            'prefarea': 1 if data['prefarea'].lower() == 'yes' else 0,
            'furnishing_furnished': 1 if data['furnishingstatus'].lower() == 'furnished' else 0,
            'furnishing_semi-furnished': 1 if data['furnishingstatus'].lower() == 'semi-furnished' else 0,
            'furnishing_unfurnished': 1 if data['furnishingstatus'].lower() == 'unfurnished' else 0,
            # Add new engineered features
            'area_per_bedroom': float(data['area']) / (int(data['bedrooms']) + 1),
            'bathrooms_per_bedroom': int(data['bathrooms']) / (int(data['bedrooms']) + 1),
            'area_per_story': float(data['area']) / (int(data['stories']) + 1)
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_file_exists': os.path.exists(DATA_PATH),
        'model_file_exists': os.path.exists(MODEL_PATH),
        'scaler_file_exists': os.path.exists(SCALER_PATH)
    })

if __name__ == '__main__':
    app.run(debug=True) 