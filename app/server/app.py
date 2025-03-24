from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import traceback
import shutil
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

app = Flask(__name__)
# Enable CORS with specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
SOURCE_MODEL_PATH = os.path.join(ROOT_DIR, 'house_price_model.keras')
MODEL_PATH = os.path.join(BASE_DIR, 'house_price_model_v3.keras')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Housing.csv')

# Input validation constants
VALID_FURNISHING_STATUS = ['furnished', 'semi-furnished', 'unfurnished']
VALID_YES_NO = ['yes', 'no']
MIN_AREA = 1
MAX_AREA = 1000000  # 1,000,000 square feet as a reasonable maximum
MIN_ROOMS = 1
MAX_ROOMS = 10
MAX_STORIES = 5
MAX_PARKING = 5

def log_transform(x):
    """Transformación logarítmica personalizada"""
    return np.log(x + 1)

def prepare_training_data():
    """Prepara los datos de entrenamiento con las mismas transformaciones"""
    try:
        # Cargar datos
        df = pd.read_csv(DATA_PATH)
        
        # Crear características derivadas
        df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
        df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 1)
        df['area_per_story'] = df['area'] / (df['stories'] + 1)
        
        return df
    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return None

def setup_preprocessor():
    """Configura el preprocesador con las mismas transformaciones usadas en el entrenamiento"""
    try:
        # Definir las columnas según su tipo
        continuous_features = ['area', 'area_per_bedroom', 'bathrooms_per_bedroom', 'area_per_story']
        discrete_features = ['bedrooms', 'bathrooms', 'stories', 'parking']
        categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                              'airconditioning', 'prefarea']
        ordinal_features = ['furnishingstatus']

        # Pipeline para variables continuas
        continuous_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', FunctionTransformer(log_transform)),
            ('scaler', RobustScaler())
        ])

        # Pipeline para variables discretas
        discrete_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # Pipeline para variables categóricas
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first'))
        ])

        # Pipeline para variables ordinales
        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(
                categories=[['unfurnished', 'semi-furnished', 'furnished']],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])

        # Combinar todos los pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('cont', continuous_transformer, continuous_features),
                ('disc', discrete_transformer, discrete_features),
                ('cat', categorical_transformer, categorical_features),
                ('ord', ordinal_transformer, ordinal_features)
            ])

        return preprocessor
    except Exception as e:
        print(f"Error setting up preprocessor: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return None

def validate_input(data):
    """Valida los datos de entrada"""
    try:
        # Verificar campos requeridos
        required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                         'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                         'parking', 'prefarea', 'furnishingstatus']
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validar valores numéricos
        if not (MIN_AREA <= float(data['area']) <= MAX_AREA):
            return False, f"Area must be between {MIN_AREA} and {MAX_AREA}"
        
        if not (MIN_ROOMS <= float(data['bedrooms']) <= MAX_ROOMS):
            return False, f"Number of bedrooms must be between {MIN_ROOMS} and {MAX_ROOMS}"
        
        if not (MIN_ROOMS <= float(data['bathrooms']) <= MAX_ROOMS):
            return False, f"Number of bathrooms must be between {MIN_ROOMS} and {MAX_ROOMS}"
        
        if not (1 <= float(data['stories']) <= MAX_STORIES):
            return False, f"Number of stories must be between 1 and {MAX_STORIES}"
        
        if not (0 <= float(data['parking']) <= MAX_PARKING):
            return False, f"Number of parking spots must be between 0 and {MAX_PARKING}"
        
        # Validar valores categóricos
        if data['mainroad'] not in VALID_YES_NO:
            return False, "mainroad must be 'yes' or 'no'"
        
        if data['guestroom'] not in VALID_YES_NO:
            return False, "guestroom must be 'yes' or 'no'"
        
        if data['basement'] not in VALID_YES_NO:
            return False, "basement must be 'yes' or 'no'"
        
        if data['hotwaterheating'] not in VALID_YES_NO:
            return False, "hotwaterheating must be 'yes' or 'no'"
        
        if data['airconditioning'] not in VALID_YES_NO:
            return False, "airconditioning must be 'yes' or 'no'"
        
        if data['prefarea'] not in VALID_YES_NO:
            return False, "prefarea must be 'yes' or 'no'"
        
        if data['furnishingstatus'] not in VALID_FURNISHING_STATUS:
            return False, f"furnishingstatus must be one of: {', '.join(VALID_FURNISHING_STATUS)}"
        
        return True, None
    except Exception as e:
        print(f"Error validating input: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return False, str(e)

def prepare_input_data(data):
    """Prepara los datos de entrada con las mismas transformaciones usadas en el entrenamiento"""
    try:
        # Convertir valores numéricos a float
        numeric_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        for field in numeric_fields:
            data[field] = float(data[field])
            print(f"Converted {field} to float: {data[field]}")
        
        # Crear un DataFrame con una sola fila
        input_df = pd.DataFrame([data])
        print("Created DataFrame with columns:", input_df.columns.tolist())
        
        # Crear características derivadas
        input_df['area_per_bedroom'] = input_df['area'] / (input_df['bedrooms'] + 1)
        input_df['bathrooms_per_bedroom'] = input_df['bathrooms'] / (input_df['bedrooms'] + 1)
        input_df['area_per_story'] = input_df['area'] / (input_df['stories'] + 1)
        
        print("Derived features:")
        print(f"area_per_bedroom: {input_df['area_per_bedroom'].values[0]}")
        print(f"bathrooms_per_bedroom: {input_df['bathrooms_per_bedroom'].values[0]}")
        print(f"area_per_story: {input_df['area_per_story'].values[0]}")
        
        return input_df
    except Exception as e:
        print(f"Error preparing input data: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise

def format_price(price_in_dollars):
    """Formatea el precio para mostrar en dólares"""
    try:
        # Formatear con 2 decimales y separadores de miles
        formatted_price = f"${price_in_dollars:,.2f}"
        
        return {
            'formatted': formatted_price,
            'in_dollars': price_in_dollars
        }
    except Exception as e:
        print(f"Error formatting price: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise

# Copy and load model
print(f"Looking for model at: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH) and os.path.exists(SOURCE_MODEL_PATH):
    print(f"Copying model from {SOURCE_MODEL_PATH} to {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    shutil.copy2(SOURCE_MODEL_PATH, MODEL_PATH)

print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")

# Load the model and create preprocessor
try:
    print("Loading model with TensorFlow version:", tf.__version__)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print(f"Model summary:")
    model.summary()
    
    # Create and fit preprocessor
    print("Loading training data...")
    training_data = prepare_training_data()
    if training_data is None:
        raise Exception("Failed to load training data")
    
    print("Setting up preprocessor...")
    preprocessor = setup_preprocessor()
    if preprocessor is None:
        raise Exception("Failed to create preprocessor")
    
    print("Fitting preprocessor with training data...")
    preprocessor.fit(training_data)
    print("Preprocessor fitted successfully!")
    
    # Crear y ajustar el scaler para el precio
    print("Setting up price scaler...")
    price_scaler = StandardScaler()
    price_scaler.fit(np.log1p(training_data['price']).values.reshape(-1, 1))
    print("Price scaler fitted successfully!")
    
except Exception as e:
    print(f"Error loading model or setting up preprocessor: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc())
    model = None
    preprocessor = None
    price_scaler = None

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Endpoint para hacer predicciones"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        print("\n=== Starting Prediction Process ===")
        
        # Get input data
        data = request.get_json()
        print("\n1. Received input data:")
        print(f"Data type: {type(data)}")
        print(f"Data keys: {list(data.keys())}")
        print(f"Data values: {data}")
        
        # Validate input
        print("\n2. Validating input...")
        is_valid, error_message = validate_input(data)
        if not is_valid:
            print(f"Validation failed: {error_message}")
            return jsonify({
                'error': error_message
            }), 400
        print("Input validation successful")
            
        # Prepare input data
        print("\n3. Preparing input data...")
        input_data = prepare_input_data(data)
        print(f"Input DataFrame shape: {input_data.shape}")
        print(f"Input DataFrame columns: {input_data.columns.tolist()}")
        print(f"Input DataFrame values:\n{input_data}")
        
        # Transform input data
        print("\n4. Transforming input data...")
        print(f"Preprocessor type: {type(preprocessor)}")
        print(f"Preprocessor fitted: {hasattr(preprocessor, 'n_features_in_')}")
        transformed_data = preprocessor.transform(input_data)
        print(f"Transformed data shape: {transformed_data.shape}")
        print(f"Transformed data type: {type(transformed_data)}")
        print(f"Transformed data:\n{transformed_data}")
        
        # Make prediction
        print("\n5. Making prediction...")
        print(f"Model type: {type(model)}")
        print(f"Model input shape: {transformed_data.shape}")
        prediction_scaled = model.predict(transformed_data)
        print(f"Raw prediction shape: {prediction_scaled.shape}")
        print(f"Raw prediction type: {type(prediction_scaled)}")
        print(f"Raw prediction:\n{prediction_scaled}")
        
        # Convert prediction to original scale
        print("\n6. Converting prediction to original scale...")
        # 1. Invertir el escalado del precio
        prediction_log = price_scaler.inverse_transform(prediction_scaled)
        print(f"After inverse scaling:\n{prediction_log}")
        
        # 2. Invertir la transformación logarítmica (log1p)
        prediction = np.expm1(prediction_log)
        print(f"After log inverse:\n{prediction}")
        
        # The prediction should now be in the correct scale
        print(f"Final prediction shape: {prediction.shape}")
        print(f"Final prediction:\n{prediction}")

        # Format price
        print("\n7. Formatting price...")
        price_in_dollars = float(prediction[0][0])  # Convert to Python float
        print(f"Price in dollars: {price_in_dollars}")

        # Format response
        print("\n8. Formatting response...")
        response = {
            'prediction': {
                'price_in_dollars': price_in_dollars,
                'price_formatted': format_price(price_in_dollars),
                'currency': 'USD'
            },
            'input_data': {
                'area': float(data['area']),
                'bedrooms': float(data['bedrooms']),
                'bathrooms': float(data['bathrooms']),
                'stories': float(data['stories']),
                'mainroad': data['mainroad'],
                'guestroom': data['guestroom'],
                'basement': data['basement'],
                'hotwaterheating': data['hotwaterheating'],
                'airconditioning': data['airconditioning'],
                'parking': float(data['parking']),
                'prefarea': data['prefarea'],
                'furnishingstatus': data['furnishingstatus']
            }
        }
        
        print("\n9. Final response:")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {list(response.keys())}")
        print(f"Response values: {response}")
        
        print("\n=== Prediction Process Completed Successfully ===")
        return jsonify(response)
        
    except Exception as e:
        print("\n=== Prediction Process Failed ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        return jsonify({
            'error': f"Error making prediction: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check the health status of the API and model"""
    model_info = {
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'tensorflow_version': tf.__version__,
        'input_requirements': {
            'area': f"Between {MIN_AREA} and {MAX_AREA} square feet",
            'bedrooms': f"Between {MIN_ROOMS} and {MAX_ROOMS}",
            'bathrooms': f"Between {MIN_ROOMS} and {MAX_ROOMS}",
            'stories': f"Between 1 and {MAX_STORIES}",
            'parking': f"Between 0 and {MAX_PARKING}",
            'yes_no_fields': ['mainroad', 'guestroom', 'basement', 
                            'hotwaterheating', 'airconditioning', 'prefarea'],
            'furnishing_options': VALID_FURNISHING_STATUS
        },
        'currency': 'USD',
        'price_unit': 'Dollars',
        'model_metrics': {
            'mse': 1711665793024.00,
            'rmse': 1418332.05,
            'rmse_percentage': 24.32,
            'mae': 995244.56,
            'r2': 0.67
        }
    }
    
    if model is not None:
        try:
            model_info['model_summary'] = str(model.summary())
        except:
            model_info['model_summary'] = 'Error getting summary'
    
    return jsonify({
        'status': 'healthy' if model is not None and preprocessor is not None else 'degraded',
        **model_info
    })

if __name__ == '__main__':
    # Run the app on all network interfaces
    print(f"Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', debug=True, port=5000)