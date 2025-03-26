from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import traceback
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 1) Cargar el preprocesador y selector entrenados (no re-ajustar) ===
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.joblib')
SELECTOR_PATH = os.path.join(BASE_DIR, 'selector.joblib')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    selector = joblib.load(SELECTOR_PATH)
    print("Preprocessor y selector cargados correctamente.")
except Exception as e:
    print("Error cargando preprocessor/selector:", e)
    preprocessor = None
    selector = None

# Si tu modelo usaba alguna función custom, defínela:
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Cargar el modelo
if os.path.exists(MODEL_PATH):
    try:
        custom_objects = {'mse': custom_mse}
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print("Error cargando el modelo:", e)
        model = None
else:
    print("Modelo no encontrado:", MODEL_PATH)
    model = None

# === 2) Cargar el mismo test set que en el notebook ===
X_TEST_PATH = os.path.join(BASE_DIR, 'X_test.csv')
Y_TEST_PATH = os.path.join(BASE_DIR, 'y_test.csv')

try:
    X_test = pd.read_csv(X_TEST_PATH)
    # Si y_test.csv tiene una columna 'SalePrice' o similar, ajusta según tu caso
    y_test = pd.read_csv(Y_TEST_PATH)['SalePrice']
    print("X_test, y_test cargados correctamente.")
except Exception as e:
    print("Error cargando X_test / y_test:", e)
    X_test = None
    y_test = None
# === ENDPOINT DE PREDICCIÓN ===
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        # Preparar el DataFrame de entrada a partir de los datos recibidos
        input_df = prepare_input_data(data)
        # Transformar con el preprocesador y aplicar el selector
        input_pre = preprocessor.transform(input_df)
        input_sel = selector.transform(input_pre)
        # Realizar la predicción (se asume que el modelo entrenó con log1p)
        y_pred_log = model.predict(input_sel)
        y_pred = np.expm1(y_pred_log).flatten()
        price_in_dollars = float(y_pred[0])
        response = {
            'prediction': {
                'price_in_dollars': price_in_dollars,
                'price_formatted': f"${price_in_dollars:,.2f}",
                'currency': 'USD'
            },
            'input_data': data
        }
        return jsonify(response)
    except Exception as e:
        print("Error during prediction:", e)
        print(traceback.format_exc())
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500
# === ENDPOINT DE EVALUACIÓN ===
@app.route('/api/evaluate', methods=['GET'])
def evaluate():
    if model is None or preprocessor is None or selector is None or X_test is None or y_test is None:
        return jsonify({'error': 'Evaluation not available'}), 500
    try:
        # Transformar X_test con el preprocesador y selector que se ajustaron en el notebook
        X_test_pre = preprocessor.transform(X_test)
        X_test_sel = selector.transform(X_test_pre)

        # Predicción (el modelo entrenó en log1p)
        y_pred_log = model.predict(X_test_sel)
        y_pred = np.expm1(y_pred_log).flatten()

        # Invertir log1p en y_test
        y_test_exp = np.expm1(y_test.values)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred))
        mae = mean_absolute_error(y_test_exp, y_pred)
        r2 = r2_score(y_test_exp, y_pred)
        rmsep = (rmse / y_test_exp.mean()) * 100

        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'RMSE (%)': f"{rmsep:.2f}%"
        }

        # Comparativa (primeras 10 filas)
        comparison = []
        n = min(10, len(y_test_exp))
        for i in range(n):
            comparison.append({
                'Precio Real (USD)': float(y_test_exp[i]),
                'Precio Predicho (USD)': float(y_pred[i]),
                'Error Absoluto (USD)': float(abs(y_test_exp[i] - y_pred[i]))
            })

        return jsonify({
            'metrics': metrics,
            'comparison': comparison
        })
    except Exception as e:
        print("Error during evaluation:", e)
        print(traceback.format_exc())
        return jsonify({'error': f"Error during evaluation: {str(e)}"}), 500

# === ENDPOINT DE PREDICCIÓN ===
# (Aquí ya depende de tu lógica, lo importante es no volver a re-fit, solo transform y predict)

@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'healthy' if model and preprocessor and selector and X_test is not None else 'degraded'
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'selector_loaded': selector is not None
    })

if __name__ == '__main__':
    print("Iniciando el servidor Flask en el puerto 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
