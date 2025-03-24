# House Price Prediction App

Esta aplicación permite predecir el precio de una casa basado en sus características, utilizando un modelo de Machine Learning entrenado con datos reales.

## Estructura del Proyecto

```
house_price_prediction_model_training/
├── app/
│   ├── client/
│   │   └── index.html
│   └── server/
│       ├── app.py
│       ├── data/
│       │   └── Housing.csv
│       ├── house_price_model_v3.keras
│       └── requirements.txt
└── README.md
```

## Requisitos Previos

- Python 3.12 o superior
- pip (gestor de paquetes de Python)
- Navegador web moderno (Chrome, Firefox, Edge, etc.)

## Configuración del Entorno

1. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
```

2. Activa el entorno virtual:
   - En Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   - En macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

3. Instala las dependencias:
```bash
cd app/server
pip install -r requirements.txt
```

## Ejecutar la Aplicación

### 1. Iniciar el Servidor

1. Asegúrate de estar en el directorio del servidor:
```bash
cd app/server
```

2. Inicia el servidor Flask:
```bash
python app.py
```

El servidor se iniciará en `http://localhost:5000`

### 2. Abrir el Cliente

1. Abre el archivo `app/client/index.html` en tu navegador web preferido.
   - Puedes hacerlo directamente haciendo doble clic en el archivo
   - O usando un servidor web local simple como `python -m http.server` en el directorio `app/client`

## Uso de la Aplicación

1. Una vez que el cliente y el servidor estén ejecutándose, verás:
   - El estado del servidor (debe mostrar "Healthy")
   - Las métricas de rendimiento del modelo
   - Un formulario para ingresar las características de la casa

2. Completa el formulario con los siguientes datos:
   - Área (en pies cuadrados)
   - Número de dormitorios
   - Número de baños
   - Número de pisos
   - Acceso a calle principal
   - Habitación de invitados
   - Sótano
   - Calefacción de agua caliente
   - Aire acondicionado
   - Espacios de estacionamiento
   - Área preferencial
   - Estado del amueblado

3. Haz clic en "Predict Price" para obtener la predicción del precio de la casa.

## Métricas del Modelo

El modelo muestra las siguientes métricas de rendimiento:
- MSE (Error Cuadrático Medio)
- RMSE (Raíz del Error Cuadrático Medio)
- RMSE % (Porcentaje del RMSE)
- MAE (Error Absoluto Medio)
- R² (Coeficiente de Determinación)

## Solución de Problemas

1. Si el servidor muestra "Status: Error":
   - Verifica que el servidor Flask esté ejecutándose
   - Asegúrate de que el puerto 5000 esté disponible
   - Revisa los logs del servidor para más detalles

2. Si las predicciones no se muestran:
   - Verifica la consola del navegador para errores
   - Asegúrate de que todos los campos del formulario estén completos
   - Verifica que los valores estén dentro de los rangos permitidos

## Tecnologías Utilizadas

- Backend:
  - Python
  - Flask
  - TensorFlow
  - scikit-learn
  - pandas
  - numpy

- Frontend:
  - HTML5
  - Bootstrap 5
  - JavaScript (Vanilla) 