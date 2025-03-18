### **Caso de Estudio: Modelo de Red Neuronal para la Predicción de Precios de Viviendas**
**Duracion:(2 Semanas)**  


---

## **1. Business Case Discovery**  
### **1.1 Contexto del Negocio**  
Eres parte de un equipo de análisis en una firma de inversión inmobiliaria. La empresa busca mejorar la precisión en la valoración de propiedades con el objetivo de maximizar la rentabilidad y minimizar riesgos. En este mercado, errores en la estimación de precios pueden resultar en pérdidas millonarias.

La empresa dispone de un dataset histórico

https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

con las siguientes variables: 

- **Características de entrada:**  
  - `area`: Área total de la vivienda en pies cuadrados.  
  - `bedrooms`: Número de habitaciones.  
  - `bathrooms`: Número de baños.  
  - `stories`: Número de pisos.  
  - `mainroad`: Si la casa está en una calle principal (Sí/No).  
  - `guestroom`: Si tiene habitación de invitados (Sí/No).  
  - `basement`: Si cuenta con sótano (Sí/No).  
  - `hotwaterheating`: Si tiene calefacción de agua caliente (Sí/No).  
  - `airconditioning`: Si cuenta con aire acondicionado (Sí/No).  
  - `parking`: Número de plazas de estacionamiento.  
  - `prefarea`: Si está en una zona preferencial (Sí/No).  
  - `furnishingstatus`: Estado del mobiliario (Amueblado, Semiamueblado, No amueblado).  
- **Variable objetivo:**  
  - `price`: Precio de venta de la propiedad (variable continua a predecir).  

### **1.2 Objetivo del Proyecto**  
Desarrollar un modelo de redes neuronales para predecir los precios de las viviendas con una precisión superior a los métodos tradicionales como la regresión lineal.

### **1.3 Stakeholders y Requisitos**  
- **Inversores:** Buscan predicciones con un margen de error inferior al 10%.
- **Equipo Legal:** Exigen que el modelo sea interpretable para justificar decisiones en auditorías.
- **Equipo Técnico:** La implementación debe ser escalable y compatible con entornos de nube (AWS/GCP), utilizando Contenedores Docker.


### **1.4 Métricas de Éxito**  
- **Error cuadrático medio (RMSE):** Inferior al 5% del precio promedio de las viviendas.  
- **Coeficiente de determinación (R²):** Mayor a 0.85.  


### **1.5 Desafíos Anticipados**  
- **Multicolinealidad:** Variables como `area`, `bedrooms` y `bathrooms` pueden estar correlacionadas.
- **Distribuciones sesgadas:** Variables binarias y categóricas pueden necesitar codificación adecuada.
- **Sobreajuste:** La red neuronal puede memorizar datos en lugar de generalizar.

---

## **2. Data Processing**  
### **2.1 Carga y Exploración Inicial**  
El alumno deberá cargar el dataset, visualizar las primeras filas y explorar la distribución de los datos con histogramas y diagramas de caja. Se recomienda el uso de `pandas` y `matplotlib`.

**Pistas:**
- Identificar distribuciones con colas largas y valores atípicos.
- Revisar la presencia de valores nulos.

### **2.2 Análisis de Correlación**  
El alumno generará una matriz de correlación para analizar relaciones entre variables. Se sugiere utilizar `seaborn`.

**Pistas:**
- Evaluar qué variables tienen alta correlación con `price`.
- Considerar la eliminación de variables redundantes.

### **2.3 Preprocesamiento de Datos**  
- **Manejo de valores faltantes:** Si existen, imputarlos con la mediana o la moda según el tipo de variable.
- **Codificación de variables categóricas:** Convertir `mainroad`, `guestroom`, `basement`, etc., en valores numéricos.
- **Normalización:** Escalar las variables numéricas con `StandardScaler` o `MinMaxScaler`. 

### **2.4 División Train-Test**  
El alumno deberá dividir los datos en entrenamiento y prueba (80/20). Se recomienda usar `train_test_split` de `sklearn`.

---

## **3. Model Planning**  
### **3.1 Definición del Problema**  
- **Tipo de Modelo:** Regresión con red neuronal.
- **Entrada:** 12 características preprocesadas.
- **Salida:** Predicción del precio.

### **3.2 Arquitectura de la Red**  
El alumno deberá diseñar una red neuronal con:
- **Capa de Entrada:** 12 neuronas (una por feature).
- **Capas Ocultas:** Dos capas densas con activación ReLU.
- **Capa de Salida:** 1 neurona con activación lineal.

### **3.3 Función de Pérdida y Optimizador**  
- **Loss Function:** Error cuadrático medio (MSE).
- **Optimizadores a comparar:** Adam y SGD con momentum.

### **3.4 Evaluación del Modelo**  
Se analizarán métricas como MAE, RMSE y R², y se usará validación cruzada (K-Fold con k=5).

---

## **4. Model Building and Selection**  
### **4.1 Implementación en Keras**  
El alumno construirá el modelo usando `keras.Sequential()`.

### **4.2 Entrenamiento del Modelo**  
Entrenar la red con un número adecuado de épocas y un tamaño de batch óptimo, asegurando una validación efectiva para evitar sobreajuste.

Ejemplo: 100 épocas con `batch_size=32` y validación del 20%.

### **4.3 Experimentación de Hiperparámetros**  
- Probar distintas tasas de aprendizaje (`0.001`, `0.0001`).
- Comparar `batch_size=16` vs `64`.
- Aplicar regularización con `Dropout` o `L2`.

### **4.4 Evaluación en Conjunto de Test**  
El alumno comparará las métricas finales con las del conjunto de entrenamiento para verificar si hay sobreajuste.

---

## **5. Presentación de Resultados**  

### **5.1 Evaluación de Predicciones**  
El alumno deberá analizar la calidad de las predicciones utilizando gráficos de dispersión, líneas de tendencia y métricas clave como RMSE y R². Se recomienda interpretar visualmente cómo se alinean las predicciones con los valores reales y detectar posibles patrones de error.  

### **5.2 Análisis de Errores**  
Se identificarán las predicciones con errores significativos, especialmente aquellas con desviaciones superiores al 5% del precio real. El alumno deberá investigar si estos errores se deben a outliers, falta de representatividad en los datos de entrenamiento o limitaciones del modelo.  

### **5.3 Comparación con Baseline (OPCIONAL)**  
Para evaluar la efectividad del modelo, se compararán sus resultados con un modelo de regresión lineal tradicional. Se analizarán diferencias en RMSE, R² y la distribución de errores para justificar el uso de una red neuronal frente a métodos más simples.

---

## **6. Deployment**  
### **6.1 Serialización del Modelo**  
Se guardará el modelo en formato `.h5` o `.keras` para su reutilización.

### **6.2 Creación de API de Predicción**  
El alumno deberá implementar un endpoint con `Flask` que reciba datos y devuelva una predicción.


---

### **Conclusión**  
Este caso de estudio permite aplicar conceptos clave del ciclo de vida de un modelo de deep learning, combinando teoría y práctica. Se espera que el alumno documente cada decisión tomada en un informe técnico.

