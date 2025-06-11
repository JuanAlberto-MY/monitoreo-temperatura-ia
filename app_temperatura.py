import streamlit as st # La librería principal para crear la interfaz web
import git remote add origin https://github.com/JuanAlberto-MY/monitoreo-temperatura-ia.git as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time

# --- PARTE 1: Entrenamiento inicial del modelo (igual que el Paso 3) ---
# Este código se ejecuta una vez cuando inicia la aplicación de Streamlit
np.random.seed(42)

# Generar datos de temperatura "normales" para el entrenamiento del modelo
temperatura_normal_entrenamiento = 25 + 2 * np.random.randn(500)
temperatura_normal_entrenamiento = np.clip(temperatura_normal_entrenamiento, 20, 30)

# Introducir fallos en los datos de entrenamiento para que el modelo aprenda a detectarlos
temperatura_con_fallos_entrenamiento = np.copy(temperatura_normal_entrenamiento)
temperatura_con_fallos_entrenamiento[50:55] = np.random.uniform(45, 55, 5) # Pico alto
temperatura_con_fallos_entrenamiento[120:125] = np.random.uniform(5, 10, 5) # Caída baja
temperatura_con_fallos_entrenamiento[200:205] = 23.0 # Valor constante
temperatura_con_fallos_entrenamiento[350:355] = np.random.uniform(40, 50, 5) # Otro pico alto

data_for_model_training = temperatura_con_fallos_entrenamiento.reshape(-1, 1)

# Entrenar el modelo Isolation Forest
# 'contamination' es la proporción de anomalías que el modelo espera encontrar en los datos
model = IsolationForest(contamination=0.03, random_state=42) # Ajustado a 3%
model.fit(data_for_model_training)
# --- FIN PARTE 1 ---

# --- PARTE 2: Configuración y Diseño de la Interfaz con Streamlit ---
# Configura el título de la página y el diseño (wide = más ancho)
st.set_page_config(page_title="Monitor de Sensor de Temperatura", layout="wide")

st.title("🌡️ Sistema de Predicción de Fallos en Sensor de Temperatura")
st.markdown("---") # Una línea divisoria

st.subheader("Monitoreo de Temperatura en Tiempo Real")

# Contenedores vacíos para que podamos actualizar su contenido dinámicamente
lectura_actual_container = st.empty()
estado_lectura_container = st.empty()
alerta_container = st.empty()
historico_container = st.empty()

# DataFrame para almacenar el historial de lecturas. Se inicializa vacío.
historial_columnas = ['Hora', 'Lectura (°C)', 'Estado', 'Tipo de Anomalía']
historial_lecturas_df = pd.DataFrame(columns=historial_columnas)

st.write("Iniciando simulación de lecturas del sensor de temperatura...")

# --- PARTE 3: Simulación de Lecturas y Detección de Anomalías en Vivo ---
# Este bucle simula el sensor enviando lecturas continuamente
for i in range(1, 151): # Simular 150 lecturas para el ejemplo
    # Generar una nueva lectura de temperatura.
    # La mayoría de las lecturas serán normales, algunas serán anomalías simuladas.
    nueva_lectura = 0.0
    tipo_anomalia = "N/A" # Para registrar el tipo de fallo simulado

    if i % 10 != 0 and i % 15 != 0 and i % 25 != 0:
        # Lectura normal: dentro del rango 20-30°C con un poco de ruido
        nueva_lectura = 25 + 2 * np.random.randn(1)[0]
        nueva_lectura = np.clip(nueva_lectura, 20, 30)
    elif i % 10 == 0:
        # Anomalía de pico alto
        nueva_lectura = np.random.uniform(45, 55, 1)[0]
        tipo_anomalia = "Pico Alto"
    elif i % 15 == 0:
        # Anomalía de caída baja
        nueva_lectura = np.random.uniform(5, 10, 1)[0]
        tipo_anomalia = "Caída Baja"
    elif i % 25 == 0:
        # Anomalía de valor constante (sensor "pegado")
        nueva_lectura = 23.0 # Se mantiene en 23.0 grados
        tipo_anomalia = "Valor Constante"

    # Predecir si la nueva lectura es una anomalía usando el modelo entrenado
    # Importante: el modelo espera un array 2D, por eso .reshape(1, -1)
    prediccion = model.predict(np.array(nueva_lectura).reshape(1, -1))

    estado_lectura = "Normal"
    color_lectura = "green"
    mensaje_alerta = ""

    if prediccion == -1: # Si el modelo predice una anomalía
        estado_lectura = "ANOMALÍA DETECTADA"
        color_lectura = "red"
        # Mostrar mensaje de alerta en la interfaz
        mensaje_alerta = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                  f"({tipo_anomalia}) en la lectura del sensor: **{nueva_lectura:.2f}°C**. " # <--- CORREGIR AQUÍ
                  f"¡Se recomienda revisar el sensor!")
    else:
        alerta_container.empty() # Si no hay anomalía, asegurarse de que no haya un mensaje de alerta visible

    # --- Actualizar el Historial de Lecturas ---
    # Añadir la lectura actual al DataFrame del historial
    nueva_fila_historial = pd.DataFrame([{
        'Hora': time.strftime('%H:%M:%S'),
        'Lectura (°C)': f"{nueva_lectura:.2f}",
        'Estado': estado_lectura,
        'Tipo de Anomalía': tipo_anomalia
    }])
    # Concatenar la nueva fila al historial existente. ignore_index=True es importante
    historial_lecturas_df = pd.concat([historial_lecturas_df, nueva_fila_historial], ignore_index=True)

    # --- Actualizar la Interfaz de Usuario de Streamlit ---
    with lectura_actual_container.container():
        # Muestra la última lectura de temperatura de forma destacada
        st.metric(label="Última Lectura de Temperatura", value=f"{nueva_lectura:.2f}°C", delta=None)

    with estado_lectura_container.container():
        # Muestra el estado (Normal/Anomalía) con color para mejor visibilidad
        st.markdown(f"<p style='color:{color_lectura}; font-size: 20px; font-weight: bold;'>Estado: {estado_lectura}</p>", unsafe_allow_html=True)

    with historico_container.container():
        st.subheader("Historial de Lecturas Recientes")
        # Mostrar solo las últimas 15 lecturas en la tabla para no saturar
        # .style.applymap() se usa para colorear filas con anomalías, haciendo más fácil detectarlas
        st.dataframe(historial_lecturas_df.tail(15).style.applymap(lambda x: 'background-color: #ffe6e6' if 'ANOMALÍA' in str(x) else '', subset=['Estado']))

    time.sleep(0.5) # Pausar 0.5 segundos para simular el tiempo entre lecturas

st.success("✅ Simulación de monitoreo de sensor de temperatura finalizada.")