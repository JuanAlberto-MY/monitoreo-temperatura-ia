¡Claro! Aquí tienes el código completo de app_temperatura.py con el tamaño del gráfico ajustado y el tema oscuro aplicado. También te doy los comandos de Git para guardar y subir los cambios.

Paso 1: Código Completo y Actualizado para app_temperatura.py
Copia todo este código y pégalo en tu archivo app_temperatura.py en Visual Studio Code, reemplazando todo el contenido existente.

Python

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt

np.random.seed(42)

temperatura_normal_entrenamiento = 25 + 2 * np.random.randn(500)
temperatura_normal_entrenamiento = np.clip(temperatura_normal_entrenamiento, 20, 30)

temperatura_con_fallos_entrenamiento = np.copy(temperatura_normal_entrenamiento)
temperatura_con_fallos_entrenamiento[50:55] = np.random.uniform(45, 55, 5)
temperatura_con_fallos_entrenamiento[120:125] = np.random.uniform(5, 10, 5)
temperatura_con_fallos_entrenamiento[200:205] = 23.0
temperatura_con_fallos_entrenamiento[350:355] = np.random.uniform(40, 50, 5)

data_for_model_training = temperatura_con_fallos_entrenamiento.reshape(-1, 1)

model = IsolationForest(contamination=0.03, random_state=42)
model.fit(data_for_model_training)

# MODIFICACIÓN AQUÍ: Tema oscuro y tamaño de página
st.set_page_config(page_title="Monitor de Sensor de Temperatura", layout="wide", theme="dark")

st.title("🌡️ Sistema de Predicción de Fallos en Sensor de Temperatura")
st.markdown("---")

st.subheader("Monitoreo de Temperatura en Tiempo Real")

lectura_actual_container = st.empty()
estado_lectura_container = st.empty()
alerta_container = st.empty()
grafico_container = st.empty()
historico_container = st.empty()

historial_columnas = ['Hora', 'Lectura (°C)', 'Estado', 'Tipo de Anomalía', 'valor_numerico']
historial_lecturas_df = pd.DataFrame(columns=historial_columnas)
historial_lecturas_df['valor_numerico'] = historial_lecturas_df['valor_numerico'].astype(float)

st.write("Iniciando simulación de lecturas del sensor de temperatura...")

for i in range(1, 151):
    nueva_lectura = 0.0
    tipo_anomalia = "N/A"

    if i % 10 != 0 and i % 15 != 0 and i % 25 != 0:
        nueva_lectura = 25 + 2 * np.random.randn(1)[0]
        nueva_lectura = np.clip(nueva_lectura, 20, 30)
    elif i % 10 == 0:
        nueva_lectura = np.random.uniform(45, 55, 1)[0]
        tipo_anomalia = "Pico Alto"
    elif i % 15 == 0:
        nueva_lectura = np.random.uniform(5, 10, 1)[0]
        tipo_anomalia = "Caída Baja"
    elif i % 25 == 0:
        nueva_lectura = 23.0
        tipo_anomalia = "Valor Constante"

    prediccion = model.predict(np.array(nueva_lectura).reshape(1, -1))

    estado_lectura = "Normal"
    color_lectura = "green"
    mensaje_alerta = ""

    if prediccion == -1:
        estado_lectura = "ANOMALÍA DETECTADA"
        color_lectura = "red"
        mensaje_alerta = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                          f"({tipo_anomalia}) en la lectura del sensor: **{nueva_lectura:.2f}°C**. "
                          f"¡Se recomienda revisar el sensor!")
        alerta_container.error(mensaje_alerta)
    else:
        alerta_container.empty()

    nueva_fila_historial = pd.DataFrame([{
        'Hora': time.strftime('%H:%M:%S'),
        'Lectura (°C)': f"{nueva_lectura:.2f}",
        'Estado': estado_lectura,
        'Tipo de Anomalía': tipo_anomalia,
        'valor_numerico': nueva_lectura
    }])
    historial_lecturas_df = pd.concat([historial_lecturas_df, nueva_fila_historial], ignore_index=True)

    with lectura_actual_container.container():
        st.metric(label="Última Lectura de Temperatura", value=f"{nueva_lectura:.2f}°C", delta=None)

    with estado_lectura_container.container():
        st.markdown(f"<p style='color:{color_lectura}; font-size: 20px; font-weight: bold;'>Estado: {estado_lectura}</p>", unsafe_allow_html=True)

    with grafico_container.container():
        st.subheader("Gráfico de Tendencia de Temperatura")
        num_lecturas_grafico = 50
        df_para_grafico = historial_lecturas_df.tail(num_lecturas_grafico)

        # MODIFICACIÓN AQUÍ: Tamaño del gráfico (ancho, alto)
        fig, ax = plt.subplots(figsize=(8, 3)) 
        
        ax.plot(df_para_grafico['Hora'], df_para_grafico['valor_numerico'], label='Temperatura', color='skyblue') # Color de la línea
        
        anomalias_grafico = df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']
        if not anomalias_grafico.empty:
            ax.scatter(anomalias_grafico['Hora'], anomalias_grafico['valor_numerico'], color='red', s=100, marker='X', label='Anomalía')

        ax.set_xlabel('Hora')
        ax.set_ylabel('Temperatura (°C)')
        ax.set_title(f'Últimas {num_lecturas_grafico} Lecturas de Temperatura')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with historico_container.container():
        st.subheader("Historial de Lecturas Recientes")
        st.dataframe(historial_lecturas_df.tail(15).style.applymap(lambda x: 'background-color: #ffe6e6' if 'ANOMALÍA' in str(x) else '', subset=['Estado']))

    time.sleep(0.5)

st.success("✅ Simulación de monitoreo de sensor de temperatura finalizada.")