
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="NYC Taxi Analysis Dashboard",
    page_icon="🚕",
    layout="wide"
)

# Título 
st.title("🚕 NYC Taxi Trip Analysis")
st.write("Dashboard para análisis de viajes en taxi y predicciones")


@st.cache_data
def load_data():
    try:
        # Intentar cargar datos anónimos primero
        df = pd.read_csv('data/nyc_taxi_clean_anonymized.csv')
    except:
        # Si no existe, cargar datos limpios
        df = pd.read_csv('data/nyc_taxi_clean.csv')
    
    # Convertir columnas de fecha
    date_cols = [col for col in df.columns if 'datetime' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df

# Mostrar progreso mientras carga
with st.spinner('Cargando datos...'):
    df = load_data()

# Sidebar con filtros
st.sidebar.header("Filtros")

# Filtro de fecha si existe
if 'pickup_datetime' in df.columns:
    min_date = df['pickup_datetime'].min()
    max_date = df['pickup_datetime'].max()
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, min_date + timedelta(days=7)),
        min_value=min_date,
        max_value=max_date
    )

# Métricas principales
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Viajes",
        value=len(df),
        delta="desde la última actualización"
    )

with col2:
    if 'duration_min' in df.columns:
        avg_duration = df['duration_min'].mean()
        st.metric(
            label="Duración Promedio (min)",
            value=f"{avg_duration:.1f}",
            delta=None
        )

with col3:
    if 'hour' in df.columns:
        peak_hour = df['hour'].mode().iloc[0]
        st.metric(
            label="Hora Pico",
            value=f"{peak_hour:02d}:00",
            delta=None
        )

# Gráficos
st.header("📊 Visualizaciones")

tab1, tab2, tab3 = st.tabs(["Distribución Temporal", "Mapa de Calor", "Predicciones"])

with tab1:
    if 'hour' in df.columns:
        # Distribución por hora
        hourly_trips = df['hour'].value_counts().sort_index()
        fig = px.bar(
            x=hourly_trips.index,
            y=hourly_trips.values,
            title="Distribución de Viajes por Hora",
            labels={'x': 'Hora del día', 'y': 'Número de viajes'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude']):
        # Mapa de calor de pickups
        st.subheader("Mapa de Densidad de Recogidas")
        fig = px.density_mapbox(
            df,
            lat='pickup_latitude',
            lon='pickup_longitude',
            radius=10,
            center=dict(lat=40.7, lon=-73.9),
            zoom=10,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Predicción de Duración")
    
    # Input para predicción
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hora del día", 0, 23, 12)
    with col2:
        distance = st.number_input("Distancia (km)", 0.1, 50.0, 5.0)
        
    # Botón para predecir
    if st.button("Predecir Duración"):
        try:
            # Intentar cargar modelo
            model_path = Path("models/taxi_duration_model.joblib")
            if model_path.exists():
                model = joblib.load(model_path)
                prediction = model.predict([[hour, distance]])
                st.success(f"Duración estimada: {prediction[0]:.1f} minutos")
            else:
                st.info("Modelo no encontrado. Entrena el modelo primero usando scripts/modelo_ml.py")
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# Footer con metadata
st.sidebar.markdown("---")
st.sidebar.markdown("### Metadata")
st.sidebar.write(f"Registros: {len(df):,}")
if 'pickup_datetime' in df.columns:
    st.sidebar.write(f"Período: {df['pickup_datetime'].min().date()} a {df['pickup_datetime'].max().date()}")