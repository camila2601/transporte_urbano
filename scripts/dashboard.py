import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path


@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    
    try:
        df = pd.read_csv('data/nyc_taxi_clean_anonymized.csv')
    except Exception:
        df = pd.read_csv('data/nyc_taxi_clean.csv')

    # Convertir columnas de fecha
    date_cols = [col for col in df.columns if 'datetime' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    return df


def main():
    # Configuración de la página (debe ser la primera llamada a Streamlit)
    st.set_page_config(
        page_title="NYC Taxi Analysis Dashboard",
        page_icon="🚕",
        layout="wide"
    )

    # Título
    st.title("🚕 NYC Taxi Trip Analysis")
    st.write("Dashboard para análisis de viajes en taxi y predicciones")

    # Título
    st.title("🚕 NYC Taxi Trip Analysis")
    st.write("Dashboard para análisis de viajes en taxi y predicciones")

    # Cargar datos
    df = load_data()

    SAMPLE_SIZE = 10000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

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
            
            # Antes de crear el mapa
            map_data = df.sample(n=min(1000, len(df)))
            fig = px.density_mapbox(
                map_data,
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
        # Construir controles básicos (valores por defecto)
        col1, col2 = st.columns(2)
        with col1:
            hour_input = st.slider("Hora del día", 0, 23, 12)
        with col2:
            distance_input = st.number_input("Distancia (km)", 0.1, 200.0, 5.0)

        # Botón para predecir
        if st.button("Predecir Duración"):
            model_path = Path("models/taxi_duration_model.joblib")
            if not model_path.exists():
                st.info("Modelo no encontrado. Entrena el modelo primero usando scripts/modelo_ml.py")
            else:
                try:
                    model = joblib.load(model_path)

                    # Intentar detectar las features esperadas por el modelo
                    expected = None
                    try:
                        expected = list(getattr(model, 'feature_names_in_', None) or [])
                    except Exception:
                        expected = None

                    if not expected:
                        # Fallback a la convención usada en el proyecto
                        expected = ['passenger_count', 'hour', 'distance_approx']

                    st.write("Características esperadas por el modelo:", expected)

                    # Construir input dict respetando el orden de features
                    input_dict = {}
                    for feat in expected:
                        if feat == 'passenger_count':
                            input_dict['passenger_count'] = [1]
                        elif feat == 'hour':
                            input_dict['hour'] = [int(hour_input)]
                        elif feat == 'distance_approx':
                            # distance_input is in km already
                            input_dict['distance_approx'] = [float(distance_input)]
                        else:
                            # Si la feature no está cubierta, rellenar con 0/NaN según sea numérica
                            input_dict[feat] = [0]

                    # Crear DataFrame con columnas en el orden esperado
                    input_df = pd.DataFrame(input_dict)
                    input_df = input_df.reindex(columns=expected)

                    # Intentar predecir
                    try:
                        prediction = model.predict(input_df)
                        st.success(f"Duración estimada: {float(prediction[0]):.1f} minutos")
                    except Exception as e:
                        st.error(f"Error al predecir con el modelo cargado: {e}")
                except Exception as e:
                    st.error(f"No se pudo cargar el modelo: {e}")

    # Footer con metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Metadata")
    st.sidebar.write(f"Registros: {len(df):,}")
    if 'pickup_datetime' in df.columns:
        st.sidebar.write(f"Período: {df['pickup_datetime'].min().date()} a {df['pickup_datetime'].max().date()}")


if __name__ == '__main__':
    main()