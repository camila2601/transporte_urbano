import os
import requests
import pandas as pd
import time

API_KEY = "c511e473f0a44628333ca698b8e9ea92"
URL = "https://api.openweathermap.org/data/2.5/weather"
DELAY = 1           # segundos entre requests
NUM_FILAS = 15      # <--- Cambia este valor para procesar más o menos filas

# Rutas seguras del proyecto

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'Data')

csv_input = os.path.join(DATA_DIR, 'nyc_taxi_clean.csv')
csv_output = os.path.join(DATA_DIR, 'nyc_taxi_weather.csv')

# Cargar dataset limpio

df = pd.read_csv(csv_input)

# Seleccionar solo las filas que queremos procesar
df = df.head(NUM_FILAS).copy()

# Crear columnas vacías para clima
df['clima_main'] = 'N/A'
df['clima_desc'] = 'N/A'

# Consultar clima fila por fila

for i, row in df.iterrows():
    lat = row['pickup_latitude']
    lon = row['pickup_longitude']
    params = {'lat': lat, 'lon': lon, 'appid': API_KEY, 'units': 'metric'}
    
    try:
        response = requests.get(URL, params=params)
        data = response.json()
        if 'weather' in data and len(data['weather']) > 0:
            df.loc[i, 'clima_main'] = data['weather'][0]['main']
            df.loc[i, 'clima_desc'] = data['weather'][0]['description']
        else:
            df.loc[i, 'clima_main'] = 'N/A'
            df.loc[i, 'clima_desc'] = 'N/A'
    except Exception:
        df.loc[i, 'clima_main'] = 'N/A'
        df.loc[i, 'clima_desc'] = 'N/A'
    
    print(f"Fila {i+1}: lat={lat}, lon={lon}, main={df.loc[i, 'clima_main']}, desc={df.loc[i, 'clima_desc']}")
    time.sleep(DELAY)

# Guardar dataset enriquecido

df.to_csv(csv_output, index=False)
print(f"✅ Datos enriquecidos con clima guardados en {csv_output}")
