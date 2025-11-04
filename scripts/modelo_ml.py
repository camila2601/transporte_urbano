import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import time
from tqdm import tqdm  # Para barras de progreso

def create_features(df):
    """Crear features básicas para el modelo."""
    print("Creando features...")
    features = pd.DataFrame()
    
    # Features básicas y simples
    features['passenger_count'] = df['passenger_count']
    features['hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour
    
    # Feature de distancia simplificada (usando diferencia de coordenadas como aproximación)
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
        features['distance_approx'] = np.sqrt(
            (df['dropoff_latitude'] - df['pickup_latitude'])**2 +
            (df['dropoff_longitude'] - df['pickup_longitude'])**2
        ) * 111  # Aproximación rápida a kilómetros (1 grado ≈ 111 km)
    
    return features

def evaluate_model(y_true, y_pred, model_name):
    """Evaluar modelo con métricas básicas."""
    results = {
        'model': model_name,
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    return results

def train_and_evaluate():
    """Entrenar y evaluar modelos de predicción de duración."""
    start_time = time.time()
    
    print("\n🔄 Cargando datos...")
    try:
        # Usar una muestra más pequeña para reducir uso de memoria
        df = pd.read_csv('data/nyc_taxi_clean.csv')
        SAMPLE_SIZE = 10000  # reducir a 10k por defecto para evitar OOM
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
        print(f"Usando {len(df):,} registros para entrenamiento (SAMPLE_SIZE={SAMPLE_SIZE})")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return
    
    print("\n🔄 Preparando features...")
    X = create_features(df)
    y = df['duration_min']
    
    print("\n🔄 Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear directorio para modelos
    Path('models').mkdir(exist_ok=True)
    
    # Definir modelos
    models = {
        'linear': LinearRegression(),
        'random_forest': RandomForestRegressor(
            n_estimators=20,  # menos árboles para bajar memoria/CPU
            max_depth=8,      # limitar profundidad
            random_state=42,
            n_jobs=1          # evitar paralelismo excesivo en memoria limitada
        )
    }
    
    results = []
    
    # Entrenar y evaluar cada modelo
    for name, model in models.items():
        print(f"\n🔄 Entrenando modelo {name}...")
        try:
            # Entrenamiento (puede subir memoria). Capturamos MemoryError para fallback.
            model.fit(X_train, y_train)
            print(f"✓ Entrenamiento completado")

            # Predicciones
            print("Realizando predicciones...")
            y_pred = model.predict(X_test)
        except MemoryError:
            print(f"⚠️ MemoryError entrenando {name}. Intentando fallback con menor complejidad...")
            # Intentar fallback: reducir n_estimators si es RandomForest
            if hasattr(model, 'n_estimators'):
                try:
                    model.set_params(n_estimators=max(5, int(model.n_estimators / 4)))
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    print(f"✓ Fallback completado con n_estimators={model.n_estimators}")
                except Exception as e:
                    print(f"❌ Fallback falló: {e}")
                    continue
            else:
                print("❌ No hay fallback disponible para este modelo")
                continue
        
        # Evaluación
        eval_results = evaluate_model(y_test, y_pred, name)
        results.append(eval_results)
        
        # Guardar modelo
        model_path = f'models/taxi_duration_{name}.joblib'
        joblib.dump(model, model_path)
        print(f"✅ Modelo guardado en {model_path}")
    
    # Imprimir resultados
    print("\n📊 Resumen de Resultados:")
    for r in results:
        print(f"\nModelo: {r['model']}")
        print(f"RMSE: {r['rmse']:.2f} minutos")
        print(f"R²: {r['r2']:.3f}")
    
    # Tiempo total
    tiempo_total = time.time() - start_time
    print(f"\n⏱️ Tiempo total de ejecución: {tiempo_total:.1f} segundos")

if __name__ == '__main__':
    train_and_evaluate()
