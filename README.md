# Transporte Urbano — Análisis de datos de taxis

Proyecto para construir un pipeline de Big Data que analiza datos de taxis (NYC Taxi sample) — preprocesamiento, análisis estadístico, modelos predictivos y visualizaciones. Incluye consideraciones de privacidad y opciones de despliegue.

## Estructura del repositorio
- `data/` — datos (samples y outputs). Atención: los CSV grandes se gestionan con Git LFS.
- `notebooks/` — notebooks de preprocesamiento, modelo y visualización.
- `scripts/` — scripts auxiliares (ETL, privacidad, modelos).
- `reportes/` — artefactos de reporte (plots, HTML, informe).

## Requisitos
- Python 3.9+ (recomendado)
- Crear entorno virtual e instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecutar
- Ejecuta el preprocesamiento (notebook o script): abre `notebooks/01_preprocesamiento.ipynb` en Jupyter o VSCode y ejecuta las celdas.
- Puedes probar localmente con el script de ejemplo:

```powershell
python .\scripts\test_preproc_run.py
```

## Git LFS y archivos grandes
Los datasets grandes se gestionan con Git LFS. Ya configuramos LFS y migramos los CSV de muestra. Si añades nuevos datos grandes, usa:

```powershell
git lfs track "*.csv"
git add .gitattributes
git add data/your_large_file.csv
git commit -m "Add large dataset to LFS"
git push origin <branch>
```

## Privacidad y ética
Incluye `scripts/privacy.py` con funciones para anonimizar identificadores y coordenadas (hashing y redondeo). Antes de publicar resultados o compartir datos, aplica anonimización y documenta el proceso.

## Siguientes pasos recomendados
- Añadir script `scripts/spark_etl.py` para demostrar procesamiento con PySpark.
- Añadir tests rápidos y pipeline CI (GitHub Actions).
- Evitar commitear entornos (`venv/`) — están en `.gitignore`.

## Contacto
Para más cambios, crea una rama y abre un pull request con descripción clara de los cambios.
