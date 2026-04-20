# Predictor de Inflación Argentina

Modelo predictivo de inflación mensual argentina usando datos oficiales del INDEC y BCRA.

## Demo en vivo
https://inflacion-arg-predictor-grqu3g3eunsws74pwwbdt3.streamlit.app

## Qué hace
- Descarga datos oficiales de inflación (IPC), tipo de cambio y salarios via la API de datos.gob.ar
- Entrena un modelo XGBoost con features de series temporales (lags, variaciones mensuales)
- Predice la inflación del próximo mes
- Explica qué variables impulsan cada predicción usando SHAP
- Muestra todo en un dashboard interactivo con Streamlit

## Resultado del modelo
- MAE: 3.45 puntos porcentuales
- Los shocks políticos como el de diciembre 2023 (25% mensual) no son predecibles con datos históricos, lo cual es un resultado honesto y esperado

## Tecnologías
- Python, Pandas, XGBoost, SHAP, Streamlit
- Datos: INDEC y BCRA via datos.gob.ar

## Cómo correrlo localmente
1. Clonar el repo
2. Crear entorno virtual: `python -m venv venv`
3. Activar: `venv\Scripts\activate`
4. Instalar dependencias: `pip install -r requirements.txt`
5. Correr: `streamlit run src/app.py`

## Aclaración
Este proyecto es de uso académico. No es una herramienta financiera.