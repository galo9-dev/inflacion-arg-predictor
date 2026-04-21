import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import requests
import os

@st.cache_data(ttl=86400)  # cachea los datos 24 horas
@st.cache_data(ttl=86400)
def cargar_datos():
    # IPC
    url_ipc = "https://apis.datos.gob.ar/series/api/series/?ids=148.3_INIVELNAL_DICI_M_26&limit=100&sort=desc&format=json"
    data_ipc = requests.get(url_ipc).json()
    df_ipc = pd.DataFrame(data_ipc["data"], columns=["fecha", "ipc"])
    df_ipc["fecha"] = pd.to_datetime(df_ipc["fecha"])
    df_ipc = df_ipc.sort_values("fecha").reset_index(drop=True)
    df_ipc["inflacion_mensual"] = df_ipc["ipc"].pct_change() * 100
    df_ipc = df_ipc.dropna()

    # Tipo de cambio
    url_tc = "https://apis.datos.gob.ar/series/api/series/?ids=168.1_T_CAMBIOR_D_0_0_26&limit=200&collapse=month&collapse_aggregation=end_of_period&format=json"
    data_tc = requests.get(url_tc).json()
    df_tc = pd.DataFrame(data_tc["data"], columns=["fecha", "tipo_cambio"])
    df_tc["fecha"] = pd.to_datetime(df_tc["fecha"])
    df_tc = df_tc[df_tc["fecha"] <= pd.Timestamp.today()]

    # Salarios
    url_sal = "https://apis.datos.gob.ar/series/api/series/?ids=149.1_TL_INDIIOS_OCTU_0_21&limit=200&format=json"
    data_sal = requests.get(url_sal).json()
    df_sal = pd.DataFrame(data_sal["data"], columns=["fecha", "salarios"])
    df_sal["fecha"] = pd.to_datetime(df_sal["fecha"])

    # Merge
    df = pd.merge(df_ipc, df_tc, on="fecha", how="inner")
    df = pd.merge(df, df_sal, on="fecha", how="inner")
    return df

df = cargar_datos()

import requests, pandas as pd

url_tc = "https://apis.datos.gob.ar/series/api/series/?ids=168.1_T_CAMBIOR_D_0_0_26&limit=5&sort=desc&collapse=month&collapse_aggregation=end_of_period&format=json"
print("TC:", requests.get(url_tc).json()["data"][:3])

url_sal = "https://apis.datos.gob.ar/series/api/series/?ids=149.1_TL_INDIIOS_OCTU_0_21&limit=5&sort=desc&format=json"
print("SAL:", requests.get(url_sal).json()["data"][:3])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "data", "modelo.pkl"), "rb") as f:
    modelo = pickle.load(f)

st.title("Predictor de Inflacion Argentina")
st.markdown("Modelo basado en datos del INDEC y BCRA. **Uso académico, no financiero.**")

# --- Sección 1: Histórico ---
st.subheader("Histórico de inflación mensual")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df["fecha"], df["inflacion_mensual"], marker="o", color="crimson")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Inflación mensual (%)")
ax1.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# --- Sección 2: Predicción próximo mes ---
st.subheader("Predicción del próximo mes")

# Construir features con los últimos datos disponibles
df_feat = df.copy()
df_feat["inflacion_lag1"] = df_feat["inflacion_mensual"].shift(1)
df_feat["inflacion_lag2"] = df_feat["inflacion_mensual"].shift(2)
df_feat["inflacion_lag3"] = df_feat["inflacion_mensual"].shift(3)
df_feat["var_tc"] = df_feat["tipo_cambio"].pct_change() * 100
df_feat["var_tc_lag1"] = df_feat["var_tc"].shift(1)
df_feat["var_tc_lag2"] = df_feat["var_tc"].shift(2)
df_feat["var_sal"] = df_feat["salarios"].pct_change() * 100
df_feat["var_sal_lag1"] = df_feat["var_sal"].shift(1)
df_feat["var_tc_3m"] = df_feat["tipo_cambio"].pct_change(3) * 100
df_feat["inflacion_3m"] = df_feat["inflacion_mensual"].rolling(3).mean().shift(1)
df_feat = df_feat.dropna().reset_index(drop=True)

ultima_fila = df_feat[["inflacion_lag1", "inflacion_lag2", "inflacion_lag3",
                         "var_tc", "var_tc_lag1", "var_tc_lag2",
                         "var_sal", "var_sal_lag1", "var_tc_3m", "inflacion_3m"]].iloc[[-1]]

prediccion = modelo.predict(ultima_fila)[0]

meses = {1:"enero",2:"febrero",3:"marzo",4:"abril",5:"mayo",6:"junio",
         7:"julio",8:"agosto",9:"septiembre",10:"octubre",11:"noviembre",12:"diciembre"}
fecha_max = df["fecha"].max()
ultimo_mes = f"{meses[fecha_max.month]} {fecha_max.year}"

st.metric(label=f"Inflación estimada para el mes siguiente a {ultimo_mes}", 
          value=f"{prediccion:.2f}%")
st.info(f"Datos disponibles hasta {ultimo_mes}. La app se actualiza automáticamente cuando el INDEC publica nuevos datos en datos.gob.ar")

st.caption("⚠️ Este modelo es experimental. Los shocks políticos y cambiarios no son predecibles con datos históricos.")

# --- Sección 3: SHAP ---
st.subheader("¿Qué variables explican la predicción?")

explainer = shap.Explainer(modelo)
shap_values = explainer(ultima_fila)

fig2, ax2 = plt.subplots(figsize=(8, 4))
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
st.pyplot(fig2)