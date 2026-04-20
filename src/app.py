import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Cargar datos y modelo
df = pd.read_csv("../data/raw/datos_finales.csv", parse_dates=["fecha"])

with open("../data/modelo.pkl", "rb") as f:
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
import locale
locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")
ultimo_mes = df["fecha"].max().strftime("%B %Y")

st.metric(label=f"Inflación estimada para el mes siguiente a {ultimo_mes}", 
          value=f"{prediccion:.2f}%")

st.caption("⚠️ Este modelo es experimental. Los shocks políticos y cambiarios no son predecibles con datos históricos.")

# --- Sección 3: SHAP ---
st.subheader("¿Qué variables explican la predicción?")

explainer = shap.Explainer(modelo)
shap_values = explainer(ultima_fila)

fig2, ax2 = plt.subplots(figsize=(8, 4))
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
st.pyplot(fig2)