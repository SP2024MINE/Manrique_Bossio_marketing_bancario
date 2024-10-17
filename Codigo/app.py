# app.py
import streamlit as st
import pandas as pd
from entrenamiento import entrenar_modelo
from ucimlrepo import fetch_ucirepo

modelo = entrenar_modelo()

st.title("Predicción de Suscripción prestamos bancarios")
st.write("Ingrese la información del cliente:")

age = st.number_input("Edad", min_value=18, max_value=100, value=30)
job = st.selectbox("Trabajo", options=["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
                                        "retired", "self-employed", "student", "technician", "services", 
                                        "unemployed", "unknown"])
marital = st.selectbox("Estado Civil", options=["divorced", "married", "single"])
education = st.selectbox("Educación", options=["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("¿Tiene crédito?", options=["yes", "no"])
balance = st.number_input("Balance Promedio Anual (en euros)", min_value=-1000, value=0)
housing = st.selectbox("¿Tiene préstamo hipotecario?", options=["yes", "no"])
loan = st.selectbox("¿Tiene préstamo personal?", options=["yes", "no"])
contact = st.selectbox("Tipo de Contacto", options=["cellular", "telephone", "unknown"])
day_of_week = st.number_input("Día de la Semana (1-7)", min_value=1, max_value=7, value=1)
month = st.selectbox("Mes del Último Contacto", options=["jan", "feb", "mar", "apr", "may", "jun", 
                                                          "jul", "aug", "sep", "oct", "nov", "dec"])
duration = st.number_input("Duración de Último Contacto (en segundos)", min_value=0, value=0)
campaign = st.number_input("Número de Contactos en Esta Campaña", min_value=1, value=1)
pdays = st.number_input("Días desde el Último Contacto en Campañas Previas (-1 si nunca fue contactado)", min_value=-1, value=-1)
previous = st.number_input("Número de Contactos Anteriores", min_value=0, value=0)
poutcome = st.selectbox("Resultado de la Campaña Anterior", options=["unknown", "other", "failure", "success"])

def predecir(input_data):
    return modelo.predict(input_data)

if st.button("Predecir"):
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'month': [month],
        'balance':[balance]
    })

    prediction = predecir(input_data)
    st.write("Predicción: ", "Suscribirá" if prediction[0] == 'yes' else "No suscribirá")
