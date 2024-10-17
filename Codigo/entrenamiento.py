import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo

def entrenar_modelo():
    # Cargar los datos
    df = fetch_ucirepo(id=222)
    X = df.data.features
    y = df.data.targets

    # Definir variables categóricas y numéricas
    variables_categoricas = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome']
    variables_numericas = ['age', 'balance', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocesamiento
    preprocesamiento = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), variables_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), variables_categoricas)
        ])

    # Crear el pipeline con preprocesamiento y modelo
    pipeline = Pipeline(steps=[
        ('preprocesamiento', preprocesamiento),
        ('modelo', LogisticRegression(max_iter=1000))
    ])

    # Entrenar el modelo
    pipeline.fit(X_entrenamiento, y_entrenamiento)

    return pipeline
