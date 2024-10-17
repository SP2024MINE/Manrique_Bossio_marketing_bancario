import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Definir variables categóricas y numéricas
variables_categoricas = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome']
variables_numericas = ['age', 'balance', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous']

# Preprocesamiento: Escalado de variables numéricas y codificación de variables categóricas
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

def entrenar_modelo(X, y):
    pipeline.fit(X, y)
    return pipeline

def predecir(X_nuevo):
    return pipeline.predict(X_nuevo)
