from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from Codigo.entrenamiento import entrenar_modelo

# Entrenar el modelo
modelo = entrenar_modelo()

# Definir la aplicación FastAPI
app = FastAPI()

# Definir el esquema de entrada para la API
class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day_of_week: int
    month: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# Definir la ruta para la predicción
@app.post("/predict/")
async def predict(data: InputData):
    # Convertir los datos de entrada en un DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Realizar la predicción
    prediction = modelo.predict(input_data)
    
    # Devolver el resultado
    return {"prediction": "Suscribirá" if prediction[0] == 'yes' else "No suscribirá"}

# Iniciar la aplicación (esto se debe ejecutar en la terminal, no en el script)
# uvicorn main:app --reload
