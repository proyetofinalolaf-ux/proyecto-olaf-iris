from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import FileResponse
from pathlib import Path

# Crear la aplicación
app = FastAPI(title="Proyecto Olaf - Modelo Iris")

# Cargar el modelo entrenado
artifact = joblib.load("iris_model.pkl")
model = artifact["model"]
target_names = artifact["target_names"]

# Definir los datos que recibirá la API
class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Ruta principal
@app.get("/")
def home():
    return {
        "mensaje": "Bienvenido a Proyecto Olaf",
        "estado": "API funcionando"
    }

# Ruta para realizar predicciones
@app.post("/predict")
def predict(flower: Flower):
    data = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    prediction = model.predict(data)[0]

    return {
        "prediccion": target_names[int(prediction)]
    }

@app.get("/web")
def pagina_web():
    archivo = Path(__file__).parent / "index.html"
    return FileResponse(archivo)