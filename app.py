from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import FileResponse
from pathlib import Path

# Directorio del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Crear la aplicación
app = FastAPI(title="Proyecto Olaf - Modelo Iris")

# Cargar el modelo entrenado
artifact = joblib.load(BASE_DIR / "iris_model.pkl")
model = artifact["model"]
target_names = artifact["target_names"]

# Datos que recibirá la API
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
@app.get("/healthz")
def health_check():
    return {
        "estado": "ok",
        "mensaje": "API funcionando correctamente"
    }

# Predicciones
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

# Página web
@app.get("/web")
def pagina_web():
    archivo = BASE_DIR / "index.html"
    return FileResponse(archivo)