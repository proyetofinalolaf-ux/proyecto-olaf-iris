from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from fastapi.responses import FileResponse
from pathlib import Path

# Directorio del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Crear la aplicaciÃ³n
app = FastAPI(title="Proyecto Olaf - Modelo Iris")

# Cargar el modelo entrenado
artifact = joblib.load(BASE_DIR / "iris_model.pkl")
model = artifact["model"]
target_names = artifact["target_names"]

# Datos que recibirÃ¡ la API
class Flower(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# Ruta principal: mostrar interfaz web
@app.get("/")
def home():
    archivo = BASE_DIR / "index.html"
    return FileResponse(archivo)

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

# PÃ¡gina web
@app.get("/web")
def pagina_web():
    archivo = BASE_DIR / "index.html"
    return FileResponse(archivo)

