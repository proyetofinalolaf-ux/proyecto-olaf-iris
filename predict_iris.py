# Prediccion de especies de flores Iris

import joblib


# 1. Cargar el modelo entrenado
artifact = joblib.load("iris_model.pkl")

model = artifact["model"]
target_names = artifact["target_names"]


# 2. Crear datos de una flor nueva
# Valores: largo y ancho del sepalo,
# largo y ancho del petalo (en centimetros)

new_flower = [[5.1, 3.5, 1.4, 0.2]]


# 3. Realizar la prediccion
prediction = model.predict(new_flower)


# 4. Obtener el nombre de la especie
predicted_species = target_names[prediction[0]]


# 5. Mostrar el resultado
print("\n========== PREDICCION IRIS ==========")

print("Datos de la flor:", new_flower[0])

print("Especie predicha:", predicted_species)

print("=====================================")