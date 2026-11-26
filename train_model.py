# Primera version: Entrenamiento y evaluacion del modelo Iris

import joblib

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train():

    # 1. Cargar el dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Entrenar el algoritmo de clasificacion
    clf = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # 4. Evaluar la precision del modelo
    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("\n========== RESULTADOS DEL MODELO ==========")

    print(f"\nExactitud en prueba: {acc * 100:.2f}%")

    # 5. Mostrar reporte de clasificacion
    print("\nReporte de clasificacion:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        )
    )

    # 6. Exportar el modelo y los nombres de las clases
    artifact = {
        "model": clf,
        "target_names": iris.target_names.tolist()
    }

    joblib.dump(artifact, "iris_model.pkl")

    print("Modelo entrenado y guardado correctamente.")
    print("Archivo generado: iris_model.pkl")


if __name__ == "__main__":
    train()