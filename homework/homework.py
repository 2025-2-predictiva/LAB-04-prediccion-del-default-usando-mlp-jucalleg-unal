# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
from glob import glob
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Configuración

class Config:
    
    BASE_DIR: Path = Path("files")
    INPUT_DIR: Path = BASE_DIR / "input"
    MODEL_DIR: Path = BASE_DIR / "models"
    OUTPUT_DIR: Path = BASE_DIR / "output"

    TRAIN_FILE: Path = INPUT_DIR / "train_data.csv.zip"
    TEST_FILE: Path = INPUT_DIR / "test_data.csv.zip"
    MODEL_FILE: Path = MODEL_DIR / "model.pkl.gz"
    METRICS_FILE: Path = OUTPUT_DIR / "metrics.json"

    CAT_COLS: List[str] = ["SEX", "EDUCATION", "MARRIAGE"]
    TARGET_COL: str = "default"

    # Hiperparámetros 
    GRID_PARAMS: Dict[str, Iterable[Any]] = {
        "selector__k": [20],
        "pca__n_components": [None],
        "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp__alpha": [0.26],
        "mlp__learning_rate_init": [0.001],
    }



# Funciones 

def limpiar_datos(path_zip: Path) -> pd.DataFrame:
    
    df = pd.read_csv(path_zip, compression="zip").copy()
    df.rename(columns={"default payment next month": Config.TARGET_COL}, inplace=True)

   
    df.drop(columns=[c for c in ["ID"] if c in df.columns], inplace=True, errors="ignore")

    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)].copy()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)

    return df.dropna()


def calcular_metricas(nombre: str, y_true, y_pred) -> Dict[str, Any]:
    
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def matriz_confusion(nombre: str, y_true, y_pred) -> Dict[str, Any]:
   
    tn, fp, fn, tp = sklearn_confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def limpiar_directorio(ruta: Path) -> None:
   
    if ruta.exists():
        for f in glob(str(ruta / "*")):
            os.remove(f)
        try:
            os.rmdir(ruta)
        except OSError:
            pass
    ruta.mkdir(parents=True, exist_ok=True)


def guardar_modelo(modelo: GridSearchCV, ruta: Path) -> None:
    
    with gzip.open(ruta, "wb") as fh:
        pickle.dump(modelo, fh)


def guardar_metricas(registros: List[Dict[str, Any]], ruta: Path) -> None:
 
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r) + "\n")



# Modelado

def construir_busqueda(vars_cat: List[str], vars_num: List[str]) -> GridSearchCV:
    """
    - OneHotEncoder (categóricas)
    - StandardScaler (numéricas)
    - PCA
    - SelectKBest
    - MLPClassifier 
    """
    preprocesamiento = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), vars_cat),
            ("num", StandardScaler(), vars_num),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("pre", preprocesamiento),
            ("selector", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("mlp", MLPClassifier(max_iter=15000, random_state=21)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=Config.GRID_PARAMS,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )



# Script principal

def main() -> None:
    # --- Carga y limpieza ---
    df_train = limpiar_datos(Config.TRAIN_FILE)
    df_test = limpiar_datos(Config.TEST_FILE)

    # --- Paso 2: split ---
    X_train = df_train.drop(columns=[Config.TARGET_COL])
    y_train = df_train[Config.TARGET_COL]
    X_test = df_test.drop(columns=[Config.TARGET_COL])
    y_test = df_test[Config.TARGET_COL]

    cat_cols = Config.CAT_COLS
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # --- Paso 3-4: pipeline + gridsearch ---
    busqueda = construir_busqueda(cat_cols, num_cols)
    busqueda.fit(X_train, y_train)

    # --- Paso 5: guardar modelo ---
    limpiar_directorio(Config.MODEL_DIR)
    guardar_modelo(busqueda, Config.MODEL_FILE)

    # --- Paso 6-7: métricas y matrices ---
    y_train_pred = busqueda.predict(X_train)
    y_test_pred = busqueda.predict(X_test)

    resultados = [
        calcular_metricas("train", y_train, y_train_pred),
        calcular_metricas("test", y_test, y_test_pred),
        matriz_confusion("train", y_train, y_train_pred),
        matriz_confusion("test", y_test, y_test_pred),
    ]
    guardar_metricas(resultados, Config.METRICS_FILE)


if __name__ == "__main__":
    main()