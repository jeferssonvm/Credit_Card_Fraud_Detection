import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def train_model(data_dir, model_dir):
    # 1. Carga de datos
    x_train = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    # 2. Cálculo del ratio para scale_pos_weight
    counts = y_train.value_counts()
    ratio = counts[0] / counts[1]
    print(f"Ratio de desbalance calculado: {ratio:.2f}") 

    # 3. Configuración de Hiperparámetros para búsqueda aleatoria
    parametros = {
        'n_estimators': [100, 300, 500, 800, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.6, 0.8, 1.0],
        'scale_pos_weight': [ratio]
    } 

    # 4. Búsqueda de mejores parámetros (RandomizedSearch)
    print("Iniciando búsqueda de hiperparámetros...")
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        ),
        param_distributions=parametros,
        n_iter=20, 
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # Ajuste del modelo utilizando validación cruzada 
    random_search.fit(x_train, y_train.values.ravel()) 
    
    best_model = random_search.best_estimator_
    print(f"Mejores parámetros encontrados: {random_search.best_params_}") 

    # 5. Evaluación rápida en consola
    y_pred = best_model.predict(x_test)
    print("\n--- Reporte de Clasificación (Umbral por defecto 0.5) ---")
    print(classification_report(y_test, y_pred)) 

    # 6. PERSISTENCIA: Guardar el modelo en la carpeta models/
    # Se guarda el modelo para que predict.py pueda cargarlo sin re-entrenar
    model_path_joblib = os.path.join(model_dir, 'xgboost_fraud_model.joblib')
    joblib.dump(best_model, model_path_joblib) 
    
    # También se guarda en formato nativo de XGBoost (.json)
    model_path_json = os.path.join(model_dir, 'xgboost_fraud_model.json')
    best_model.save_model(model_path_json) 
    
    print(f"Modelo guardado exitosamente en: {model_dir}")

if __name__ == "__main__":

    data_dir = '../data/processed/'
    model_dir = '../models/'

    # Asegurar que la carpeta de modelos exista antes de intentar guardar 
    os.makedirs(model_dir, exist_ok=True) 

    train_model(data_dir, model_dir)