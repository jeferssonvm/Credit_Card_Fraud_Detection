import joblib
import pandas as pd
import os

def make_prediction(new_data_path):
    # 1. Cargar el escalador y el modelo
    scaler = joblib.load('../models/robust_scaler.pkl') 
    model = joblib.load('../models/xgboost_fraud_model.joblib') 

    # 2. Cargar datos
    data = pd.read_csv(new_data_path)
    data.columns = data.columns.str.lower()

    # 3. LIMPIEZA CRÍTICA: Eliminar la columna 'class' si existe
    if 'class' in data.columns:
        data = data.drop(columns=['class'])  

    # 4. Aplicar el escalado robusto  
    data[['time', 'amount']] = scaler.transform(data[['time', 'amount']]) 

    # 5. Predicción  
    probs = model.predict_proba(data)[:, 1] 
    predictions = (probs >= 0.3).astype(int) 
    
    return predictions
if __name__ == "__main__":
    # Ajusta la indentación a 4 espacios exactos
    new_data_path = '../data/raw/creditcard.csv' 
    print("Iniciando proceso de predicción...")
    preds = make_prediction(new_data_path)
    print("Primeras 10 predicciones (0=Normal, 1=Fraude):")
    print(preds[:10])