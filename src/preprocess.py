import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

def preprocess_data(input_path, output_dir):
    df = pd.read_csv(input_path)

    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.lower()

    x = df.drop('class', axis=1)
    y = df['class']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    x_train[['time', 'amount']] = scaler.fit_transform(x_train[['time', 'amount']])
    x_test[['time', 'amount']] = scaler.transform(x_test[['time', 'amount']])

    x_train.to_csv(f'{output_dir}x_train.csv', index=False)
    x_test.to_csv(f'{output_dir}x_test.csv', index=False)
    y_train.to_csv(f'{output_dir}y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}y_test.csv', index=False)

    joblib.dump(scaler, '../models/robust_scaler.pkl')
    
    print("¡Preprocesamiento completado! Datos en /processed/ y escalador en /models/.")

if __name__ == "__main__":
    preprocess_data('../data/raw/creditcard.csv', '../data/processed/')