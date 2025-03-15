import pandas as pd
from sklearn.model_selection import train_test_split
from utils import remove_outliers, normalize_data

SEED = 42  # Semilla para reproducibilidad

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Realiza el preprocesamiento de los datos (limpieza, codificación y escalado)."""
    # Eliminar columnas irrelevantes
    df.drop(columns=["Posted On", "Point of Contact"], inplace=True, errors='ignore')

    # Separar la información de "Floor"
    df[['Current Floor', 'Total Floors']] = df['Floor'].str.extract(r'(\d+|Ground)\D+(\d+)?')
    df['Current Floor'] = df['Current Floor'].replace('Ground', 0).astype(float)
    df['Total Floors'] = df['Total Floors'].astype(float)
    df.drop(columns=['Floor'], inplace=True, errors='ignore')

    # Aplicar One-Hot Encoding a columnas con pocas categorías
    one_hot_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Codificación por frecuencia para 'Area Locality'
    if 'Area Locality' in df.columns:
        locality_counts = df['Area Locality'].value_counts()
        df['Area Locality Encoded'] = df['Area Locality'].map(locality_counts)
        df.drop(columns=['Area Locality'], inplace=True)

    # Manejo de valores nulos
    df.dropna(inplace=True)

    # Escalado de variables numéricas
    numerical_columns = ['BHK', 'Size', 'Bathroom', 'Area Locality Encoded']
    df[numerical_columns] = normalize_data(df[numerical_columns])

    # Filtrar outliers en Rent
    df = remove_outliers(df, 'Rent')

    return df

def split_data(df):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=['Rent'])
    y = df['Rent']
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

if __name__ == "__main__":
    df = load_data("dataset.csv")
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Datos preparados y divididos correctamente.")
