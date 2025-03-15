import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    """Normaliza los datos usando Min-Max Scaling."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def inverse_transform(scaler, data):
    """Desescala los datos a su rango original."""
    return scaler.inverse_transform(data)

def remove_outliers(df, column):
    """Elimina los outliers de una columna usando el método del rango intercuartil (IQR)."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
