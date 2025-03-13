import pandas as pd 

def cargar_datos(ruta_archivo):
    """
    Carga datos desde un archivo CSV y retorna un DataFrame. 

    Args: 
        ruta_archivo (str): Ruta al archivo CSV de datos.

    Returns: 
        pandas.DataFrame: DataFrame con los datos cargados o None si hay error.
    """
    try: 
        # Cargar el archivo CSV
        df = pd.read_csv(ruta_archivo)
        print(f"Datos cargados exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        return None  # Retorna None en caso de error

# Llamar a la función y asignar el resultado a df
df = cargar_datos(r"C:/Taller_1/archive/House_Rent_Dataset.csv")

# Verifica si df se cargó correctamente antes de imprimir
if df is not None:
    print(df.head())  # Muestra las primeras 5 filas
else:
    print("No se pudo cargar el DataFrame.")
