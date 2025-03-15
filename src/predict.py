import tensorflow as tf
import pandas as pd
import data_loader

def predict_rent(model_path, new_data_path, output_path="predictions.csv"):
    """Realiza predicciones con el modelo entrenado."""
    df = data_loader.load_data(new_data_path)
    df = data_loader.preprocess_data(df)
    X_new = df.values

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_new)

    df['Predicted Rent'] = y_pred
    df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    predict_rent("model.h5", "new_data.csv")
