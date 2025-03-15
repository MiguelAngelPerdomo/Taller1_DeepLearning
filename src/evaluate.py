import tensorflow as tf
import numpy as np
import pandas as pd
import data_loader

def evaluate_model(model_path, test_data_path):
    """Evalúa el modelo con datos de prueba."""
    df = data_loader.load_data(test_data_path)
    df = data_loader.preprocess_data(df)
    X_train, X_test, y_train, y_test = data_loader.split_data(df)

    model = tf.keras.models.load_model(model_path)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Loss (MSE): {loss}, MAE: {mae}")

if __name__ == "__main__":
    evaluate_model("../models/house_rent_model.h5", "test_dataset.csv")
