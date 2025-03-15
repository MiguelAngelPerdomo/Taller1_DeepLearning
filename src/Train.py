import data_loader
import model as mdl
import tensorflow as tf

def train_model(data_path):
    """Entrena el modelo con los datos proporcionados."""
    df = data_loader.load_data(data_path)
    df = data_loader.preprocess_data(df)
    X_train, X_test, y_train, y_test = data_loader.split_data(df)

    model_instance = mdl.build_model(X_train.shape[1])

    history = model_instance.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=16, 
        validation_data=(X_test, y_test), 
        verbose=1
    )

    model_instance.save('model.h5')
    print("Entrenamiento completado y modelo guardado como 'model.h5'")

if __name__ == "__main__":
    train_model("dataset.csv")
