import pandas as pd
import tensorflow as tf

# Cargar modelo
RUTA_MODELO = '../models/modelo.h5'



model = tf.load_model(RUTA_MODELO)

predict = model.predict()

