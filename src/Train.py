import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Preprocesamiento
df.drop(['Posted On', 'Point of Contact'], axis=1, inplace=True)
df[['Floor', 'Total Floors']] = df['Floor'].str.extract(r'(\d+) out of (\d+)', expand=True).astype(float)
df.fillna({'Floor': 0, 'Total Floors': df['Total Floors'].median()}, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, columns=['Area Type', 'Furnishing Status', 'Tenant Preferred', 'City'], drop_first=True)

# Separar variables
X = df.drop('Rent', axis=1)
y = df['Rent']

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convertir a tensores para PyTorch
torch_X_train = torch.tensor(X_train, dtype=torch.float32)
torch_y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
torch_X_test = torch.tensor(X_test, dtype=torch.float32)
torch_y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Modelo PyTorch
class RentPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(RentPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_pytorch = RentPredictionModel(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model_pytorch.parameters(), lr=0.01)

# Entrenamiento PyTorch
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model_pytorch(torch_X_train)
    loss = criterion(outputs, torch_y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Modelo TensorFlow/Keras
model_tf = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mse')
model_tf.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
