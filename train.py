import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import sys

# Odczyt parametrów z argumentów skryptu
# Użycie: python 05lab_train.py 10 32
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

print(f"Trenowanie modelu: {epochs} epok, batch size: {batch_size}")

# Wczytaj dane
train_df = pd.read_csv("data/diamonds_train.csv")
valid_df = pd.read_csv("data/diamonds_valid.csv")

X_train = train_df.drop(columns=["price"]).astype(np.float32)
y_train = train_df["price"].astype(np.float32)
X_valid = valid_df.drop(columns=["price"]).astype(np.float32)
y_valid = valid_df["price"].astype(np.float32)

# Model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

model.save("model.h5")
print("Model zapisany jako model.h5")
