import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Wczytaj model
model = tf.keras.models.load_model("model.h5", compile=False)

# Wczytaj dane testowe
test_df = pd.read_csv("data/diamonds_test.csv")
X_test = test_df.drop(columns=["price"]).astype(np.float32)
y_true = test_df["price"].astype(np.float32)

# Wykonaj predykcje
predictions = model.predict(X_test).flatten()

# Oblicz metryki
rmse = np.sqrt(mean_squared_error(y_true, predictions))
mae = mean_absolute_error(y_true, predictions)

# Zapisz predykcje do pliku
output_df = pd.DataFrame(predictions, columns=["Predicted_Price"])
output_df.to_csv("predictions.csv", index=False)
print("Predykcje zapisane do predictions.csv")

# Zapisz metryki do pliku tekstowego
with open("metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.close()
    
print("Metryki zapisane do metrics.txt")