from som import som
from utils import plot_quantizaion_error, plot_topology
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Crea la cartella results se non esiste
os.makedirs('results', exist_ok=True)

save_to = "results/som_weights.csv"  # Salva anche i pesi nella cartella results
df = pd.read_csv('data/iris.data', header=None)
data = df.iloc[:, 0:4].values

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
som_model = som(grid_height=10, grid_width=10, input_dim=train_data.shape[1], learning_rate=0.02, rad=4, epoc=150)

print("inizio addestramento")
train_errors = som_model.train(train_data, save_to)
print("addestramento completato")
print(f"pesi salvati in {save_to}")

# Calcola errore finale sul test set
test_error = som_model.quantization_error(test_data)
# Crea una lista di errori test (costante) per il plot
test_errors = [test_error] * len(train_errors)

print(f"errore medio training set (finale): {train_errors[-1]:.4f}")
print(f"errore medio test set: {test_error:.4f}")

# Plot e salvataggio degli errori
plot_quantizaion_error(train_errors, test_errors, save_path='results/quantization_error.png')

# Plot e salvataggio della topologia
plot_topology(som_model.weights, save_path='results/topology_map.png')

print("Tutti i file sono stati salvati nella cartella 'results'")