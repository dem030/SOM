from som import som
from utils import  plot_quantizaion_error, plot_topology
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



df = pd.read_csv('data/iris.csv', header=None)
data= df.iloc[:,0:4].values


train_data,test_data = train_test_split(data, test_size=0.2,random_state=42)
som_model = som(grid_height=10,grid_width=10, input_dim=train_data.shape[1], learning_rate=0.1,rad=5, epoc=50)
print("inizio addestramento")
som_model.train(train_data, save_to="som_weights.npy")
print("addestramento completato")
print("pesi salvati in {save_to}")

train_error = som_model.get_train_errors(train_data)
test_error = som_model.get_test_errors(test_data)

print("errore medio training set: {train_error: .4f}")
print("errore medio test set: {test_error: .4f}")


plot_quantizaion_error([train_error], [test_error])

plot_topology(som_model.weights)

print("mappa topologica creata")
