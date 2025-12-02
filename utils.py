import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def save_weights(weights, filename = 'som_weights.npy'):
    np.save(filename, weights)
    print(f"pesi salvati in {filename}")


def load_weights(filename = 'som_weights.npy'):
    weights = np.load(filename)
    print(f"pesi caricati da {filename}")
    return weights

def plot_quantizaion_error(train_errors,test_errors):
    

