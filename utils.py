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
    plt.figures(figsize=(10,6))
    plt.plot(train_errors, label='Train Quantization Error')
    plt.plot(test_errors, label='Test Quantization Error')
    plt.xlabel('Epochs')
    plt.ylabel('Quantization Error')
    plt.title('Quantization Error over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_topology(weights):
    h, w, dim = weights.shape
    #vettorializza i neuroni
    flat= weights.reshape (h * w, dim)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flat)
    x = reduced[:, 0].reshape(h, w)
    y = reduced[:, 1].reshape(h, w)

    plt.figure(figsize=(8,6))
    plt.title("mappa topologica SOM attraverso PCA")
    plt.scatter(x,y, c='blue')
    plt.plot(x,y, 'k-', alpha=0.3)
    plt.plot(x.T,y.T,'k-', alpha=0.3)
    plt.show()