import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def save_weights(weights, filename='som_weights.csv'):
    """Salva i pesi in formato CSV"""
    h, w, d = weights.shape
    
    rows = []
    for i in range(h):
        for j in range(w):
            row = {'row': i, 'col': j}
            for dim in range(d):
                row[f'weight_{dim}'] = weights[i, j, dim]
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"pesi salvati in {filename}")


def load_weights(filename='som_weights.csv'):
    """Carica i pesi da un file CSV"""
    df = pd.read_csv(filename)
    
    # Ricostruisci la struttura 3D dei pesi
    h = df['row'].max() + 1
    w = df['col'].max() + 1
    d = len([col for col in df.columns if col.startswith('weight_')])
    
    weights = np.zeros((h, w, d))
    for _, row in df.iterrows():
        i = int(row['row'])
        j = int(row['col'])
        for dim in range(d):
            weights[i, j, dim] = row[f'weight_{dim}']
    
    print(f"pesi caricati da {filename}")
    return weights

def plot_quantizaion_error(train_errors, test_errors, save_path='quantization_error.png'):
    """Plotta e salva il grafico degli errori di quantizzazione"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='Train Quantization Error', marker='o')
    plt.plot(test_errors, label='Test Quantization Error', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Quantization Error')
    plt.title('Quantization Error over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico errori salvato in {save_path}")
    plt.show()

def plot_topology(weights, save_path='topology_map.png'):
    """Plotta e salva la mappa topologica della SOM"""
    h, w, dim = weights.shape
    # Vettorializza i neuroni
    flat = weights.reshape(h * w, dim)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flat)
    x = reduced[:, 0].reshape(h, w)
    y = reduced[:, 1].reshape(h, w)

    plt.figure(figsize=(8, 6))
    plt.title("Mappa topologica SOM attraverso PCA")
    plt.scatter(x, y, c='blue', s=50)
    plt.plot(x, y, 'k-', alpha=0.3)
    plt.plot(x.T, y.T, 'k-', alpha=0.3)
    plt.xlabel('Prima Componente Principale')
    plt.ylabel('Seconda Componente Principale')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Mappa topologica salvata in {save_path}")
    plt.show()