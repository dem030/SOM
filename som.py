import numpy as np
import pandas as pd  # Aggiungi questo import

class som:
    def __init__(self, grid_height, grid_width, input_dim, learning_rate, rad, epoc):
        self.height = grid_height
        self.width = grid_width
        self.input_dim = input_dim 
        self.weights = np.random.rand(grid_height, grid_width, input_dim)
        self.initial_lr = learning_rate
        self.initial_rad = rad
        self.epochs = epoc
    
    def find_bmu(self, input_vector):
        min_dist = float('inf')
        bmu_idx = (0, 0)
        for i in range(self.height):
            for j in range(self.width):
                w = self.weights[i][j]
                dist = np.linalg.norm(input_vector - w)
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx, min_dist
    
    def decay_learning_rate(self, epoch):
        return self.initial_lr * np.exp(-epoch / self.epochs)
    
    def decay_radius(self, epoch):
        return self.initial_rad * np.exp(-epoch / self.epochs)
    
    def weights_updated(self, input_vector, bmu_idx, learning_rate, radius):
        for i in range(self.height):
            for j in range(self.width):
                dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if dist_to_bmu <= radius:
                    influence = np.exp(-dist_to_bmu**2 / (2 * (radius**2)))
                    self.weights[i][j] += learning_rate * influence * (input_vector - self.weights[i][j])
                    
    def train(self, train_data, save_to='som_weights.csv'):
        train_errors = []
        for epoch in range(self.epochs):
            lr = self.decay_learning_rate(epoch)
            rad = self.decay_radius(epoch)
            for input_vector in train_data:
                bmu_coord, _ = self.find_bmu(input_vector)
                self.weights_updated(input_vector, bmu_coord, lr, rad)
            
            # Calcola errore alla fine di ogni epoca
            epoch_error = self.quantization_error(train_data)
            train_errors.append(epoch_error)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoca {epoch + 1}/{self.epochs}, Errore: {epoch_error:.4f}")
        
        # Salva in CSV
        self.save_weights_csv(save_to)
        print(f"training completato, file salvati in {save_to}")
        return train_errors
    
    def save_weights_csv(self, filename='som_weights.csv'):
        """Salva i pesi in formato CSV"""
        # Reshape dei pesi: da (height, width, input_dim) a (height*width, input_dim)
        h, w, d = self.weights.shape
        weights_2d = self.weights.reshape(h * w, d)
        
        # Crea DataFrame con coordinate e pesi
        rows = []
        for i in range(h):
            for j in range(w):
                row = {'row': i, 'col': j}
                for dim in range(d):
                    row[f'weight_{dim}'] = self.weights[i, j, dim]
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Pesi salvati in formato CSV: {filename}")
    
    def load_weights_csv(self, filename='som_weights.csv'):
        """Carica i pesi da un file CSV"""
        df = pd.read_csv(filename)
        
        # Ricostruisci la struttura 3D dei pesi
        h = df['row'].max() + 1
        w = df['col'].max() + 1
        d = len([col for col in df.columns if col.startswith('weight_')])
        
        self.weights = np.zeros((h, w, d))
        for _, row in df.iterrows():
            i = int(row['row'])
            j = int(row['col'])
            for dim in range(d):
                self.weights[i, j, dim] = row[f'weight_{dim}']
        
        print(f"Pesi caricati da {filename}")
        return self.weights

    def quantization_error(self, data):
        tot = 0
        for input_vector in data:
            _, dist = self.find_bmu(input_vector)
            tot += dist
        return tot / len(data)