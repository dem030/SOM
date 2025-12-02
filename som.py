import numpy as np
class som:
    def __init__(self, grid_height, grid_width, input_dim, learning_rate, rad, epoc):
        self.height = grid_height
        self.width = grid_width
        self.input_dim = input_dim 
        self.weights = np.random.rand(grid_height,grid_width,input_dim)
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
    
    def weights_updated( self, input_vector, bmu_idx, learning_rate, radius):
        for i in range(self.height):
            for j in range(self.width):
                dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if dist_to_bmu <= radius:
                    influence = np.exp(-dist_to_bmu**2 / (2 * (radius**2)))
                    self.weights[i][j] += learning_rate * influence * (input_vector - self.weights[i][j])
                    
    def train(self, train_data, save_to ='som_weights.npy'):
        for epoch in range(self.epochs):
            lr = self.decay_learning_rate(epoch)
            rad = self.decay_radius(epoch)
            for input_vector in train_data:
                bmu_coord, _ = self.find_bmu(self, input_vector)
                self.weights_updated(input_vector, bmu_coord, lr, rad)
        np.save(save_to, self.weights)
        print("training completato, file salvati in {save_to}")
    

    def quantization_error(self, data):
        tot = 0
        for input_vector in data:
            _, dist = self.find_bmu(input_vector)
            tot += dist
        return tot / len(data)


            
