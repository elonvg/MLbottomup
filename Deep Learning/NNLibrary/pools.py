import numpy as np


class MaxPool():
    def __init__(self, size=2, stride=2):
        self.size = size # Assumes n x n pooling
        self.stride = stride # Assumes same stride in all directions
    
    def forward(self, x):
        self.x = x # Shape: [batch_size, x_channels, x_rows, x_cols]
        batch_size, x_channels, x_rows, x_cols = x.shape

        fm_rows = (x_rows - self.size) // self.stride + 1
        fm_cols = (x_cols - self.size) // self.stride + 1
        feature_maps = np.zeros([batch_size, x_channels, fm_rows, fm_cols])

        for n in range(batch_size):
            for k in range(x_channels):
                for i in range(fm_rows):
                    for j in range(fm_cols):
                        row_start = i * self.stride
                        col_start = j * self.stride

                        window = x[n, k, row_start:row_start+self.size, col_start:col_start+self.size]  
                        feature_maps[n, k, i, j] = np.max(window)

        return feature_maps # Shape: [batch_size, x_channels, fm_rows, fm_cols]
    
    def backward(self, r_grad):
        local_grad = np.zeros_like(self.x)
        batch_size, fm_channels, fm_rows, fm_cols = r_grad.shape

        for n in range(batch_size):
            for k in range(fm_channels):
                for i in range(fm_rows):
                    for j in range(fm_cols):
                        row_start = i * self.stride
                        col_start = j * self.stride

                        window = self.x[n, k, row_start:row_start+self.size, col_start:col_start+self.size]
                        local_grad[
                            n, k, row_start:row_start+self.size, col_start:col_start+self.size
                            ] += np.where(window == np.max(window), 1, 0) * r_grad[n, k, i, j]

        return local_grad # Shape: [batch_size, x_channels, x_rows, x_cols]
    
class Flatten():
    def forward(self, x):
        self.input_size = x.shape
        batch_size = x.shape[0]
        return x.reshape([batch_size, -1])
    
    def backward(self, r_grad):
        return r_grad.reshape(self.input_size)

