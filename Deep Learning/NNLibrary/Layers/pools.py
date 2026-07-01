import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from NNLibrary.Layers.learnable import Layer

class MaxPool(Layer):
    def __init__(self, size=2, stride=None):
        self.size = size # Assumes n x n pooling
        self.stride = size if stride == None else stride # Assumes same stride in all directions
    
    def forward(self, x):
        self.x = x # Shape: [batch_size, x_channels, x_rows, x_cols]

        windows = sliding_window_view(x, window_shape=(self.size, self.size), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :] # Stride
        
        feature_maps = np.amax(windows, axis=(4, 5))

        self.max_mask = windows == feature_maps[:, :, :, :, None, None] # Max mask for backprop

        return feature_maps # Shape: [batch_size, x_channels, fm_rows, fm_cols]
    
    def backward(self, r_grad):

        batch_size, fm_channels, fm_rows, fm_cols = r_grad.shape

        local_grad = np.zeros_like(self.x, dtype=np.float32)

        grad_windows = self.max_mask * r_grad[:, :, :, :, None, None]

        for i in range(fm_rows):
            for j in range(fm_cols):
                row_start = i * self.stride
                col_start = j * self.stride
                
                local_grad[:, :, row_start:row_start+self.size, col_start:col_start+self.size] += grad_windows[:, :, i, j, :, :]

        return local_grad# Shape: [batch_size, x_channeles, x_rows, x_cols]
    
class Flatten(Layer):
    def forward(self, x):
        self.input_size = x.shape
        batch_size = x.shape[0]
        return x.reshape([batch_size, -1])
    
    def backward(self, r_grad):
        return r_grad.reshape(self.input_size)

