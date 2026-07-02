import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Layer:
    def __init__(self):
        self.cache = {}
        self.record = True

    def record_cache(self, key, var):
        if self.record:
            self.cache[key] = var

    def paras_grads(self):
        return []

class LinearLayer(Layer):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.normal(scale=0.01, size=[in_dim, out_dim]).astype(np.float32)
        self.B = np.zeros(out_dim).astype(np.float32)

        self.dW = np.zeros_like(self.W, dtype=np.float32)
        self.dB = np.zeros_like(self.B, dtype=np.float32)

        self.cache = {}
        self.record = True

    def forward(self, x):
        self.record_cache('x', x) # Shape: [batch_size, in_dim]
        out = x @ self.W + self.B # Shape: [batch_size, out_dim]
        return out
    
    def backward(self, r_grad):
        x = self.cache['x']
        self.dW = x.T @ r_grad # Shape: [in_dim, out_dim]
        self.dB = np.sum(r_grad, axis=0) # Sum over all batches
        return r_grad @ self.W.T # Shape: [batch_size, in_dim]
    
    def paras_grads(self):
        return [
            (self.W, self.dW),
            (self.B, self.dB),
        ]
    
    def count_parameters(self):
        return self.W.size + self.B.size


class ConvLayer(Layer):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.kernel = np.random.normal(scale=0.01, size=[out_channels, in_channels, kernel_size, kernel_size]).astype(np.float32) # Assumes n x n kernel
        self.bias = np.zeros(out_channels).astype(np.float32)

        self.dK_prev = 0
        self.dB_prev = 0

        self.cache = {}
        self.record = True
    
    def forward(self, x):
        self.record_cache('x', x) # Shape: [batch_size, x_channels, x_rows, x_cols]

        windows = sliding_window_view(x, window_shape=(self.kernel_size, self.kernel_size), axis=(2, 3))

        feature_maps = np.tensordot(windows, self.kernel, axes=([1, 4, 5], [1, 2, 3])) # Multiply and sum over kernel and window for all in_channels
        self.feature_maps = np.transpose(feature_maps, (0, 3, 1, 2)) # Transpose to get right shape

        return self.feature_maps + self.bias[None, :, None, None] # Shape: [batch_size, out_channels, fm_rows, fm_cols]
    
    def backward(self, r_grad):
        x = self.cache['x']

        _, _, fm_rows, fm_cols = r_grad.shape

        local_grad = np.zeros_like(x, dtype=np.float32)

        windows = sliding_window_view(x, window_shape=(fm_rows, fm_cols), axis=(2, 3))
        self.dK = np.transpose(np.tensordot(windows, r_grad, axes=([0, 4, 5], [0, 2, 3])), (3, 0, 1, 2)) # Shape: [out_channels, in_channels, kernel_size, kernel_size]
        
        self.dB = np.sum(r_grad, axis=(0, 2, 3)) # Shape: [out_channels]

        for i in range(fm_rows):
            for j in range(fm_cols):
                local_grad[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += np.transpose(np.tensordot(self.kernel, r_grad[:, :, i, j], axes=([0], [1])), (3, 0, 1, 2))

        return local_grad# Shape: [batch_size, x_channeles, x_rows, x_cols]
    
    def paras_grads(self):
        return [
            (self.kernel, self.dK),
            (self.bias, self.dB),
        ]

    def count_parameters(self):
        return self.kernel.size + self.bias.size

