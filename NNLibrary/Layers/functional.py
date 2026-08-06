import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from NNLibrary.Layers.learnable import Layer

class MaxPool(Layer):
    def __init__(self, size=2, stride=None):
        self.size = size # Assumes n x n pooling
        self.stride = size if stride == None else stride # Assumes same stride in all directions
    
        self.cache = {}
        self.record = True

    def forward(self, x):
        self.record_cache('x', x) # Shape: [batch_size, x_channels, x_rows, x_cols]

        windows = sliding_window_view(x, window_shape=(self.size, self.size), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :] # Stride
        
        feature_maps = np.amax(windows, axis=(4, 5))

        max_mask = windows == feature_maps[:, :, :, :, None, None] # Max mask for backprop
        self.record_cache('max_mask', max_mask)

        return feature_maps # Shape: [batch_size, x_channels, fm_rows, fm_cols]
    
    def backward(self, r_grad):
        x = self.cache['x']

        _, _, fm_rows, fm_cols = r_grad.shape

        local_grad = np.zeros_like(x, dtype=np.float32)

        max_mask = self.cache['max_mask']
        grad_windows = max_mask * r_grad[:, :, :, :, None, None]

        for i in range(fm_rows):
            for j in range(fm_cols):
                row_start = i * self.stride
                col_start = j * self.stride
                
                local_grad[:, :, row_start:row_start+self.size, col_start:col_start+self.size] += grad_windows[:, :, i, j, :, :]

        return local_grad # Shape: [batch_size, x_channeles, x_rows, x_cols]

class MaxUnPool(Layer):
    def __init__(self, size=2, stride=None):
        self.size = size
        self.stride = size if stride == None else stride

        self.cache = {}
        self.record = True

    def forward(self, x):
        # TODO: Write forward
        pass

    def backward(self, r_grad):
        # TODO: Write backprop
        pass


class Upsample(Layer):
    def __init__(self, scale=2):
        self.scale = scale

    def forward(self, x):
        # x shape: [batch_size, n_channels, n_rows, n_cols]

        batch_size, n_channels, n_rows, n_cols = x.shape
        out_rows, out_cols = self.scale * n_rows, self.scale * n_cols

        out = np.zeros([batch_size, n_channels, out_rows, out_cols], dtype=np.float32)

        for c in range(n_channels):
            for i in range(n_rows):
                for j in range(n_cols):
                    fill = np.zeros([self.scale, self.scale], dtype=np.float32) + x[:, c, i, j]
                    out[:, c, i * self.scale : i * self.scale + self.scale, j * self.scale : j * self.scale + self.scale] = fill

        return out

    def backward(self, r_grad):
        # TODO: Write backprop
        pass


    
class Flatten(Layer):
    def __init__(self):
        self.cache = {}
        self.record = True

    def forward(self, x):
        self.record_cache('input_size', x.shape)
        batch_size = x.shape[0]
        return x.reshape([batch_size, -1])
    
    def backward(self, r_grad):
        input_size = self.cache['input_size']
        return r_grad.reshape(input_size)

class Dropout(Layer):
    def __init__(self, p):
        self.p = p

        self.cache = {}
        self.record = True

    def forward(self, x):
        if self.record:
            mask = np.where(np.random.rand(*x.shape).astype(np.float32) > self.p, 1, 0) / (1 - self.p)
            self.record_cache('mask', mask)
            return mask * x
        return x
    
    def backward(self, r_grad):
        mask = self.cache['mask']
        return mask * r_grad
