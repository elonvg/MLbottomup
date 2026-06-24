import numpy as np

class LinearLayer():
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.normal(scale=0.01, size=[in_dim, out_dim])
        self.B = np.zeros(out_dim)

    def forward(self, x):
        self.x = x # Shape: [batch_size, in_dim]
        self.out = self.x @ self.W + self.B # Shape: [batch_size, out_dim]
        return self.out
    
    def backward(self, r_grad):
        self.dW = self.x.T @ r_grad # Shape: [in_dim, out_dim]
        self.dB = np.sum(r_grad, axis=0) # Sum over all batches
        return r_grad @ self.W.T # Shape: [batch_size, in_dim]
    
    def update(self, lr):
        self.W -= self.dW * lr
        self.B -= self.dB * lr

    def count_parameters(self):
        return self.W.size + self.B.size


class ConvLayer():
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.kernel = np.random.normal(size=[out_channels, in_channels, kernel_size, kernel_size]) # Assumes n x n kernel
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        self.x = x # Shape: [batch_size, x_channels, x_rows, x_cols]
        batch_size, x_channels, x_rows, x_cols = x.shape

        fm_rows, fm_cols = (x_rows - self.kernel_size) + 1, (x_cols - self.kernel_size) + 1
        self.feature_maps = np.zeros([batch_size, self.out_channels, fm_rows, fm_cols])

        self.windows = []

        for n in range(batch_size):
            for k in range(self.out_channels):
                for i in range(fm_rows):
                    for j in range(fm_cols):
                        window = x[n, :, i:i+self.kernel_size, j:j+self.kernel_size]
                        
                        self.feature_maps[n, k, i, j] = np.sum(self.kernel[k] * window) + self.bias[k]
                
        return self.feature_maps # Shape: [batch_size, out_channels, fm_rows, fm_cols]
    
    def backward(self, r_grad):

        batch_size, fm_channels, fm_rows, fm_cols = self.feature_maps.shape

        self.dK = np.zeros_like(self.kernel)
        local_grad = np.zeros_like(self.x, dtype=np.float64)

        for n in range(batch_size):
            for k in range(fm_channels):
                for i in range(fm_rows):
                    for j in range(fm_cols):
                        window = self.x[n, :, i:i+self.kernel_size, j:j+self.kernel_size]

                        self.dK[k] += window * r_grad[n, k, i, j] # Shape: [out_channels, in_channels, kernel_size, kernel_size]

                        local_grad[n, :, i:i+self.kernel_size, j:j+self.kernel_size] += self.kernel[k] * r_grad[n, k, i, j]
        
        self.dB = np.sum(r_grad, axis=(0, 2, 3)) # Shape: [out_channels]

        return local_grad # Shape: [batch_size, x_channeles, x_rows, x_cols]
    
    def update(self, lr):
        self.kernel -= self.dK * lr
        self.bias -= self.dB * lr

    def count_parameters(self):
        return self.kernel.size + self.bias.size

