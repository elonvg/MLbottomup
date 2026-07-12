import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Layer:
    def __init__(self):
        self.cache = {}
        self.record = True

    def record_cache(self, key, var):
        """
        Function for recording variables for backprop
        """
        if self.record:
            self.cache[key] = var

    def paras_grads(self):
        return []

class LinearLayer(Layer):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.normal(scale=0.01, size=[in_dim, out_dim]).astype(np.float32)
        self.B = np.zeros(out_dim, dtype=np.float32)

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
        self.bias = np.zeros(out_channels, dtype=np.float32)

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
    
class RNN(Layer):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.W_xh = np.random.normal(scale=2/hidden_dim, size=[in_dim, hidden_dim]).astype(np.float32)
        self.W_hh = np.random.normal(scale=2/hidden_dim, size=[hidden_dim, hidden_dim]).astype(np.float32)
        self.W_hout = np.random.normal(scale=2/hidden_dim, size=[hidden_dim, out_dim]).astype(np.float32)
        self.B_h = np.zeros(hidden_dim, dtype=np.float32)
        self.B_out = np.zeros(out_dim, dtype=np.float32)

        self.dW_xh = np.zeros_like(self.W_xh, dtype=np.float32)
        self.dW_hh = np.zeros_like(self.W_hh, dtype=np.float32)
        self.dW_hout = np.zeros_like(self.W_hout, dtype=np.float32)
        self.dB_h = np.zeros_like(self.B_h, dtype=np.float32)
        self.dB_out = np.zeros_like(self.B_out, dtype=np.float32)

        self.cache = {}
        self.record = True


    def forward(self, x):
        self.record_cache('x', x)
        batch_size, T, in_dim = x.shape

        # Encode sequence
        x_encoded = x @ self.W_xh # Shape: [batch_size, T, hidden_dim]

        # Compute hidden states and output
        hidden_states = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        outputs = np.zeros((batch_size, T, self.out_dim)).astype(np.float32)
        for t in range(T):
            h_prev = hidden_states[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            h = np.tanh(h_prev @ self.W_hh + x_encoded[:, t, :] + self.B_h)
            hidden_states[:, t, :] = h
            outputs[:, t, :] = h @ self.W_hout + self.B_out
        
        self.record_cache('hidden_states', hidden_states)
        self.record_cache('outputs', outputs)

        return outputs

    def backward(self, r_grad):
        x = self.cache['x'] # Shape: [batch_size, T, in_dim]
        hidden_states = self.cache['hidden_states']
        batch_size, T, hidden_dim = hidden_states.shape

        # Zero
        self.dW_xh = np.zeros_like(self.W_xh, dtype=np.float32)
        self.dW_hh = np.zeros_like(self.W_hh, dtype=np.float32)
        self.dW_hout = np.zeros_like(self.W_hout, dtype=np.float32)
        self.dB_h = np.zeros_like(self.B_h, dtype=np.float32)
        self.dB_out = np.zeros_like(self.B_out, dtype=np.float32)

        local_grad = np.zeros_like(x, dtype=np.float32)

        dydz_tplus1 = np.zeros([batch_size, hidden_dim], dtype=np.float32)
        for t in reversed(range(T)):
            h_t = hidden_states[:, t, :]
            h_tminus1 = hidden_states[:, t-1, :] if t > 0 else np.zeros([batch_size, hidden_dim], dtype=np.float32)

            dydh_t = r_grad[:, t, :] @ self.W_hout.T + dydz_tplus1 @ self.W_hh.T # Shape: [batch_size, hidden_dim]
            dhdz_t = (1 - h_t**2)
            dydz_t = dydh_t * dhdz_t # Shape: [batch_size, hidden_dim]

            self.dW_xh += x[:, t, :].T @ dydz_t # Shape: [in_dim, hidden_dim]
            self.dW_hh += h_tminus1.T @ dydz_t # Shape: [hidden_dim, hidden_dim]
            self.dB_h += np.sum(dydz_t, axis=0) # Shape: [hidden_dim]
            self.dW_hout += h_t.T @ r_grad[:, t, :] # Shape: [hidden_dim, out_dim]
            self.dB_out += np.sum(r_grad[:, t, :], axis=0) # Shape: [out_dim]

            local_grad[:, t, :] = dydz_t @ self.W_xh.T # Shape: [batch_size, in_dim]

            dydz_tplus1 = dydz_t # Carry forward

        return local_grad
    
    def paras_grads(self):
        return [
            (self.W_xh, self.dW_xh),
            (self.W_hh, self.dW_hh),
            (self.B_h, self.dB_h),
            (self.W_hout, self.dW_hout),
            (self.B_out, self.dB_out),
        ]

    def count_parameters(self):
        return self.W_xh.size + self.W_hh.size + self.B_h.size + self.W_hout.size + self.B_out.size

