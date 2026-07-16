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
        """
        Method for returning parameters and gradients
        """
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

        self.Wx = np.random.normal(scale=np.sqrt(2/in_dim), size=[in_dim, hidden_dim]).astype(np.float32)
        self.Wh = np.random.normal(scale=np.sqrt(2/hidden_dim), size=[hidden_dim, hidden_dim]).astype(np.float32)
        self.Wout = np.random.normal(scale=np.sqrt(2/hidden_dim), size=[hidden_dim, out_dim]).astype(np.float32)
        self.Bh = np.zeros(hidden_dim, dtype=np.float32)
        self.Bout = np.zeros(out_dim, dtype=np.float32)

        self.dWx = np.zeros_like(self.Wx, dtype=np.float32)
        self.dWh = np.zeros_like(self.Wh, dtype=np.float32)
        self.dWout = np.zeros_like(self.Wout, dtype=np.float32)
        self.dBh = np.zeros_like(self.Bh, dtype=np.float32)
        self.dBout = np.zeros_like(self.Bout, dtype=np.float32)

        self.cache = {}
        self.record = True


    def forward(self, x):
        self.record_cache('x', x)
        batch_size, T, in_dim = x.shape

        # Encode sequence
        x_encoded = x @ self.Wx # Shape: [batch_size, T, hidden_dim]

        # Compute hidden states and output
        hidden_states = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        outputs = np.zeros((batch_size, T, self.out_dim)).astype(np.float32)
        for t in range(T):
            h_prev = hidden_states[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            h = np.tanh(h_prev @ self.Wh + x_encoded[:, t, :] + self.Bh)
            hidden_states[:, t, :] = h
            outputs[:, t, :] = h @ self.Wout + self.Bout
        
        self.record_cache('hidden_states', hidden_states)

        return outputs

    def backward(self, r_grad):
        x = self.cache['x'] # Shape: [batch_size, T, in_dim]
        hidden_states = self.cache['hidden_states']
        batch_size, T, hidden_dim = hidden_states.shape

        # Zero
        for g in (self.dWx, self.dWh, self.dWout, self.dBh, self.dBout):
            g.fill(0)

        local_grad = np.zeros_like(x, dtype=np.float32)

        dydz_tplus1 = np.zeros([batch_size, hidden_dim], dtype=np.float32)
        for t in reversed(range(T)):
            h_t = hidden_states[:, t, :]
            h_tminus1 = hidden_states[:, t-1, :] if t > 0 else np.zeros([batch_size, hidden_dim], dtype=np.float32)

            dydh_t = r_grad[:, t, :] @ self.Wout.T + dydz_tplus1 @ self.Wh.T # Shape: [batch_size, hidden_dim]
            dhdz_t = (1 - h_t**2)
            dydz_t = dydh_t * dhdz_t # Shape: [batch_size, hidden_dim]

            self.dWx += x[:, t, :].T @ dydz_t # Shape: [in_dim, hidden_dim]
            self.dWh += h_tminus1.T @ dydz_t # Shape: [hidden_dim, hidden_dim]
            self.dBh += np.sum(dydz_t, axis=0) # Shape: [hidden_dim]
            self.dWout += h_t.T @ r_grad[:, t, :] # Shape: [hidden_dim, out_dim]
            self.dBout += np.sum(r_grad[:, t, :], axis=0) # Shape: [out_dim]

            local_grad[:, t, :] = dydz_t @ self.Wx.T # Shape: [batch_size, in_dim]

            dydz_tplus1 = dydz_t # Carry forward

        return local_grad
    
    def paras_grads(self):
        return [
            (self.Wx, self.dWx),
            (self.Wh, self.dWh),
            (self.Bh, self.dBh),
            (self.Wout, self.dWout),
            (self.Bout, self.dBout),
        ]

    def count_parameters(self):
        return self.Wx.size + self.Wh.size + self.Bh.size + self.Wout.size + self.Bout.size

# TODO:
# Dont use Layer sigmoid

def _sigmoid(z):
    z = np.clip(z, -80, 80)
    return 1/(1+np.exp(-z))

class LSTM(Layer):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        D = in_dim + hidden_dim
        self.W = np.random.normal(scale=np.sqrt(2/D), size=[D, 4 * hidden_dim]).astype(np.float32)
        self.B = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.Wout = np.random.normal(scale=np.sqrt(2/hidden_dim), size=[hidden_dim, out_dim]).astype(np.float32)
        self.Bout = np.zeros(out_dim, dtype=np.float32)
        
        self.dW = np.zeros_like(self.W, dtype=np.float32)
        self.dB = np.zeros_like(self.B, dtype=np.float32)
        self.dWout = np.zeros_like(self.Wout, dtype=np.float32)
        self.dBout = np.zeros_like(self.Bout, dtype=np.float32)

        self.cache = {}
        self.record = True

    def forward(self, x):
        self.record_cache('x', x)
        batch_size, T, in_dim = x.shape

        # Init activations
        concats = np.zeros((batch_size, T, self.in_dim + self.hidden_dim)).astype(np.float32)
        forgets = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        c_tildes = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        update_gates = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        output_gates = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        # Init hidden states and output
        cell_states = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        hidden_states = np.zeros((batch_size, T, self.hidden_dim)).astype(np.float32)
        outputs = np.zeros((batch_size, T, self.out_dim)).astype(np.float32)

        for t in range(T):
            C_prev = cell_states[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            h_prev = hidden_states[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

            hx = np.concatenate((h_prev, x[:, t, :]), axis=1)
            concats[:, t, :] = hx
            z = hx @ self.W + self.B # Shape: [batch_size, 4 * hidden_dim]
            H = self.hidden_dim
            zf, zc, zu, zo = z[:, 0:H], z[:, H:2*H], z[:, 2*H:3*H], z[:, 3*H:4*H]

            # "Forget"
            forget = _sigmoid(zf)
            Cf = C_prev * forget
            # Store
            forgets[:, t, :] = forget

            # "Input" / Update cell
            c_tilde = np.tanh(zc)
            update_gate = _sigmoid(zu)
            C = Cf + c_tilde * update_gate
            # Store
            update_gates[:, t, :] = update_gate
            c_tildes[:, t, :] = c_tilde
            cell_states[:, t, :] = C

            # "Output" / Compute state
            output_gate = _sigmoid(zo)
            output_gates[:, t, :] = output_gate
            h = output_gate * np.tanh(C)
            # Store
            hidden_states[:, t, :] = h

            # Out
            outputs[:, t, :] = h @ self.Wout + self.Bout

        self.record_cache('concats', concats)
        self.record_cache('forgets', forgets)
        self.record_cache('c_tildes', c_tildes)
        self.record_cache('update_gates', update_gates)
        self.record_cache('output_gates', output_gates)
        self.record_cache('cell_states', cell_states)
        self.record_cache('hidden_states', hidden_states)

        return outputs
    
    def backward(self, r_grad):
        x = self.cache['x'] # Shape: [batch_size, T, in_dim]
        concats = self.cache['concats']
        forgets = self.cache['forgets']
        c_tildes = self.cache['c_tildes']
        update_gates = self.cache['update_gates']
        output_gates = self.cache['output_gates']
        cell_states = self.cache['cell_states']
        hidden_states = self.cache['hidden_states']

        batch_size, T, hidden_dim = hidden_states.shape

        # Zero
        for g in (self.dW, self.dB, self.dWout, self.dBout):
            g.fill(0)

        local_grad = np.zeros_like(x, dtype=np.float32)

        dC_tplus1 = np.zeros([batch_size, hidden_dim], dtype=np.float32)
        dh_tplus1 = np.zeros([batch_size, hidden_dim], dtype=np.float32)
        for t in reversed(range(T)):
            # Fetch current
            hx = concats[:, t, :]
            forget = forgets[:, t, :]
            c_tilde = c_tildes[:, t, :]
            update_gate = update_gates[:, t, :]
            output_gate = output_gates[:, t, :]
            C = cell_states[:, t, :]
            C_tminus1 = cell_states[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            h = hidden_states[:, t, :]

            dh = r_grad[:, t, :] @ self.Wout.T + dh_tplus1
            dC = dh * output_gate * (1 - np.tanh(C)**2) + dC_tplus1

            dc_tilde = dC * update_gate

            dzf = dC * C_tminus1 * (forget * (1 - forget))
            dzc = dc_tilde * (1 - c_tilde**2)
            dzu = dC * c_tilde * (update_gate * (1 - update_gate))
            dzo = dh * np.tanh(C) * (output_gate * (1 - output_gate))
            dz = np.concatenate([dzf, dzc, dzu, dzo], axis=1)

            self.dW += hx.T @ dz
            self.dB += np.sum(dz, axis=0)
            self.dWout += h.T @ r_grad[:, t, :]
            self.dBout += np.sum(r_grad[:, t, :], axis=0)

            dhx = dz @ self.W.T
            dC_tplus1 = dC * forget
            dh_tplus1 = dhx[:, :self.hidden_dim]

            local_grad[:, t, :] = dhx[:, self.hidden_dim::]

        return local_grad
    
    def paras_grads(self):
        return [
            (self.W, self.dW),
            (self.B, self.dB),
            (self.Wout, self.dWout),
            (self.Bout, self.dBout),
        ]

    def count_parameters(self):
        return self.W.size + self.B.size + self.Wout.size + self.Bout.size


