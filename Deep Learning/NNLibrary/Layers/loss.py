import numpy as np
from NNLibrary.Layers.learnable import Layer

def safelog(x, eps=1e-10):
    x_safe = np.array(x, dtype=float)
    x_safe[x_safe <= 0] = eps
    return np.log(x_safe)

class Loss:
    def __init__(self):
        pass

    def record_cache(self, key, var, record):
        if record:
            self.cache[key] = var

class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.cache = {}

    def forward(self, out, y, record=True):
        self.record_cache('out', out, record)
        self.record_cache('y', y, record)
        N = y.size // y.shape[-1]
        self.record_cache('N', N, record)
        loss = -np.sum(y * safelog(out) + (1 - y) * safelog(1 - out)) / N
        return loss
    
    def backward(self):
        out = self.cache['out']
        y = self.cache['y']
        N = self.cache['N']
        local_grad = - (y / out - (1 - y) / (1 - out)) / N
        return local_grad
    

class CrossEntropy(Loss):
    def __init__(self):
        self.cache = {}

    def forward(self, out, y, record=True):
        self.record_cache('out', out, record)
        self.record_cache('y', y, record)
        N = y.size // y.shape[-1]
        self.record_cache('N', N, record)
        loss = -np.sum(y * safelog(out)) / N
        return loss

    def backward(self):
        out = self.cache['out']
        y = self.cache['y']
        N = self.cache['N']
        local_grad = - (y / out) / N
        return local_grad
    

class SoftmaxCE(Loss):
    def __init__(self, T=1):
        self.T = T # Temperature
        self.cache = {}

    def forward(self, x, y, record=True):
        self.record_cache('y', y, record)

        # Softmax
        e = np.exp((x - np.max(x)) / self.T) # Numerically stable
        s = e / np.sum(e, axis=-1, keepdims=True)
        self.record_cache('s', s, record)

        # Cross Entropy
        N = y.size // y.shape[-1]
        self.record_cache('N', N, record)

        return -np.sum(y * safelog(s)) / N
    
    def backward(self):
        s = self.cache['s']
        y = self.cache['y']
        N = self.cache['N']

        return (s - y) / N
