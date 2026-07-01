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
        N = len(y)
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
        N = len(y)
        self.record_cache('N', N, record)
        loss = -np.sum(y * safelog(out)) / N
        return loss

    def backward(self):
        out = self.cache['out']
        y = self.cache['y']
        N = self.cache['N']
        local_grad = - (y / out) / N
        return local_grad
