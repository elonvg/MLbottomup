import numpy as np
from NNLibrary.Layers.learnable import Layer

class Sigmoid(Layer):
    def __init__(self):
        self.cache = {}
        self.record = True

    def forward(self, x):
        x = np.clip(x, -80, 80)
        out = 1/(1+np.exp(-x))
        self.record_cache('out', out)
        return out
    
    def backward(self, r_grad):
        out = self.cache['out']
        local_grad = out * (1 - out)
        return r_grad * local_grad
    
class Softmax(Layer):
    def __init__(self, T=1):
        self.T = T
        self.cache = {}
        self.record = True

    def forward(self, x):
        e = np.exp((x - np.max(x)) / self.T) # Numerically stable
        out = e / np.sum(e, axis=-1, keepdims=True)
        self.record_cache('out', out)
        return out

    def backward(self, r_grad):
        s = self.cache['out']
        dot = np.sum(r_grad * s, axis=-1, keepdims=True)
        return s * (r_grad - dot)