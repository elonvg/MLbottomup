import numpy as np
from NNLibrary.Layers.learnable import Layer

class Sigmoid(Layer):
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, r_grad):
        local_grad = self.out * (1 - self.out)
        return r_grad * local_grad
    
class Softmax(Layer):
    def forward(self, x):
        x = np.clip(x, -500, 500)
        e = np.exp(x)
        self.out = e / np.sum(e, axis=1, keepdims=True)
        return self.out

    def backward(self, r_grad):
        s = self.out
        dot = np.sum(r_grad * s, axis=1, keepdims=True)
        return s * (r_grad - dot)