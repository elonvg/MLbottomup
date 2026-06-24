import numpy as np

class Sigmoid():
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, r_grad):
        local_grad = self.out * (1 - self.out)
        return r_grad * local_grad
