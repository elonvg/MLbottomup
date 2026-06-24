import numpy as np

class BinaryCrossEntropy():
    def forward(self, o, y):
        self.y = y
        self.o = o
        N = len(y)
        self.out = -np.sum(y * np.log(o) + (1 - y) * np.log(1 - o)) / N
        return self.out
    
    def backward(self):
        N = len(self.y)
        local_grad = - (self.y / self.o - (1 - self.y) / (1 - self.o)) / N
        return local_grad