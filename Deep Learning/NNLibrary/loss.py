import numpy as np

def safelog(x, eps=1e-10):
    x_safe = np.array(x, dtype=float)
    x_safe[x_safe <= 0] = eps
    return np.log(x_safe)

class BinaryCrossEntropy():
    def forward(self, o, y):
        self.o = o
        self.y = y
        self.N = len(y)
        self.out = -np.sum(y * safelog(o) + (1 - y) * safelog(1 - o)) / self.N
        return self.out
    
    def backward(self):
        local_grad = - (self.y / self.o - (1 - self.y) / (1 - self.o)) / self.N
        return local_grad
    

class CrossEntropy():
    def forward(self, o, y):
        self.o = o
        self.y = y
        self.N = len(y)
        self.out = -np.sum(y * safelog(o)) / self.N
        return self.out

    def backward(self):
        local_grad = - (self.y / self.o) / self.N
        return local_grad
