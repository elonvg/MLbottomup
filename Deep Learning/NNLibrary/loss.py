import numpy as np

class BinaryCrossEntropy():
    def forward(self, Y, O):
        self.Y = Y
        self.O = O
        N = len(Y)
        self.out = -np.sum(Y * np.log(O) + (1 - Y) * np.log(1 - O)) / N
        return self.out
    
    def backward(self):
        N = len(self.Y)
        local_grad = - (self.Y / self.O - (1 - self.Y) / (1 - self.O)) / N
        return local_grad