import numpy as np

class Network():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, record=True):
        for layer in self.layers:
            layer.record = record
            x = layer.forward(x)
        return x

    def backprop(self, loss_fn, optimizer):
        grad = loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        for layer in self.layers:
            for para, grad in layer.paras_grads():
                optimizer.step(para, grad)


    def count_parameters(self):
        n = 0
        for layer in self.layers:
            if hasattr(layer, 'count_parameters'):
                n += layer.count_parameters()
        
        return n