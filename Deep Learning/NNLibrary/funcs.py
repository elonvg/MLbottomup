import numpy as np

def evaluate(network, x):
    for layer in network:
        x = layer.forward(x)
    return x

def backprop(network, loss_fn, optimizer):
    grad = loss_fn.backward()
    for layer in reversed(network):
        grad = layer.backward(grad)

    for layer in network:
        for para, grad in layer.paras_grads():
            optimizer.step(para, grad)


