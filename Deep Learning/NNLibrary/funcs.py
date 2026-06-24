import numpy as np

def evaluate(network, x):
    for layer in network:
        x = layer.forward(x)
    return x

def backprop(network, loss_fn, lr):
    grad = loss_fn.backward()
    for layer in reversed(network):
        grad = layer.backward(grad)

    for layer in network:
        if hasattr(layer, 'update'):
            layer.update(lr)


def total_parameters(network):
    n = 0
    for layer in network:
        if hasattr(layer, 'count_parameters'):
            n += layer.count_parameters()
    
    print(f"Total parameter count: {n}")
    return n