import numpy as np

class GradientDescent():
    def __init__(self, lr):
        self.lr = lr

    def step(self, paras, grads):
        paras -= grads * self.lr

class MomentumDescent():
    def __init__(self, lr, beta=0.5):
        self.lr = lr
        self.beta = beta
        self.steps = {}

    def step(self, paras, grads):
        id_paras = id(paras)

        if id_paras not in self.steps:
            self.steps[id_paras] = np.zeros_like(paras, dtype=np.float32)
        
        prev_step = self.steps[id_paras]
        step = self.beta * prev_step + (1 - self.beta) * grads
        self.steps[id_paras] = step
        
        paras -= step * self.lr

class Adam():
    def __init__(self, lr, beta=0.5, gamma=0.5, eps=1e-6):
        self.lr = lr
        self.beta = beta # High beta means to trust long-running averages
        self.gamma = gamma
        self.eps = eps
        self.steps = {}

    def step(self, paras, grads):
        id_paras = id(paras)

        if id_paras not in self.steps:
            self.steps[id_paras] = {
                "m" : np.zeros_like(paras, dtype=np.float32),
                "v" : np.zeros_like(paras, dtype=np.float32),
                "t" : 0
            }

        prev_step = self.steps[id_paras]

        prev_step["t"] += 1
        t = prev_step["t"]
        
        m_prev = prev_step["m"]
        m = self.beta * m_prev + (1 - self.beta) * grads
        prev_step["m"] = m

        v_prev = prev_step["v"]
        v = self.gamma * v_prev + (1 - self.gamma) * grads ** 2
        prev_step["v"] = v

        # Bias correction
        m_tilde = m / (1 - self.beta ** t)
        v_tilde = v / (1 - self.gamma ** t)
        
        paras -= m_tilde / (np.sqrt(v_tilde) + self.eps) * self.lr
