import numpy as np

class DataLoader():
    def __init__(self, x, y, batch_size):
        self.batch_size = batch_size
        self.batches_x = np.array([x[i:i + batch_size] for i in range(0, len(x), batch_size)])
        self.batches_y = np.array([y[i:i + batch_size] for i in range(0, len(x), batch_size)])

    def get_batches(self):
        return self.batches_x, self.batches_y

    def shuffle_batches(self):
        perm = np.random.permutation(len(self.batches_x))
        self.batches_x = self.batches_x[perm]
        self.batches_y = self.batches_y[perm]
    