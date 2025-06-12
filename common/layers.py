import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        self.grads[0][...] = self.dW
        self.grads[1][...] = self.db
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
        self.params = []  # Add this line
        self.grads = []   # Add this line

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax output
        self.t = None  # target labels

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # Convert labels to one-hot if needed
        if self.t.ndim == 1:
            t_one_hot = np.zeros_like(self.y)
            t_one_hot[np.arange(batch_size), self.t] = 1
        else:
            t_one_hot = self.t

        dx = (self.y - t_one_hot) / batch_size
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
    else:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
