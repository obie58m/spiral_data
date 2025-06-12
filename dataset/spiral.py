import numpy as np
import matplotlib.pyplot as plt

def load_data(N=100, K=3):
    x = np.zeros((N*K, 2))  # data
    t = np.zeros(N*K, dtype=np.int32)  # labels

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        theta = np.linspace(j * 4, (j+1) * 4, N) + np.random.randn(N) * 0.2
        x[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        t[ix] = j

    return x, t

if __name__ == '__main__':
    x, t = load_data()
    markers = ['o', '^', 'x']
    for i in range(3):
        plt.scatter(x[t == i, 0], x[t == i, 1], marker=markers[i])
    plt.title("Spiral Data")
    plt.grid(True)
    plt.show()
