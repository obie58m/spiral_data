import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet


max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0
global_step = 0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append((global_step, avg_loss))
            total_loss, loss_count = 0, 0
            
        global_step += 1
def plot_decision_boundary(model, x, t):
    # Define the grid
    h = 0.01
    x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
    y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.predict(grid)
    predicted = np.argmax(scores, axis=1)
    predicted = predicted.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predicted, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot the spiral dataset points
    markers = ['o', 's', '^']
    for i in range(3):
        plt.scatter(x[t == i][:, 0], x[t == i][:, 1], label=f'Class {i}', s=20, marker=markers[i])
    
    plt.title('Decision Boundary of Trained Model')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        
if loss_list:
    iterations, losses = zip(*loss_list)
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if t.ndim == 2:
    t = np.argmax(t, axis=1)
    
plot_decision_boundary(model, x, t)