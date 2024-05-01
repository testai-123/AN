import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def step(x):
    return np.where(x >= 0, 1, 0.0)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

# Define x range
x = np.linspace(-5, 5, 100)

# Compute activation functions
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_softmax = softmax(x)
y_step = step(x)
y_leaky_relu = leaky_relu(x)

# Plot activation functions
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title('Sigmoid')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title('ReLU')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, y_tanh, label='Tanh', color='green')
plt.title('Tanh')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(x, y_step, label='Step', color='purple')
plt.title('Step')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, y_softmax, label='SoftMax', color='Orange')
plt.title('Softmax')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='magenta')
plt.title('Leaky ReLU')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.tight_layout()
plt.show()
