import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp_values = np.exp(inputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def backward(self, dvalues, y_true):
        no_samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(no_samples), y_true] -= 1
        self.dinputs = self.dinputs / no_samples

class Loss_CategoricalCrossentropy:
    def __init__(self):
        pass
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidence = y_pred_clipped[range(len(y_pred)), y_true]
        negative_log = -np.log(correct_confidence)
        return np.mean(negative_log)

# Create layers
layer1 = Layer_Dense(2, 128)
layer2 = Layer_Dense(128, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# Loss and accuracy
loss = loss_function.forward(activation2.output, y)
print('Loss:', loss)
predictions = np.argmax(activation2.output, axis=1)
accuracy = np.mean(predictions == y)
print('Accuracy:', accuracy)

# Backward pass
softmax_loss = Activation_Softmax()
softmax_loss.backward(activation2.output, y)
layer2.backward(softmax_loss.dinputs)
activation1.backward(layer2.dinputs)
layer1.backward(activation1.dinputs)

print('dweights:', layer1.dweights)
print('dbiases:', layer1.dbiases)

for epoch in range(1, 1000001):
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    loss = loss_function.forward(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    softmax_loss = Activation_Softmax()
    softmax_loss.backward(activation2.output, y)
    layer2.backward(softmax_loss.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    
    layer1.weights -= 0.01 * layer1.dweights
    layer1.biases -= 0.01 * layer1.dbiases
    layer2.weights -= 0.01 * layer2.dweights
    layer2.biases -= 0.01* layer2.dbiases
    
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
