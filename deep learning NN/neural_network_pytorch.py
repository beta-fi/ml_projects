import torch
import torch.nn as nn
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
model = NeuralNetwork()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(3001):
    
    # 1. Forward pass
      outputs = model(X)
    
    # 2. Calculate loss
      loss = loss_function(outputs, y)
    
    # 3. Zero gradients — prevents accumulation from previous loop
      optimizer.zero_grad()
    
    # 4. Backward pass
      loss.backward()
    
    # 5. Update weights
      optimizer.step()
    
if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}')
predictions = torch.argmax(outputs, dim=1)
accuracy = (predictions == y).float().mean()
print (accuracy)