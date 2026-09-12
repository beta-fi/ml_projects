import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.optim as optim
dataset = Planetoid(root='./data', name='Cora')
data=dataset[0]
class GraphConvoultionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1=GCNConv(1433,64)
        self.dropout = nn.Dropout(p=0.5)
        self.lay2=GCNConv(64,7)
    def forward(self,x,edge_index):
        x=torch.relu(self.lay1(x, edge_index))
        x=self.dropout(x) 
        x=self.lay2(x,edge_index)  
        return x
model=GraphConvoultionalNetwork()
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=5e-4)

for epoch in range(250):
       model.train()
       optimizer.zero_grad()
       outputs = model(data.x, data.edge_index)
       loss=criterion(outputs[data.train_mask],data.y[data.train_mask])
       loss.backward()
       optimizer.step()
       total=0
       correct=0
       model.eval()
       with torch.no_grad():
            outputs = model(data.x, data.edge_index)
            valloss=criterion(outputs[data.val_mask],data.y[data.val_mask])
            predicted = torch.argmax(outputs, dim=1)
            total    += data.y.size(0)
            correct  += (predicted == data.y).sum().item()
            print(f'Epoch {epoch+1}/250, Loss: {loss:.4f}, valloss: {valloss:.4f}, Accuracy: {correct/total:.4f}')
            
    






