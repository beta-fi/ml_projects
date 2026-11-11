import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import global_mean_pool, GATConv
from sklearn.model_selection import train_test_split
from rdkit import Chem
from sklearn.metrics import average_precision_score
from rdkit.Chem import AllChem
import pubchempy as pcp

import numpy as np
import time 
OUT = r'C:\Users\Gaurav Kumar\Documents\antibiotics-gnn'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv('C:/Users/Gaurav Kumar/Antibiotics_Chemprop/data/article_training.csv')


def one_hot_encode(val, choices):
   
    if val not in choices:
        val = choices[-1]
    return [float(val == choice) for choice in choices]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    
    allowed_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53, 'Other']   
    allowed_degrees = [0, 1, 2, 3, 4, 5, 'Other']               
    allowed_bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp_uint = generator.GetFingerprintAsNumPy(mol)
    
    
    node_features = []
    for atom in mol.GetAtoms():
        atom_type = one_hot_encode(atom.GetAtomicNum(), allowed_atoms) 
        degree = one_hot_encode(atom.GetDegree(), allowed_degrees)     
        
        charge = [float(atom.GetFormalCharge())]                      
        aromatic = [float(atom.GetIsAromatic())]                       
        
        # TOTAL FEATURES: 10 + 7 + 1 + 1 = 19 features per atom
        atom_feats = atom_type + degree + charge + aromatic
        node_features.append(atom_feats)

    source = []
    target = []
    bonder=[]
    for bond in mol.GetBonds():
        bond_feat = one_hot_encode(bond.GetBondType(), allowed_bonds)
        bonder.append(bond_feat)
        bonder.append(bond_feat)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        source.append(u)
        target.append(v)
        source.append(v)
        target.append(u)

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_attr = torch.tensor(bonder, dtype=torch.float32).reshape(-1, 4)
    edge_index = torch.tensor([source, target], dtype=torch.long).reshape(2, -1)
    fp_tensor = torch.tensor(fp_uint, dtype=torch.float32)
    return x, edge_index,edge_attr,fp_tensor

grahpy_boi = []
for i in range(len(df)):
    activity = df['activity'].iloc[i]
    result = smiles_to_graph(df['smiles'].iloc[i])
    
    if result is None:
        continue
    
    x, edge_index,edge_attr,fp_tensor = result
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor(activity, dtype=torch.long),edge_attr=edge_attr,fp=fp_tensor)
    grahpy_boi.append(graph)

labels = [graph.y.item() for graph in grahpy_boi]


train_data, temp_data = train_test_split(grahpy_boi, test_size=0.2, stratify=labels, random_state=42)
labels1 = [graph.y.item() for graph in temp_data]
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=labels1, random_state=42)
active_fps = [data.fp for data in train_data if data.y.item() == 1]
actives_tensor = torch.stack(active_fps)

train_labels = [graph.y.item() for graph in train_data]
count_0 = train_labels.count(0)
count_1 = train_labels.count(1)

weight_0 = 1.0 / count_0 if count_0 > 0 else 1.0
weight_1 = 1.0 / count_1 if count_1 > 0 else 1.0

sample_weights = [weight_0 if label == 0 else weight_1 for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


trainloader = DataLoader(train_data, batch_size=32, sampler=sampler)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)



class GraphConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = GATConv(19, 32, heads=2,edge_dim=4)
        
        
        self.layer2 = GATConv(64, 16, heads=2,edge_dim=4)
        
       
        self.fc = nn.Linear(32+2048, 2) 
        
    def forward(self, x, edge_index, batch,edge_attr,fp):
        x = torch.relu(self.layer1(x, edge_index,edge_attr))
        x = self.layer2(x, edge_index,edge_attr) 
        x = global_mean_pool(x, batch)
        fp = fp.view(x.size(0), -1)
        combined = torch.cat([x, fp], dim=-1)
        x = self.fc(combined)
        return x



model = GraphConvolutionalNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-2)

if os.path.exists('model.pt'):
    model.load_state_dict(torch.load('model.pt'))
else:
 for epoch in range(35):
    
    model.train()
    running_loss = 0.0
    for batch in trainloader:
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch,batch.edge_attr,batch.fp)
        loss = loss_fn(outputs, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(trainloader)
    
  
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): 
        for batch in val_loader:
            val_outputs = model(batch.x, batch.edge_index, batch.batch,batch.edge_attr,batch.fp)
            v_loss = loss_fn(val_outputs, batch.y.squeeze(-1))
            val_loss += v_loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    
    
    print(f"Epoch: {epoch+1}/35 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
torch.save(model.state_dict(), 'model.pt')
total_hits=0
correct_hits=0
predicted_hits=0
correct=0
total=0
all_probs=[]
all_hits=[]

model.eval()
with torch.no_grad():
    for batch in testloader:
        outputs = model(batch.x, batch.edge_index, batch.batch,batch.edge_attr,batch.fp)
        probs=torch.softmax(outputs, dim=1)
        hit_probs = probs[:, 1]
        predicted = (hit_probs > 0.7).long()
        true_y = batch.y.squeeze(-1)
        
        total_hits += (true_y == 1).sum().item()
        correct_hits += ((predicted == 1) & (true_y == 1)).sum().item()
        predicted_hits += (predicted == 1).sum().item()
        correct += (predicted == true_y).sum().item()
        total += true_y.size(0)
        all_probs.extend(hit_probs.tolist())
        all_hits.extend(true_y.tolist())
      
       
import requests
import urllib.parse
import time

def get_name(smi):
    try:
        encoded_smi = urllib.parse.quote(smi, safe='')
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smi}/synonyms/json"

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            synonyms = data.get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])
            for s in synonyms:
                if not any(c.isdigit() for c in s):
                    return s
            if synonyms:
                return synonyms[0]
        else:
            print("PubChem status:", response.status_code)   
    except Exception as e:
        print("PubChem error:", e)
    finally:
        time.sleep(0.5)  
    return "unknown"
precision = correct_hits / predicted_hits if predicted_hits > 0 else 0.0
recall = correct_hits / total_hits if total_hits > 0 else 0.0
prc_auc = average_precision_score(all_hits, all_probs)


print(f"\n--- Test Results ---")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Actual Positive Count:    {total_hits}")
print(f"Predicted Positive Count: {predicted_hits}")
print(f'Accuracy :{correct/total}')
print(f'prc-auc :{prc_auc}')
with open(os.path.join(OUT, 'test_metrics.txt'), 'w') as f:
    f.write(f"PRC-AUC {prc_auc:.4f}\nRecall {recall:.4f}\nPrecision {precision:.4f}\n"
            f"Positives {total_hits}\nPredicted {predicted_hits}\nAccuracy {correct/total:.4f}\n")



import pandas as pd

df2 = pd.read_csv('C:/Users/Gaurav Kumar/coronavirus_data/data/broad_repurposing_library.csv')
screen_graphs = []
kept_smiles = []
for i in range(len(df2)):
    smi = df2['smiles'].iloc[i]
    result = smiles_to_graph(smi)
    if result is None:
        continue
    x, edge_index, edge_attr, fp_tensor = result   # 4 values, not 3
    screen_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, fp=fp_tensor))
    kept_smiles.append(smi) 


screen_loader=DataLoader(screen_graphs, batch_size=64,shuffle=False)
model.eval()
screen_probs=[]
with torch.no_grad(): 
    for batch in screen_loader:
        out=model(batch.x, batch.edge_index, batch.batch,batch.edge_attr,batch.fp)
        screen_probs.extend(torch.softmax(out, dim=1)[:, 1].tolist())

results = pd.DataFrame({'smiles': kept_smiles, 'hit_prob': screen_probs})
results = results.sort_values('hit_prob', ascending=False)
print(results[['smiles', 'hit_prob']].head(10))
top10 = results.head(10).copy()
top10['name'] = top10['smiles'].apply(get_name)
print(top10[['name', 'hit_prob', 'smiles']].to_string())
top10.to_csv(os.path.join(OUT, 'top10_pre_novelty.csv'), index=False)
results.to_csv(os.path.join(OUT, 'screening_predictions.csv'), index=False)

screen_fps = torch.stack([g.fp for g in screen_graphs])      

# Tanimoto: shared / (a + b - shared)
shared = screen_fps @ actives_tensor.T                        
a = screen_fps.sum(dim=1, keepdim=True)                         
b = actives_tensor.sum(dim=1).unsqueeze(0)                      
tanimoto = shared / (a + b - shared)

max_sim = tanimoto.max(dim=1).values                            

results = pd.DataFrame({'smiles': kept_smiles,
                        'hit_prob': screen_probs,
                        'max_sim': max_sim.tolist()})

novel = results[(results.hit_prob > 0.5) & (results.max_sim < 0.3)]

pattern = Chem.MolFromSmarts('O=c1c(C=O)cncc1')

def has_core(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.HasSubstructMatch(pattern)




novel = results[(results.hit_prob > 0.5) & (results.max_sim < 0.3)]
novel = novel.copy()
novel['has_core'] = novel['smiles'].apply(has_core)
novel = novel[~novel['has_core']]
novel = novel.sort_values('hit_prob', ascending=False)
novel = novel.head(30).copy()                          
novel['name'] = novel['smiles'].apply(get_name)         

print(novel[['name', 'hit_prob', 'max_sim']])          
 
 
novel.to_csv(os.path.join(OUT, 'novel_candidates.csv'), index=False)



seed_results=[]
screen_loader=DataLoader(screen_graphs,batch_size=64,shuffle=False)


seed_results = []
seeds = [88, 32, 56, 91, 95]

for seed in seeds:
    torch.manual_seed(seed) 
    model = GraphConvolutionalNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    model2_filename = f'model_{seed}.pt'
    model2_path = os.path.join(OUT, model2_filename)
    
    if os.path.exists(model2_path):
        model.load_state_dict(torch.load(model2_path))
    else: 
        for epoch in range(35):
            model.train()
            for batch in trainloader:
                optimizer.zero_grad()
                outputs = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.fp)
                loss = loss_fn(outputs, batch.y.squeeze(-1))
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), model2_path) 
        
    model.eval()
    screen_probs = []             
    with torch.no_grad():
        for batch in screen_loader:
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.fp)
            screen_probs.extend(torch.softmax(out, dim=1)[:, 1].tolist())
    
    seed_results.append(screen_probs)


arr = np.array(seed_results)   
mean = arr.mean(axis=0)
std = arr.std(axis=0)
print("Ensemble array shape:", arr.shape)


resul = pd.DataFrame({'smiles': kept_smiles, 'std': std, 'mean': mean, 'max_sim': max_sim.tolist()})
resul = resul[(resul['mean'] > 0.5) & (resul['std'] < 0.09) & (resul.max_sim < 0.3)]
resul['has_core'] = resul['smiles'].apply(has_core)
resul = resul[~resul['has_core']]
resul = resul.sort_values('mean', ascending=False)
resul['name'] = resul['smiles'].apply(get_name)   
resul.to_csv(os.path.join(OUT, 'resul.csv'), index=False)

print("\n--- Final Filtered Screening Results ---")
print(resul[['name', 'std', 'mean', 'max_sim']]) 


check_files = ['screening_predictions.csv', 'top10_pre_novelty.csv', 'novel_candidates.csv', 'resul.csv', 'test_metrics.txt']
for seed in seeds:
    check_files.append(f'model_{seed}.pt')

print("\n--- Output Verification Loop ---")
for fn in check_files:
    p = os.path.join(OUT, fn)
    print(f"{fn:<30} -> {'OK' if os.path.exists(p) and os.path.getsize(p) > 0 else 'MISSING'}")

print('\nAll pipeline files saved successfully in:', OUT)
  