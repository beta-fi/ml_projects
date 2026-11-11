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
import requests
import urllib.parse
import numpy as np
import time 

coco=pd.read_csv(r"C:\Users\Gaurav Kumar\Downloads\coconut_csv_lite-10-2026\coconut_csv_lite-10-2026.csv")


mol_wei=coco['molecular_weight']<=700
weight=coco[mol_wei]
weight['inchi_key_14'] = weight['standard_inchi_key'].str[:14]
weight = weight.drop_duplicates(subset=['inchi_key_14'])
coco_sam=weight.sample(n=20000,random_state=42)
print(len(weight))
print(coco_sam['molecular_weight'].max())
print(len(coco_sam))
hit = coco_sam[coco_sam['canonical_smiles'] == "CCOC(=O)[C@H]1CC2=C(O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@]3(O)CCO)C(OC)=C3OC=C4C3=C2[C@H](C1)C[C@@H]1C(=O)NC[C@@H]41"]
print(hit)
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




coco_graphs = []
kept_keys = []     
failed = 0
kept_smiles=[]
for i in range(len(coco_sam)):
    smi = coco_sam['canonical_smiles'].iloc[i]
    result = smiles_to_graph(smi)
    if result is None :
        failed += 1
        continue
    x, edge_index, edge_attr, fp_tensor = result
    coco_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, fp=fp_tensor))
    kept_keys.append(coco_sam['inchi_key_14' ].iloc[i])
    kept_smiles.append(coco_sam['canonical_smiles'].iloc[i])

print(kept_smiles[123])
print(kept_keys[123])
print(len(coco_graphs))
print(len(kept_keys))
print(failed)
screen_loader=DataLoader(coco_graphs,batch_size=64,shuffle=False)

seeds= [88, 32, 56, 91, 95]
seed_results = []

for seed in seeds:
    torch.manual_seed(seed) 
    model = GraphConvolutionalNetwork()
    
    model2_filename = f'model_{seed}.pt'
    model2_path = os.path.join(OUT, model2_filename)
    if os.path.exists(model2_path):
            model.load_state_dict(torch.load(model2_path))
    else:
        raise FileNotFoundError("The requested file is missing.")

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

print(len(seed_results[0]))
print(len(seed_results))
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

screen_fps = torch.stack([g.fp for g in coco_graphs])   


shared = screen_fps @ actives_tensor.T                        
a = screen_fps.sum(dim=1, keepdim=True)                         
b = actives_tensor.sum(dim=1).unsqueeze(0)                      
tanimoto = shared / (a + b - shared)

max_sim = tanimoto.max(dim=1).values                            



results = pd.DataFrame({
    'smiles': kept_smiles,
    'mean': mean,
    'std': std,
    'max_sim': max_sim.tolist()
})

results.to_csv(os.path.join(OUT, 'results_all.csv'), index=False)                     


pattern = Chem.MolFromSmarts('O=c1c(C=O)cncc1')

def has_core(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.HasSubstructMatch(pattern)
print(len(results))

novel1 = results[(results['mean'] > 0.5) & (results.max_sim < 0.3)&(results["std"]<0.09)]
novel1 = novel1.copy()
novel1['has_core'] = novel1['smiles'].apply(has_core)
novel1 = novel1[~novel1['has_core']]
novel1 = novel1.sort_values('mean', ascending=False)
print(len(novel1))
novel1 = novel1.head(100).copy()                          
   

print(novel1[[ 'mean', 'max_sim','std']])          
novel1.to_csv(os.path.join(OUT, 'coconut_candidates.csv'), index=False)


novel_30=novel1.head(30)
rest_gg = results[~results['smiles'].isin(novel_30['smiles'])]
rest_gg['has_core'] = rest_gg['smiles'].apply(has_core)
rest_gg=rest_gg[~rest_gg['has_core']]
coco_final=rest_gg.sample(n=100,random_state=42)                        
print(len(coco_final))

novel_30 = novel1.head(30).copy()
novel_30['group'] = 'model'
coco_final['group'] = 'random'
combined=pd.concat([novel_30,coco_final])
combined = combined.sample(frac=1, random_state=123).reset_index(drop=True)
combined['id'] = range(1, len(combined) + 1)
why_so_serious=combined[["id","smiles","group","mean"]]

smezir=combined[['id','smiles']].copy()
smezir["known_target"]=""
smezir["ecoli_evidence"]=""
smezir.to_csv(os.path.join(OUT,"check_sheet.csv"),index=False)
why_so_serious.to_csv(os.path.join(OUT,"KEY.csv"),index=False)
