# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:58:23 2023

@author: c
"""

# ======== 基础package =========
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score

# ======== My packages ========
from option import get_opt
from GraphNet import SAGENet,GATNet,GCNNet
from load_data import load_processed_data


# ======== torch ===========
import torch
import torch.nn.functional as F

# ======== geometric =======
from torch_geometric.loader import DataLoader

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='../data', name='Cora')
re_node = 1
opt = get_opt()
dataset = load_processed_data(opt,shuffle_seed = 0)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GATNet().to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = torch.nn.CrossEntropyLoss() # reduction='none',weight = torch.tensor([0.05,0.1,0.4,0.4,0.2,0.05])

data=dataset
#model.train()
for epoch in range(200):
    
    optimizer.zero_grad()
    out = model(data)
        
    # 0 为 unlabelled node
    train_mask = data.train_mask
    output_loss = loss(out[train_mask], data.y[train_mask])
    if re_node == 1:
        output_loss = torch.mul(output_loss,data.rn_weight[train_mask])
        output_loss = output_loss.mean()
    
    
    output_loss.backward()
    optimizer.step()
    # print('\r loss:%.3f' %output_loss)


model.eval()
correct = 0
sum_num = 0
test_mask = data.test_mask
# for data in loader:
    
pred = model(data).argmax(dim=1)
correct += int((pred[test_mask] == data.y[test_mask]).sum())
sum_num += len(data.y[test_mask])
acc = int(correct) / sum_num

w_f1 = f1_score(data.y[test_mask],pred[test_mask],average='weighted')
print(f'Accuracy: {acc:.4f}\nF1 score:{w_f1:.4f}')    


print([sum(data.y[data.train_mask]==i) for i in range(5)])
    
    