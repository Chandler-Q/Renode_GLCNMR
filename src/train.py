# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 08:31:28 2023

@author: c
"""
# =========load dataset========
from load_data import load_processed_data
from option import get_opt
from GraphNet import GLCNMR,GCNNet,SAGENet,GATNet

# =========== torch =========
import torch
import torch.nn.functional as F
from sklearn.metrics import  f1_score, accuracy_score
import copy


def train(opt):
    
    device = opt.device
    print('=='*20)
    print('\nLoading dataset {}'.format(opt.data_name))
    dataset = load_processed_data(opt,shuffle_seed=opt.shuffle_seed)
    
    data = dataset.to(device)
    
    opt.input_dim = data.x.shape[1]
    opt.output_dim = data.y.max()+1
    
    if opt.Net == 'GLCNMR':
        # model = GLCNMR(opt.input_dim,opt.hidden_dim,opt.output_dim,opt=opt,rn_weight=data.rn_weight[data.train_mask]).to(device)
        model = GLCNMR(opt.input_dim,opt.hidden_dim,opt.output_dim,opt=opt,rn_weight = None).to(device)
    elif opt.Net == 'GCN':
        model = GCNNet(opt.input_dim,opt.hidden_dim,opt.output_dim).to(device)
    
    elif opt.Net == 'SAGE':
        model = SAGENet(opt.input_dim,opt.hidden_dim,opt.output_dim).to(device)
        
    elif opt.Net == 'GAT':
        model = GATNet(opt.input_dim,opt.hidden_dim,opt.output_dim).to(device)
    else:
        raise ValueError("opt.Net must from one of [GLCNMR, GCN, SAGE, GAT]")
        
    
    optimizer = torch.optim.Adam(set(model.parameters()),lr=0.01,weight_decay = opt.weight_decay)
    
    
    loss_record = []
    valid_acc = [0]
    for epoch in range(opt.epoch):
        model.train()
        
        optimizer.zero_grad()
        
        out = model(dataset)
        
        if opt.Net == 'GLCNMR':
            loss = model.loss(out[data.train_mask],data.y[data.train_mask])
            # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        val_acc = accuracy_score(y_pred=out[data.valid_mask].argmax(1),y_true=data.y[data.valid_mask])
        
        if val_acc > valid_acc[-1]:
            best_model = copy.deepcopy(model)
        valid_acc.append(val_acc)
        loss_record.append(loss.detach().tolist())
        
    y = accuracy_score(best_model(data).argmax(1)[data.test_mask],data.y[data.test_mask])
    return best_model,valid_acc,y,loss_record



if __name__ == '__main__':
    opt = get_opt()
    best_model,valid_acc,y,loss_record = train(opt)
    print('=='*20)
    print("validation accuracy: %.2f %%" %(max(valid_acc)*100))
    print("  test     accuracy: %.2f %%" %(y*100))
        
        
    # sum(best_model(data).argmax(1)==2)
    # sum(data.y==3)
    