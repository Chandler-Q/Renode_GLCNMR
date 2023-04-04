# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:08:35 2023

@author: c
"""

# ======== basic packages =========
import numpy as np
import pandas as pd

# ======== torch ===============
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv,GATConv,SAGEConv

from torch_geometric.utils import get_laplacian
import math

class SAGENet(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        # torch.nn.Sequential
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr='mean')    # num of node features ->
        self.conv2 = SAGEConv(hidden_dim, hidden_dim,aggr='mean')   # num of y classes
        self.lin1 = nn.Linear(hidden_dim, output_dim)
      
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x=x,edge_index=edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)


        return x
    
    
class GATNet(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        # torch.nn.Sequential
        self.conv1 = GATConv(input_dim, hidden_dim, aggr='mean')    # num of node features ->
        self.conv2 = GATConv(hidden_dim, hidden_dim,aggr='mean')   # num of y classes
        self.lin1 = nn.Linear(hidden_dim, output_dim)
      
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x=x,edge_index=edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        
        x = self.conv2(x=x, edge_index=edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)


        return x
    
class GCNNet(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        # torch.nn.Sequential
        self.conv1 = GCNConv(input_dim, hidden_dim)    # num of node features ->
        self.conv2 = GCNConv(hidden_dim, output_dim)   # num of y classes

      
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x=x,edge_index=edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x=x, edge_index=edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        

        return x


class SparseGraphLearn(nn.Module):
    
    def __init__(self,input_dim,output_dim,bias = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # x = (n,d) Q = (d,p)
        self.Q = nn.Parameter(torch.Tensor(input_dim,output_dim),requires_grad=True)
        
        # x_ = (n,p) a = (p,1)
        self.a = nn.Parameter(torch.Tensor(output_dim,1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.Q, a=math.sqrt(10))
        init.kaiming_uniform_(self.a, a=math.sqrt(10))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.Q)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        num_node = x.shape[0]
        x_ = torch.mm(x,self.Q)  # (n,d) * (d,p) = (n,p)
        if self.bias is not None:
            x_ += self.bias
        # (n,p) - (n,p) = (n,p)
        S = torch.abs(x_.index_select(0,edge_index[0])-x_.index_select(0,edge_index[1]))
        
        # (n,p)
        S = torch.sigmoid(torch.mm(S,self.a))
        
        S = torch.sparse_coo_tensor(indices=edge_index, values=S.reshape(-1), 
                                    size=[num_node, num_node], dtype=torch.float32).to_dense()
        
        S = F.softmax(S,dim=1)
        # x_为下一层输入，S是计算得到的本层loss
        
        return x_,S
        
    def get_config(self):
        return {"output_dim": self.output_dim, "input_dim": self.input_dim}
        
        
class GLCNMR(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,opt=None,rn_weight=None,bias = True):
        super().__init__()
        '''
        input:
            input_dim: 输入层特征数目
            hidden_dim： embedding 后 特征数目
            output_dim:  输出特征数目，一般为分类类别数目
            
        '''
        self.GL = SparseGraphLearn(input_dim,hidden_dim)
        
        # self.GCN = GCNConv(hidden_dim,output_dim)
        
        self.lin = nn.Linear(hidden_dim, output_dim)
        
        self.loss1 = 0      # loss1为Graph Learning loss
        self.loss2 = 0      # loss2为smooth loss
        self.rn_weight = rn_weight
        
        if opt is None:
            self.lambda1 = 0.001
            self.lambda2 = 0.001
            self.lamb = 0.5
        else:
            self.lambda1 = opt.lambda1
            self.lambda2 = opt.lambda2
            self.lamb = opt.lamb
          
        self.ce = nn.CrossEntropyLoss(reduction='none')
            
    def forward(self,data):
        
        # x用于网络层继续传播， S用于计算loss
        # x_ = (n,p)  S = (n,n)
        self.edge_index = data.edge_index
        
        x_,S = self.GL(data)
        
        self.x_ = x_.detach()
        self.S = S.detach()
        
        # 得到S的拉普拉斯变化 Ds^-1/2 * S * Ds^-1/2 
        S = S.to_sparse_coo()
        # L = I - Ds^-1/2 * S * Ds^-1/2 
        L = get_laplacian(edge_index=S.indices(),edge_weight=S.values(),normalization='sym',dtype=torch.float32,num_nodes=S.size()[0])
        L = torch.sparse_coo_tensor(indices=L[0], values=L[1].reshape(-1), size=list(S.size()), dtype=torch.float32).to_dense()
        L = -L+torch.eye(S.size()[0])
        
        x = torch.mm(L,x_)
        
        x = self.lin(x)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.GCN(x=x_,edge_index=self.edge_index)
        x = F.softmax(x,dim=1)
        self.x = x.detach()
        
        
        return x
        
    def loss(self,x,y):
        
        num_node = self.x_.size(0)
        nor = torch.norm(self.x_.index_select(0,self.edge_index[0])-self.x_.index_select(0,self.edge_index[1]),
                         p=2,dim=1)**2
        
        nor = torch.sparse_coo_tensor(indices=self.edge_index, values=nor.reshape(-1), 
                                      size=[num_node, num_node], dtype=torch.float32).to_dense()
        
        self.loss1 = torch.mul(nor,self.S).mean() + self.lamb*torch.norm(self.S)
        
        S = self.S.to_sparse_coo()
        L = get_laplacian(edge_index=S.indices(),edge_weight=S.values(),normalization='sym',dtype=torch.float32,num_nodes=num_node)
        L = torch.sparse_coo_tensor(indices=L[0], values=L[1].reshape(-1), size=[num_node, num_node], dtype=torch.float32).to_dense()
        
        self.loss2 = torch.trace(torch.mm(torch.mm(self.x.T,L),self.x)).mean()
        
        if self.rn_weight is not None: 
            celoss = torch.mul(self.rn_weight,self.ce(x,y)).mean()
        else:
            celoss = self.ce(x,y).mean()
        
        loss = celoss + self.lambda1*self.loss1 + self.lambda2*self.loss2
        return loss
    