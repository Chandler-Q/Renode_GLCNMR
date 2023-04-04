# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:00:21 2023

@author: c
"""
import torch
import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    
    # traing
    parser.add_argument('--Net', default='GLCNMR', type=str,help='opt.Net must from one of [GLCNMR, GCN, SAGE, GAT]')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--least-epoch', default=20, type=int)
    parser.add_argument('--early-stop', default=20, type=int)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    
    # Dataset and renode-weight
    parser.add_argument('--data-path', default='..//data', type=str, help="data path (dictionary)")
    parser.add_argument('--data-name',  default='myowndataset', type=str, help="data name")#cora myowndataset
    parser.add_argument('--Graph-index',  default=0, type=int, help="the index of myowndataset")#cora myowndataset
    parser.add_argument('--rn-weight',  default=True, type=bool, help="re-node or not")#cora myowndataset
    
    parser.add_argument('--shuffle-seed',  default=123, type=int, help="shuffle seed for valid and test split")
    parser.add_argument('--train-each', default=20, type=int, help="the training size of each class, used in none imbe type")
    parser.add_argument('--valid-each', default=30, type=int, help="the validation size of each class")
    
    # Graph Net setting
    parser.add_argument('--input-dim',  default=9, type=int, help="input features dim")#cora myowndataset
    parser.add_argument('--hidden-dim',  default=32, type=int, help="hidden layer dim,if dataset is myown then hidden is embedding dim")
    parser.add_argument('--output-dim',  default=5, type=int, help="number of class")
    parser.add_argument('--lambda1',  default=0.001, type=int, help="lambda1 for loss1")
    parser.add_argument('--lambda2',  default=0.001, type=int, help="lambda2 for loss1")
    parser.add_argument('--lamb',  default=0.05, type=int, help="lambda in loss1")
    
    
    parser.add_argument('--size-imb-type', default='step', type=str, help="the imbalace type of the training set") #none, step
    parser.add_argument('--labeling-ratio', default=0.01, type=float, help="the labeling ratio of the dataset, used in step imb type")
    parser.add_argument('--head-list',  default=[0,1,2], type=int, nargs='+', help="list of the majority class, used in step imb type")
    parser.add_argument('--imb-ratio',  default=1.0, type=float, help="the ratio of the majority class size to the minoriry class size, used in step imb type") 
    parser.add_argument('--test-ratio',  default=0.5, type=float, help="the ratio of the majority class size to the minoriry class size, used in step imb type") 

    #Pagerank 
    parser.add_argument('--pagerank-prob', default=0.75, type=float,help="probility of going down instead of going back to the starting position in the random walk")
    parser.add_argument('--ppr-topk', default=-1,type=int)
    
    #ReNode
    parser.add_argument('--renode-reweight', '-rr',   default=0,   type=int,   help="switch of ReNode") # 0 (not use) or 1 (use)
    parser.add_argument('--rn-base-weight',  '-rbw',  default=0.5, type=float, help="the base  weight of renode reweight")
    parser.add_argument('--rn-scale-weight', '-rsw',  default=1.0, type=float, help="the scale weight of renode reweight")
    
    
    
    opt = parser.parse_args()
    
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return opt
    # 

if __name__ == '__main__':
    opt = get_opt()
