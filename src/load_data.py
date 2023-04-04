import random
import codecs
import copy
import math
import os,sys

import torch
import torch.nn.functional as F
import numpy as np
# from utils import index2dense
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from MyOwnDataset import MyOwnDataset
from sklearn.model_selection import train_test_split
#access a quantity-balanced training set: each class has the same training size train_each
def get_split(opt,all_idx,all_label,nclass = 10):
    train_each = opt.train_each
    valid_each = opt.valid_each

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break

    assert sum(train_list)==train_each*nclass
    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node

#access a quantity-imbalanced training set; the training set follows the step distribution.
def get_step_split(opt,all_idx,all_label,nclass=7):
    
    if opt.data_name == 'myowndataset':
        train_idx = all_idx
        valid_idx,test_idx = train_test_split(all_idx,test_size=opt.test_ratio)
        
        train_node = [np.where(all_label==i)[0].tolist() for i in range(all_label.max()+1)]
        return train_idx,valid_idx,test_idx,train_node
    
    base_valid_each = opt.valid_each

    imb_ratio = opt.imb_ratio
    head_list = opt.head_list if len(opt.head_list)>0 else [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * opt.labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1) 

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list: 
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    assert sum(valid_list)==total_valid_size
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node


#return the ReNode Weight
def get_renode_weight(opt,data):

    ppr_matrix = data.Pi  #personlized pagerank  ppr即求出来的P,（2708*2708）
    gpr_matrix = data.gpr.float() #class-accumulated personlized pagerank ppr即将其转化为,（2708*7）
    
    base_w  = opt.rn_base_weight
    scale_w = opt.rn_scale_weight
    nnode = ppr_matrix.size(0)   # 2708个节点数目
    unlabel_mask = data.train_mask.int().ne(1)#unlabled node，ne是将tensor中的元素与1对比，不等返回True


    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1) # 对列求和
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix # 将样本类sum(Pi)减去其本类的Pk,对应公式中的j != yv
    rn_matrix = torch.mm(ppr_matrix,gpr_rn) # 将 pagerank * 转移概率，(2708,2708) * (2708,7) = (2708,7)
    
    label_matrix = F.one_hot(torch.clamp(data.y, min=0),gpr_matrix.size(1)).float() 
    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99 #exclude the influence of unlabeled node
    
    #computing the ReNode Weight
    train_size    = torch.sum(data.train_mask.int()).item()
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]
    
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * data.train_mask.float()
   
    return rn_weight


#loading the processed data
def load_processed_data(opt,shuffle_seed = 0, ppr_file=''):
    
    data_name = opt.data_name
    print("\nLoading {} data with shuffle_seed {}".format(data_name,shuffle_seed))

    
    data_path = opt.data_path
    if opt.data_name == 'myowndataset':
        target_dataset = MyOwnDataset(root=data_path+'//MyOwnDataset')
    elif opt.data_name == 'cora':
        target_dataset = Planetoid(data_path, name='Cora')

    target_data=target_dataset[opt.Graph_index]


    #the random seed for the dataset splitting
    if opt.data_name == 'myowndataset':    # 自己数据集中0为unlabelled node
        mask_list = [i for i in range(target_data.num_nodes) if target_data.y[i]!=0]
        target_data.y = target_data.y-1
    else:
        mask_list = [i for i in range(target_data.num_nodes)]
        
    target_data.num_classes = np.max(target_data.y.numpy())+1    
    random.seed(shuffle_seed)
    random.shuffle(mask_list)

    #none = quantity-balance; step = step quantity-imbalance
    if opt.size_imb_type == 'none':
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_split(opt,mask_list,target_data.y.numpy(),nclass=target_data.num_classes)
    elif opt.size_imb_type == 'step':
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_step_split(opt,mask_list,target_data.y.numpy(),nclass=target_data.num_classes)


    target_data.train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.valid_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.test_mask  = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    target_data.train_mask_list = train_mask_list

    target_data.train_mask[torch.tensor(train_mask_list).long()] = True
    target_data.valid_mask[torch.tensor(valid_mask_list).long()] = True
    target_data.test_mask[torch.tensor(test_mask_list).long()]   = True

    
    # calculating the Personalized PageRank Matrix if not exists.
    ppr_file = opt.data_path + opt.data_name +'.pt'
    if os.path.exists(ppr_file):
        target_data.Pi = torch.load(ppr_file)
    else: 
        pr_prob = 1 - opt.pagerank_prob
        A = torch.tensor(to_scipy_sparse_matrix(target_data.edge_index).toarray())
        A_hat   = A.to(opt.device) + torch.eye(A.size(0)).to(opt.device) # add self-loop
        D       = torch.diag(torch.sum(A_hat,1))
        D       = D.inverse().sqrt()
        A_hat   = torch.mm(torch.mm(D, A_hat), D)
        target_data.Pi = pr_prob * ((torch.eye(A.size(0)).to(opt.device) - (1 - pr_prob) * A_hat).inverse())
        target_data.Pi = target_data.Pi.cpu()
        torch.save(target_data.Pi,ppr_file)   

    # calculating the ReNode Weight
    gpr_matrix = [] # the class-level influence distribution
    for iter_c in range(target_data.num_classes):
        # 将每个样本（列）的每个class（行）求均值代表其为该class概率
        iter_Pi = target_data.Pi[torch.tensor(target_data.train_node[iter_c]).long()] # iter_c这个class的P矩阵的行,k*2708
        iter_gpr = torch.mean(iter_Pi,dim=0).squeeze()  # 对行求均值，（1*2708），squeeze后变为（2708）
        gpr_matrix.append(iter_gpr)  # 将不同class的样本Pi均值加入该列表，变为存有7个calss tensor的list（）

    temp_gpr = torch.stack(gpr_matrix,dim=0) # 将gpr_matrix由存有class num个tensor的list变为 tensor
    temp_gpr = temp_gpr.transpose(0,1)  # 行列互换
    target_data.gpr = temp_gpr  # 2708*7,每行代表一个样本的
        
    target_data.rn_weight =  get_renode_weight(opt,target_data) #ReNode Weight 

    
    return target_data


