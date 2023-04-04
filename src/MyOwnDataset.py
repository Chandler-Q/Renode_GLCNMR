# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:39:34 2023

@author: c
"""

# ======== 基础package =========
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ======== torch ===========
import torch

# ======== geometric =======
from torch_geometric.data import Data,InMemoryDataset


''' Dataset establish
        
        为了构建 dataset 需要将所有的节点筛选，进行编码，构建边，以及节点特征矩阵构建
        
'''

                     

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root='data', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # A list of files in the raw_dir which needs to be found in order to skip the download.
        # 原数据要放在raw内
        return ['temp.csv']

    @property
    def processed_file_names(self):
        # A list of files in the processed_dir which needs to be found in order to skip the processing.
        return ['MyDataset.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # url="https://data.cityofchicago.org/resource/sxs8-h27x.csv?$where=time between '2023-01-01T00:00:00' and '2023-01-31T00:00:00'"
        pass

    def process(self):
        # Read data into huge `Data` list.
        # all Graph need to be saved in the follow list
        data_list = []
        
        df = pd.read_csv('.\\'+self.root+'\\raw\\'+ self.raw_file_names[0])
        # df = pd.read_csv(r'..\data\raw\temp.csv')
        df['TIME'] = pd.to_datetime(df['TIME'])
        
        def encode_y(x):
            if x==-1:
                return 0
            elif x>=0 and x<10:
                return 1
            elif x>=10 and x<20:
                return 2
            elif x>=20 and x<30:
                return 3
            elif x>=30 and x<40:
                return 4
            else:
                return 5
        # 数据编码
        df['SPEED'] = df['SPEED'].map(encode_y)
        le = LabelEncoder()
        df['DIRECTION'] = le.fit_transform(df['DIRECTION'])
        df['STREET'] = le.fit_transform(df['STREET'])
        df['FROM_STREET'] = le.fit_transform(df['FROM_STREET'])
        df['TO_STREET'] = le.fit_transform(df['TO_STREET'])
        # ID也需要重新排列
        df['SEGMENT_ID'] = le.fit_transform(df['SEGMENT_ID'])
        
        # node features name
        node_features_name = ['DIRECTION', 'FROM_STREET','TO_STREET', 'LENGTH', 
                         'BUS_COUNT','MESSAGE_COUNT','HOUR', 'DAY_OF_WEEK', 'MONTH']

        
        Points = pd.concat([df['START_LOCATION'],df['END_LOCATION']],ignore_index=True).unique()
        
        # 点与ID 对应字典，Point:ID, str:int
        ID_dic = dict()
        
        for i in range(len(Points)):
            ID_dic[Points[i]] = i
        
        df['START_LOCATION'] = df.START_LOCATION.map(ID_dic)
        df['END_LOCATION'] = df.END_LOCATION.map(ID_dic)
        
        def encode_edge_index(temp_df):
            temp_df.reset_index(inplace = True)
            nodes_num = len(temp_df)
            start_node = []
            end_node =[]
            
            # 以SEGMENT_ID为点的ID
            for i in range(nodes_num):
                end = temp_df.loc[i,'END_LOCATION']                 # 找到该条记录的end_location
                start = temp_df[temp_df['START_LOCATION']==end]     # 找到以所找到的end ID 为start的记录
                start_node.extend([temp_df.loc[i,'SEGMENT_ID']]*len(start))                 # 开始节点是以 end 为开始点，以start未开始的各项SEGMENT_ID
                end_node.extend(start['SEGMENT_ID'].values)
        

            return np.array([start_node,end_node])
        
        
        # 将每次更新的数据都作为一个Data输出
        grouped_df = df.groupby(by = 'SEGMENT_ID',axis=0)
        for i in range(144):
            temp_df = grouped_df.take([i])
            
            # x feature, the arcs of street as the node 
            x = torch.tensor(temp_df[node_features_name].values,dtype=torch.float32)
            
            # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(encode_edge_index(temp_df))
            

            
            # y (torch.Tensor) real label 
            y = torch.tensor(np.array(temp_df.SPEED.values))
            
            
            # apeend the data into Dataset
            data_list.append(Data(x=x, edge_index = edge_index,  y = y))
        
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])





