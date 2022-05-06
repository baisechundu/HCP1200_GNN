# -*-coding:utf-8 -*-
"""
@user: baiyang
@email: baiyang01@qianxin.com
@time: 2022/4/27 11:48
@file: load_data.py
@desc: 
"""
import argparse
import os
import warnings

import numpy as np
import torch
from torch_geometric.data import DataLoader

from imports.HCP1200Dataset import HCPDataset
from imports.utils import train_val_test_split

warnings.filterwarnings('ignore')
# cur_dir = os.path.dirname(os.path.abspath(__file__))
cur_dir = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str,
                    default=os.path.join(cur_dir, '辅助数据文件/dataset/raw_bak'),
                    help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')

#################### Parameter Initialization #######################
name = "hcp1200"
opt = parser.parse_args()

fold = opt.fold
data_path = opt.datapath
data_name_list = np.array(os.listdir(data_path))
tr_index, val_index, te_index = train_val_test_split(fold=fold, data_address=data_path)
train_data_name = data_name_list[tr_index]
test_data_name = data_name_list[te_index]
val_data_name = data_name_list[val_index]
print(train_data_name, '\n', test_data_name, '\n', val_data_name)
################## Parameter Initialization Ended #####################

dataset = HCPDataset(data_path, name)
dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0

tr_index, val_index, te_index = train_val_test_split(fold=fold)
train_dataset = dataset[tr_index]
val_dataset = dataset[val_index]
test_dataset = dataset[te_index]

train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

for data in train_loader:
    print(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
