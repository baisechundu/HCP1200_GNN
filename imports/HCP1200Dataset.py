# -*-coding:utf-8 -*-
"""
@user: baiyang
@email: baiyang01@qianxin.com
@time: 2022/4/27 12:11
@file: HCP1200Dataset.py
@desc: 
"""
import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

from read_hcp1200_stats_parall import read_data
from torch_geometric.loader import DataLoader

class HCPDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(HCPDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.name = name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in os.listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}'.format(self.name)


if __name__ == '__main__':
    cur_path = os.path.dirname(os.getcwd())
    print(cur_path)
    name = "HCP1200"
    data_test = HCPDataset(osp.join(cur_path, "辅助数据文件/dataset"), name)
    print(data_test.data)
    train_loader = DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)
    for data in train_loader:
        print(data)
              # data.edge_index, data.batch, data.edge_attr, data.pos)

