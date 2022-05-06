'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import os
import glob
import h5py

import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
from imports.gdc import GDC


def split(data, batch):
    # node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cumsum(torch.from_numpy(np.array([data.y[k, ].shape[-1] for k in range(data.y.shape[0])])), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = torch.cat([torch.tensor([0]), torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)])

    return data, slices


def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list, att_list = [], [], []

    # parallar computing
    # cores = max(4, multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(processes=cores)
    # pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = list(map(func, onlyfiles))
    # res = pool.map(func, onlyfiles)

    # pool.close()
    # pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1] + j * res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j] * res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch)

    data, slices = split(data, batch_torch)

    return data, slices


def read_sigle_data(data_dir, filename, use_gdc=False):
    single_data = h5py.File(osp.join(data_dir, filename), "r")

    # read edge and edge attribute
    # pcorr = np.abs(temp['pcorr'][()])
    adj_matrix = single_data['adj'][:]

    num_nodes = adj_matrix.shape[0]
    G = from_numpy_matrix(adj_matrix)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = adj_matrix[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = single_data['fea'][()]
    label = single_data['label'][()]

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(), data.edge_index.data.numpy(), data.x.data.numpy(), data.y.data.item(), num_nodes

    else:
        return edge_att.data.numpy(), edge_index.data.numpy(), att, label, num_nodes


if __name__ == "__main__":
    data_dir = '/Users/baiyang/Desktop/programs_python/new_code/辅助数据文件/dataset/raw'
    filename = '118023.h5'
    res = read_sigle_data(data_dir, filename)
    for i in res:
        try:
            print(i.shape)
        except:
            print(i)
            pass

    data, slice = read_data(data_dir)
    print(data)
    print(slice)
