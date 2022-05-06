# -*-coding:utf-8 -*-
"""
@user: baiyang
@email: baiyang01@qianxin.com
@time: 2022-04-26
@file: utils.py
@desc:
"""
import os
import random

import h5py
import networkx as nx
import nibabel as nib
import numpy as np
import scipy.io as sio
import torch
from networkx.convert_matrix import from_numpy_matrix
from nilearn import connectome
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

from data_process import slide_window_data

cur_path = os.path.dirname(__file__)
print(f"当前路径是{cur_path}")


def read_label(label_path):
    index_list = []
    brain_label = nib.load(label_path)
    brain_arr = np.asanyarray(brain_label.dataobj, dtype=np.int)
    for i in range(1, 181):
        temp_index = np.where(brain_arr[0] == i)
        index_list.extend(temp_index)
    return index_list


def voxel_to_brain_arr(brain_index_lists, voxel_feature_arr):
    def transformation(target_arr):
        return np.vectorize(lambda group: np.mean(target_arr[group]), otypes=[np.float32])(brain_index_lists)

    res = np.apply_along_axis(transformation, 1, voxel_feature_arr)
    return res


def get_timeseries(Left_BRAIN_INDEX_LISTS,
                   Right_BRAIN_INDEX_LISTS,
                   data_path):
    subject_IDs = []
    timeseries = []
    filename_list = [element for element in os.listdir(data_path) if element.isdigit()]
    for filename in filename_list:
        subject_IDs.append(filename)
        print(os.path.join(data_path, filename))

        sub_path = os.path.join(data_path, filename)
        # os.chdir(sub_path)
        filename = 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
        file_path = os.path.join(sub_path, filename)
        img = nib.load(file_path)
        data = img.get_data()
        data_left = data[:, :29696]
        data_right = data[:, 29696:59412]

        left_roi_data = voxel_to_brain_arr(Left_BRAIN_INDEX_LISTS, data_left)
        right_roi_data = voxel_to_brain_arr(Right_BRAIN_INDEX_LISTS, data_right)

        roi_data = np.vstack((left_roi_data.T, right_roi_data.T)).T
        timeseries.append(roi_data)

    return timeseries, subject_IDs


def subject_connectivity(timeseries, subjects, kind, save=True,
                         save_path=os.path.join(cur_path, '../辅助数据文件/dataset/resting_HCP')):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['correlation', 'partial correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform(timeseries)

    if save:
        for i, subj_id in enumerate(subjects):
            subject_file = os.path.join(save_path, subj_id,
                                        subj_id + '_' + kind.replace(' ', '_') + '.mat')
            sio.savemat(subject_file, {'connectivity': connectivity[i]})
        # total_connectivity = np.sum(connectivity, 0)
        # row, col = np.diag_indices_from(total_connectivity)
        # total_connectivity[row, col] = 1
        return connectivity  # , total_connectivity


def trans_adj(subject_IDs, save_path=os.path.join(cur_path, '../辅助数据文件/dataset')):
    for subject in subject_IDs:
        os.chdir(os.path.join(save_path, 'raw'))
        f = h5py.File(subject + '.h5', 'r')
        adj_matrix = f['adj'][:]
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

        att = f['fea'][:]
        label = f['label'][:]

        att_torch = torch.from_numpy(att).float()
        y_torch = torch.from_numpy(np.array(label)).long()  # classification

        data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)


def load_data(dataset_path, subject, label_dict):
    cut_data_list = os.listdir(dataset_path)
    sub_file_list = []
    sub_data, sub_label = [], []
    for file_num in range(0, len(cut_data_list)):
        if cut_data_list[file_num].split("_")[0] == subject:
            sub_file_list.append(cut_data_list[file_num])
    for file in sub_file_list:
        if "Cue" in file or "gambling" in file:
            continue
        else:
            file_path = os.path.join(dataset_path, file)
            file_label = label_dict[file.split("_")[1]]
            res = slide_window_data(file_path, file_label)
            sub_data.extend(res)

    return sub_data


def save_atrr(dataset_path,
              subject_IDs,
              label_dict,
              kind,
              variable='connectivity',
              adj_path=os.path.join(cur_path, '../辅助数据文件/dataset/resting_HCP'),
              save_path=os.path.join(cur_path, '../辅助数据文件/dataset', )):
    for i, subject in enumerate(subject_IDs):
        sub_data = load_data(dataset_path, subject, label_dict)

        # 特征和标签
        data = np.zeros((len(sub_data), len(sub_data[0][0]), len(sub_data[0][0][0])))
        label = np.zeros(len(sub_data), )
        for data_len in range(len(sub_data)):
            data[data_len, :, :] = sub_data[data_len][0]
            label[data_len,] = sub_data[data_len][1]

        # 邻接矩阵
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        fl = os.path.join(adj_path, subject, subject + "_" + kind.replace(' ', '_') + ".mat")
        adj_matrix = sio.loadmat(fl)[variable]

        if not os.path.exists(os.path.join(save_path, 'raw')):
            os.makedirs(os.path.join(save_path, 'raw'))
        os.chdir(os.path.join(save_path, 'raw'))
        f = h5py.File(subject + '.h5', 'w')
        f['fea'] = data
        f['adj'] = adj_matrix
        f['label'] = label  # 读数据 f = h5py.File(subject + '.h5','r')   f['label'][:]
        f.close()


def train_val_test_split(kfold=5,
                         fold=0,
                         seed_num=111,
                         data_address=os.path.join(cur_path, '../辅助数据文件/dataset/raw_bak')):
    files = os.listdir(data_address)
    file_num = len(files)
    id = list(range(file_num))


    random.seed(seed_num)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123, shuffle=True)
    kf2 = KFold(n_splits=kfold - 1, shuffle=True, random_state=666)

    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id


def readdata(data_dir):
    onlyfiles = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    onlyfiles.sort()
    for single_file in onlyfiles:
        single_data = h5py.File(single_file, 'r')
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

        att = single_data['fea'][:]
        label = single_data['label'][:]

        att_torch = torch.from_numpy(att).float()
        y_torch = torch.from_numpy(np.array(label)).long()

