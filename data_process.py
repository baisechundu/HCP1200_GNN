# /usr/bin/python3
# -*- coding:utf-8 -*-
"""
# @time: 2021/12/9 23:18
# @Author: baiyang
# Email: 308982012@qq.com
# File: data_process.py
# @software: PyCharm
"""
import os

import numpy as np
import torch

BRAIN_CUT_DATA_PATH = r"./辅助数据文件/dataset/cut_brain_data"
if os.path.exists(BRAIN_CUT_DATA_PATH):
    pass
else:
    BRAIN_CUT_DATA_PATH = r"./cut_brain_data"


# 读取脑区数据, 将每个脑区数据块处理为固定时间(6s,约8个时间节点)的滑动窗口
def Normalize(data):
    assert data.shape[-1] == 360, "shape of data are not right!"
    tmp_data = data.transpose(1, 0)
    row_max, row_min = np.max(tmp_data, 1), np.min(tmp_data, 1)
    shrink_size = np.power(row_max - row_min, -1).flatten()
    shrink_arr = np.diag(shrink_size)
    res = np.matmul(shrink_arr, tmp_data - row_min.reshape(row_min.shape[0], -1))
    return res.transpose(1, 0)


# 计算邻接矩阵,暂时用三维空间坐标代替
def adj_matrix(adj_csv_path=r".\辅助数据文件\HCP-MMP1_UniqueRegionList.csv"):
    data = np.loadtxt(open(adj_csv_path, "r"), delimiter=",", skiprows=1, usecols=[-4, -3, -2])
    adj_normalized = np.corrcoef(data)
    node_num = adj_normalized.shape[0]

    top_k_idx = [adj_normalized[i].argsort()[::-1][0:4] for i in range(node_num)]
    mask_matrix = np.array([[1 if i in top_k_idx[j] else 0 for i in range(node_num)] for j in range(node_num)])

    # 由于只取相邻9个脑区的相关系数,因此构造一个mask矩阵将邻接距离超过9的系数置为0,再与原邻接矩阵进行Hadamard乘积
    # mask_matrix = np.array([[1 if abs(i - j) < 2 else 0 for i in range(node_num)] for j in range(node_num)])
    adj = torch.tensor(mask_matrix * adj_normalized).to(torch.float32)


    # 针对graphSAGE\SortPooling模型
    # non_zero_loc = np.nonzero(adj != 0)
    # adj_array = np.vstack((non_zero_loc[0], non_zero_loc[1]))
    # adj_array = adj_array.T
    # edges = []
    # for i, x in enumerate(adj_array):
    #   edges.append([adj_array[i][0], adj_array[i][1]])  # 若A->B有变 则B->A 也有边
    #   edges.append([adj_array[i][1], adj_array[i][0]])  # 给的数据是没有从0开始需要转换
    # edges = torch.tensor(edges, dtype=torch.int64).T
    # edges = torch.tensor(non_zero_loc, dtype=torch.int64).T

    # return edges
    return adj


# 针对每个数据块,处理为滑动窗口
def slide_window_data(single_data_path, label="None"):
    res = []
    brain_data = np.load(single_data_path)
    #     try:
    all_brain_data = np.concatenate((brain_data["left_brain"], brain_data["right_brain"]), axis=1)
    all_brain_data = Normalize(all_brain_data)
    #     print(all_brain_data.shape, all_brain_data_nor.shape)
    duration = all_brain_data.shape[0]
    # for i in range(duration - 7):
    #    assert type(label) == int, "The type of label is wrong, should be int"
    #    res.append((all_brain_data[i:i + 8, :], label))
    for i in range(0, duration - 13, 14):
        assert type(label) == int, "The type of label is wrong, should be int"
        res.append((all_brain_data[i:i + 14, :], label))
        if i != duration - 14:
            res.append((all_brain_data[duration - 14:, :], label))
    #     except:
    #         print("The data block may not have this label")
    #         print(single_data_path)

    return res


# 划分训练集,验证集和测试集
def load_data(dataset_path, label_dict):
    all_data = []
    cut_data_list = os.listdir(dataset_path)
    for file in cut_data_list:
        if "Cue" in file or "gambling" in file:
            continue
        else:
            file_path = os.path.join(BRAIN_CUT_DATA_PATH, file)
            file_label = label_dict[file.split("_")[2]]
            res = slide_window_data(file_path, file_label)
            all_data.extend(res)

    # 将数据特征和标签文件分离
    feature, label = list(map(np.array, zip(*all_data)))
    feature_arr, label_array = torch.tensor(feature).permute(0, 2, 1), torch.tensor(label).unsqueeze(1)

    # 设置种子数, 划分训练集、验证集和测试集
    sample_num = len(label)
    np.random.seed(200)
    index_list = [i for i in range(sample_num)]
    np.random.shuffle(index_list)

    train_size, test_size = int(sample_num * 0.7), int(sample_num * 0.15)

    train_index = index_list[:train_size]
    val_index = index_list[train_size:train_size + test_size]
    test_index = index_list[train_size + test_size:]

    # train_data, train_label = feature_arr[train_index], label_array[train_index]
    # val_data, val_label = feature_arr[val_index], label_array[val_index]
    # test_data, test_label = feature_arr[test_index], label_array[test_index]
    return feature_arr, label_array, train_index, val_index, test_index


# if __name__ == '__main__':
#    adj = adj_matrix()
#    print(adj)
#    features, labels, idx_train, idx_val, idx_test = load_data(BRAIN_CUT_DATA_PATH, DataCut.task_to_label_dict())
#    train_data = features[idx_train, :]
#    train_labels = labels[idx_train]
#    print(train_data.shape, train_labels.shape)
#    train_dataset = MyDataset(train_data, train_labels)
#    tensor_dataloader = DataLoader(dataset=train_dataset,
#                                   batch_size=32,
#                                   shuffle=True,
#                                   num_workers=0)
