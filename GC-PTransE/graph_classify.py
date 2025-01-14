from __future__ import division
from __future__ import print_function  # 输出函数

import time
import argparse

import torch.optim as optim
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pynvml
from pathlib import Path
import os
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):  # 对mx节点做归一化
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对结点（矩阵）按行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.  # 如果是无穷大（0的倒数），转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_positive_negative_predictions(output, threshold=0.5):
    # 将概率值转换为二分类结果
    predictions = output.argmax(dim=1)
    # 根据阈值判断正负例子
    positive_predictions = (output[:, 1] >= threshold).nonzero().squeeze()
    negative_predictions = (output[:, 0] >= threshold).nonzero().squeeze()
    return positive_predictions, negative_predictions

# 在测试阶段计算混淆矩阵
def calculate_confusion_matrix(output, labels):
    # 将概率值转换为预测结果
    predictions = output.argmax(dim=1)
    # 获取类别数量
    num_classes = output.shape[1]
    # 初始化混淆矩阵为全零矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes)
    # 计算混淆矩阵中的交叉频数
    for i in range(len(labels)):
        confusion_matrix[labels[i], predictions[i]] += 1
    return confusion_matrix

def find_misclassified_samples(output, labels):
    # 将概率值转换为预测结果
    predictions = output.argmax(dim=1)

    # 找到被误判的数据的索引
    misclassified_indices = (predictions != labels).nonzero().squeeze()

    # 找到被误判成的类别
    misclassified_classes = predictions[misclassified_indices]

    return misclassified_indices, misclassified_classes


class Graph_Classify:


    def load_data(self, file_path):

        """Load citation network dataset (cora only for now)"""
        # 打印Loading dataset
        # print('Loading {} dataset...'.format(dataset))
        print('Loading {} dataset...'.format(file_path))
        # 导入content文件
        idx_features_labels = np.genfromtxt("{}.content".format(file_path),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 取特征，第一列到倒数第二列
        # print("Labels array:", idx_features_labels[:, -1])
        labels = encode_onehot(idx_features_labels[:, -1])  # 取出所属类别

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 取出节点索引  idx
        idx_map = {j: i for i, j in enumerate(idx)}  # 构造节点的索引字典，将索引转换成从0-整个索引结长度的字典编号 idx_map
        edges_unordered = np.genfromtxt("{}.cites".format(file_path),  # 导入edge数据  表示两个编号节点之间有一条边
                                        dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)  # 将之前的edges_unordered编号转换成idx_map字典编号后的边
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # 构建边的邻接矩阵：有边为1，没边为0
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix  计算转置矩阵  将有向图转成无向图
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)  # 对特征做了归一化的操作  normalize归一化函数:按行做了归一化
        adj = normalize(adj + sp.eye(adj.shape[0]))  # 对A+I做归一化的操作  adj + sp.eye(adj.shape[0]):邻接矩阵加上单位阵的操作

        # 20
        # idx_train = range(130)
        # idx_val = range(130, 170)
        # idx_test = range(170, 215)


        # idx_train = range(80)
        # idx_val = range(80, 106)
        # idx_test = range(106, 132)
        # 40
        # idx_train = range(260)
        # idx_val = range(260, 345)
        # idx_test = range(345, 430)
        # # #
        # 60
        # idx_train = range(390)
        # idx_val = range(390, 515)
        # idx_test = range(515, 645)

        # 80
        # idx_train = range(505)
        # idx_val = range(505, 675)
        # idx_test = range(675, 843)
        # # # #

        # #100
        # idx_train = range(645)
        # idx_val = range(645, 859)
        # idx_test = range(859, 1073)

        #100
        idx_train = range(2290)
        idx_val = range(2290, 2576)
        idx_test = range(2576, 2856)


        # 将numpy的数据转换成torch格式
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test


    # 定义训练模型
    def train(self, epoch, model,optimizer, features, adj, idx_train, labels, args, idx_val, idx_test):
        t = time.time()
        model.train()
        optimizer.zero_grad()  # 梯度清零
        output = model(features, adj)  # 运行模型，输入参数（features,ad）
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 损失函数
        acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
        loss_train.backward()  # 反向传播
        optimizer.step()  # 更新梯度

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


    # 定义测试模型
    def test(self, model, features, adj, idx_test, labels):
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU {i} 使用率: {util.gpu}%")

        pynvml.nvmlShutdown()

        model.eval()
        output = model(features, adj)
        positive_predictions, negative_predictions = get_positive_negative_predictions(output, threshold=0.5)

        # 输出正负例子结果
        print("Positive Predictions:", positive_predictions)
        print("Negative Predictions:", negative_predictions)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        # 获取混淆矩阵
        confusion_matrix = calculate_confusion_matrix(output, labels)
        # 打印混淆矩阵
        print("Confusion Matrix全部:")
        print(confusion_matrix)

        misclassified_indices, misclassified_classes = find_misclassified_samples(output, labels)

        for i in range(len(misclassified_indices)):
            index = misclassified_indices[i].item()
            true_class = labels[index].item()
            misclassified_class = misclassified_classes[i].item()
            print(f"Node {index} - True Class: {true_class}, Misclassified as Class {misclassified_class} ")
            # print(f"Node {index} - True Class: {true_class}, Misclassified as Class: {misclassified_class} (Index in All Data: {index})")

        # 输出被误判的数据的索引
        print("Misclassified Samples Index: ", misclassified_indices)

        preds = output[idx_test].max(1)[1].type_as(labels)

        # print(len(preds))
        # print(len(labels[idx_test]))
        # print(np.array(preds))
        # print(np.array(labels[idx_test]))

        y_pred = np.array(preds)
        y_test = np.array(labels[idx_test])

        # f1_score = calculate_f1_score(y_test, y_pred)
        # print("F1 Score:", f1_score)
        # sklearn.metrics.precision_score(y_test, y_pred, labels=None, pos_label=1,
        #                                 average='binary', sample_weight=None)

        # 准确率
        accuracy_result = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        print("accuracy:%s" % accuracy_result)
        print("测试准确率: {:.3f}%".format(accuracy_result * 100))

        # 精确率
        percision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
        print("precision:%s" % percision_result)
        print("测试精确率: {:.3f}%".format(percision_result * 100))

        # 召回率
        recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
        print("recall:%s" % recall_result)
        print("测试召回率: {:.3f}%".format(recall_result * 100))

        # F1-score
        f1_score_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
        print("f1_score:%s" % f1_score_result)
        print("测试F1-score值: {:.3f}%".format(f1_score_result * 100))
        #
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))


# GCN模型
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # 定义两层GCN
        self.gc1 = GraphConvolution(nfeat, nhid)  # 构造第一层GCN
        self.gc2 = GraphConvolution(nhid, nclass)  # 构造第二层GCN
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)  # 对每一个节点做softmax


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 随机化参数
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)  # 稀疏矩阵的相乘
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


