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

import os
from PCRA_Program.PCRA import PCRA
from GC_PTransE.TestRun import TestRun
from GC_PTransE.TrainRun import TrainRun
import graph_classify
from graph_classify import GCN
from graph_classify import Graph_Classify

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

def PCRA_run():
    file_path = "resource_classification/path_data/1_APT-C-06/confident.txt"
    if not os.path.exists(file_path):
        pcra = PCRA()
        pcra.run()

if __name__ == '__main__':

    print("Classfify:")

    # Training settings  一些参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=28,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Create an instance of the class
    graph_classifier = Graph_Classify()

    # 获取当前文件（graph.py）所在目录
    current_dir = os.path.dirname(__file__)


    file_path1 = os.path.join(current_dir, 'data', 'cve_ioc4个关系')

    # Load data  导入数据集
    adj, features, labels, idx_train, idx_val, idx_test = graph_classifier.load_data(file_path1)

    # Model and optimizer  构造GCN模型，初始化参数，两层GCN
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),  # 构造optimizer
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model  训练过程
    t_total = time.time()
    for epoch in range(args.epochs):
        graph_classifier.train(epoch, model, optimizer, features, adj, idx_train, labels, args, idx_val, idx_test)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    graph_classifier.test(model, features, adj, idx_test, labels)

    print("Predict Train or test? y/n")
    train_flag = input().strip() == "y"
    if train_flag:
        PCRA_run()
        train_run = TrainRun()
        train_run.train_run()
    else:
        test_run = TestRun()
        test_run.test_run()



