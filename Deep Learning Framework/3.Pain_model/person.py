# -*- coding: utf-8 -*-
"""
Training network
==============
**Author**: `zhibin Li`__
#python person.py

"""
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from dataset import dataload
import pandas as pd
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123123)  # #6折交叉，交叉No.
parser.add_argument('--train_log_dir', type=str, default='./result')  # #6折交叉，交叉No.
args = parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(args.seed)



    """
    ====================================
    1、Load data
    """

    """
    ====================================
    1、Load data
    """
    train_path = r'/root/data/usr/Zhibin/15.96x96_skins/Alldata/Pain_data'

    trainset = dataload.Gesture_Dataset(root_dir = train_path)


    # 数据集划分
    train_size = int(0.8 * len(trainset))
    vali_size = len(trainset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(trainset, [train_size, vali_size])

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=60,
                                             shuffle=True, num_workers=1)


    validation_data = torch.utils.data.DataLoader(validation_dataset, batch_size=60,
                                                  shuffle=False, num_workers=1)




    """
    ====================================
    2、Person
    """
    all_features = []
    all_labels = []
    for step, data in enumerate(train_data):

        x = data[0] #[60,3]
        y = data[1] #[60,1]

        all_features.append(x)
        all_labels.append(y)

    # 拼接
    X = torch.cat(all_features, dim=0)  # (N, 3)
    Y = torch.cat(all_labels, dim=0)  # (N, 1)

    # 转 numpy
    X = X.cpu().numpy()
    Y = Y.cpu().numpy().reshape(-1)

    # 拼成 DataFrame，便于计算和展示
    df = pd.DataFrame(
        np.hstack([X, Y.reshape(-1, 1)]),
        columns=["feature1", "feature2", "feature3", "Y"]
    )

    # Pearson 相关系数矩阵
    print(df.corr(method="pearson"))



    #
    #     all_f1.append(feature1.detach().cpu().numpy())
    #     all_f2.append(feature2.detach().cpu().numpy())
    #     all_f3.append(feature3.detach().cpu().numpy())
    #     all_y.append(Y.detach().cpu().numpy())
    #
    # # 拼接
    # f1 = np.concatenate(all_f1).flatten()
    # f2 = np.concatenate(all_f2).flatten()
    # f3 = np.concatenate(all_f3).flatten()
    # y = np.concatenate(all_y).flatten()

    # # 计算相关系数矩阵
    # corr_matrix = np.corrcoef([f1, f2, f3, y])
    # print(corr_matrix)