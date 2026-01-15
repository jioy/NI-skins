# -*- coding: utf-8 -*-
"""
Training network
==============
**Author**: `zhibin Li`__
#python train.py

"""
import torch
import torchvision.transforms as transforms
import numpy as np
import argparse
from dataset import dataload
import pandas as pd
import os
import random

import matplotlib
matplotlib.use("Agg")   # 使用无界面后端
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

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
    train_dataset, test_dataset = torch.utils.data.random_split(trainset, [train_size, vali_size])

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=60,
                                             shuffle=True, num_workers=1)


    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=60,
                                                  shuffle=False, num_workers=1)




    """
    ====================================
    2、train
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



    import statsmodels.api as sm

    # 特征和标签
    X_mat = df[["feature1","feature2","feature3"]].to_numpy()
    X_sm = sm.add_constant(X_mat)   # 加上常数项 β0
    y = df["Y"].to_numpy()

    # 拟合 Binomial GLM（logit 链接函数）
    model = sm.GLM(y, X_sm, family=sm.families.Binomial())
    res = model.fit()

    # model = sm.OLS(y, X_sm)
    # res = model.fit()

    print(res.summary())

    # 系数
    beta0, beta1, beta2, beta3 = res.params
    print("β0:", beta0, "β1:", beta1, "β2:", beta2, "β3:", beta3)





    """
    ====================================
    3、test
    """
    all_features = []
    all_labels = []
    for step, data in enumerate(test_data):
        x = data[0]  # [60,3]
        y = data[1]  # [60,1]

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

    beta0, beta1, beta2, beta3 = res.params


    def predict_prob(x1, x2, x3):
        z = beta0 + beta1 * x1 + beta2 * x2 + beta3 * x3
        return 1 / (1 + np.exp(-z))


    # 特征矩阵
    X_mat = df[["feature1", "feature2", "feature3"]].to_numpy()
    Y_true = df["Y"].to_numpy()

    # 预测概率
    Y_pred = predict_prob(X_mat[:, 0], X_mat[:, 1], X_mat[:, 2])

    r2 = r2_score(Y_pred, Y_true)
    print("R²:", r2)

    # ------------------------
    # 4. 绘制散点图 (预测 vs 真实)
    # ------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_true, Y_pred, alpha=0.5, s=20, c="blue", label="Samples")

    # 参考对角线（理想情况 Y_pred=Y_true）
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label="Ideal")

    plt.xlabel("Predicted Y")
    plt.ylabel("True Y")
    plt.title("Predicted vs True (Test set)")
    plt.legend()
    plt.grid(True)

    plt.xlabel("Sample index")
    plt.ylabel("Y")
    plt.title("True vs Predicted (Test set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片
    plt.savefig("test_result.png", dpi=300)  # 保存为PNG



    #保存结果

    # 假设 Y_true, Y_pred 是 numpy 数组或 list
    df = pd.DataFrame({
        "Y_true": Y_true,
        "Y_pred": Y_pred
    })

    # 保存到 Excel
    df.to_excel("prediction_results.xlsx", index=False)