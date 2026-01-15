# -*- coding: utf-8 -*-
"""
Test TSNE network
==============
**Author**: `zhibin Li`__
#python test_TSNE.py

"""
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
import time
import random
import math
import argparse
from torch.optim import lr_scheduler
import argparse
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter

from dataset import dataload
import model.Models as Models

from sklearn.metrics import davies_bouldin_score


from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123123)
parser.add_argument('--train_log_dir', type=str, default='./result')
args = parser.parse_args()

log_path = os.path.join(args.train_log_dir, "log")
write = SummaryWriter(log_dir=log_path)  # log file


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
    0、Training parameters
    """
    # Number of workers for dataloader
    workers = 10

    # Batch size during training
    batch_size = 20  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    # Number of training epochs
    num_epochs = 3

    # Learning rate for optimizers
    lr = 0.001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Compress factor
    Compress_Q = 1 # 1

    """
    ====================================
    1、Load data
    """

    """
    ====================================
    1、Load data
    """
    data_path = r'/root/data/usr/Zhibin/15.96x96_skins/Alldata/Obj_data'
    dataset = dataload.Object_Dataset(data_path, Compress_Q = Compress_Q)

    all_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)




    """
    ====================================
    2、Load model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the generator
    net = Models.CNN_3D_Lite(num_classes=10).to(device)  #CNN_3D_Lite
    net = nn.DataParallel(net, list(range(ngpu)))
    print('device', device)
    print(net)

    state_dict = torch.load('./result/'+'/Compress_Q='+ str(Compress_Q)+'.pth', weights_only=True)
    net.load_state_dict(state_dict)

    """
    ====================================
    2、Feature extraction
    """


    x = torch.randn(10, 30, 96, 96).to(device)
    features = net(x,return_feature=True)
    print(features.shape)




    """
    ====================================
    Test
    """
    net.eval()
    epoch_loss = 0.0
    acc = 0.0

    feture_all = []
    lable_all = []

    with torch.no_grad():
        for val_data in all_data:
            val_sensor, val_labels = val_data
            val_labels = val_labels.view(val_labels.size(0))
            val_sensor = val_sensor.to(device, dtype=torch.float)  # 转换统一数据格式
            val_labels = val_labels.to(device, dtype=torch.long)


            feture_all.append(net(val_sensor,return_feature=True).cpu().numpy())
            lable_all.append(val_labels.cpu().numpy())


            outputs = net(val_sensor,return_feature=False)
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()


        val_num = dataset.__len__()  # 测试集总数
        val_accurate = acc / val_num


        print('test_accuracy: %.3f \n' %
              (val_accurate))



        feture_all = np.concatenate(feture_all, axis=0)
        lable_all = np.concatenate(lable_all, axis=0)

    """
    ====================================
    T-SNE
    """

    print('feture_shape:', feture_all.shape)
    print('lable_shape:', lable_all.shape)

    X_2d = TSNE(
        n_components=2,
        perplexity = 30,  # 强化局部拉伸
        learning_rate=500,  # 更平稳收敛
        init='pca',  # 初始分布更合理
        random_state=1212
    ).fit_transform(feture_all)


    # Step 2: 计算 DBI
    dbi = davies_bouldin_score(X_2d, lable_all)
    print("Davies-Bouldin Index (DBI):", dbi)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    labels_unique = np.unique(lable_all)

    # 每个类别分别绘制，用于图例标注
    for label in labels_unique:
        idx = lable_all == label
        plt.scatter(
            X_2d[idx, 0], X_2d[idx, 1],
            c=[cmap(label)],
            label=f'Class {label}',  # 可改为你的类别名称
            s=15
        )

    plt.title("TSNE visualized in 2D")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # 添加右侧图例
    plt.legend(title="Class", loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout()



    plt.savefig(f"./result/tsne_result_Q={Compress_Q}_DBI={dbi:.2f}.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"./result/tsne_result_Q={Compress_Q}_DBI={dbi:.2f}.png", format='png', bbox_inches='tight')

print('Finished Test')