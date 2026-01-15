# -*- coding: utf-8 -*-
"""
Training network
==============
**Author**: `zhibin Li`__
#python train.py
nohup python train.py > trainlog.log 2>&1 &

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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123123)  # #6折交叉，交叉No.
parser.add_argument('--train_log_dir', type=str, default='./result')  # #6折交叉，交叉No.
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
    batch_size = 20

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    # Number of training epochs
    num_epochs = 16

    # Learning rate for optimizers
    lr = 0.001   # 0.0001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Compress factor
    Compress_Q = 2000

    """
    ====================================
    1、Load data
    """

    """
    ====================================
    1、Load data
    """
    train_path = r'/root/data/usr/Zhibin/15.96x96_skins/Alldata/Gesture_data'

    trainset = dataload.Gesture_Dataset(root_dir = train_path, Compress_Q = Compress_Q)


    # 数据集划分
    train_size = int(0.7 * len(trainset))
    vali_size = len(trainset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(trainset, [train_size, vali_size])

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)


    validation_data = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=workers)



    """
    ====================================
    2、Load model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the generator
    net = Models.CNN_3D_Lite(num_classes=10).to(device)   #CNN_3D_Lite
    net = nn.DataParallel(net, list(range(ngpu)))
    print('device', device)
    print(net)

    """
    ====================================
    3、Initial set
    """

    # Loss Functions and Optimizers
    loss_function = nn.CrossEntropyLoss()

    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)

    savetime = int(time.time())
    save_path = args.train_log_dir + '/Compress_Q=' + str(Compress_Q) + '.pth'

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    best_acc = 0.0
    Accuracy = []

    iteration = 0

    """
    ====================================
    4、Train
    """
    for epoch in range(num_epochs):
        ########################################## train ###############################################

        # 训练集进度条
        train_loader_tqdm = tqdm(train_data, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        epoch_loss = 0.0
        acc = 0.0
        for step, data in enumerate(train_loader_tqdm):  # 遍历训练集，step从0开始计算

            net.train()  # 训练过程中开启 Dropout
            iteration = iteration + 1
            sensor, labels = data  # 获取训练集的图像和标签
            labels = labels.view(labels.size(0))
            sensor = sensor.to(device, dtype=torch.float)  # 转换到设备
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()  # 清除历史梯度
            outputs = net(sensor)  # 正向传播

            loss = loss_function(outputs, labels)  # 计算损失


            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数

            # 更新进度条描述
            train_loader_tqdm.set_postfix({
                'Train Loss': f'{loss.item():.4f}',
            })

            predict_y = torch.max(outputs, dim=1)[1]

            acc += (predict_y == labels.to(device)).sum().item()
            epoch_loss += loss.item()  # 计算损失

        train_num = train_dataset.__len__()  # 测试集总数
        write.add_scalar('train_loss', epoch_loss/train_num, epoch)  # 可视化损失
        write.add_scalar('train_acc', acc/train_num, epoch)  # 可视化准确率

        ########################################### validate ###########################################
       ########################################### validate ###########################################
        # 验证过程中关闭 Dropout
        net.eval()
        epoch_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            for val_data in validation_data:
                val_sensor, val_labels = val_data
                val_labels = val_labels.view(val_labels.size(0))
                val_sensor = val_sensor.to(device, dtype=torch.float)  # 转换统一数据格式
                val_labels = val_labels.to(device, dtype=torch.long)

                outputs = net(val_sensor)
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                acc += (predict_y == val_labels.to(device)).sum().item()
                epoch_loss += loss_function(outputs, val_labels).item()  # 计算损失


            val_num = validation_dataset.__len__()  # 测试集总数

            val_accurate = acc / val_num
            Accuracy.append(val_accurate)

            # 保存准确率最高的那次网络参数
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

            print('[epoch %d] test_accuracy: %.3f \n' %
                  (epoch + 1, val_accurate))


        write.add_scalar('val_loss', epoch_loss / val_num, epoch)  # 可视化损失
        write.add_scalar('val_acc', val_accurate, epoch)  # 可视化准确率

    print('Finished Training')

    write.close()