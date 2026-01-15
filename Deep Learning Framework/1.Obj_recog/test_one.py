# -*- coding: utf-8 -*-
"""
Test TSNE network
==============
**Author**: `zhibin Li`__
#python test_one.py

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
    workers = 1

    # Batch size during training
    batch_size = 20  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

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



    """
    ====================================
    Test
    """
    net.eval()
    epoch_loss = 0.0
    acc = 0.0

    feture_all = []
    lable_all = []






    val_sensor, val_labels = dataset.getoneframe(3,14)  # lable,num

    val_sensor = val_sensor.unsqueeze(0)
    # val_labels = val_labels.unsqueeze(0)

    print(val_sensor.size())
    print(val_labels.size())


    val_sensor = val_sensor.to(device, dtype=torch.float)  # 转换统一数据格式
    val_labels = val_labels.to(device, dtype=torch.long)


    outputs = net(val_sensor,return_feature=False)
    print("softmax:", outputs)
    predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出

    lable =  val_labels.cpu().numpy()
    if (lable==0):
        lable_str = 'Tennis'
    if (lable==1):
        lable_str = 'Mouse'
    if (lable==2):
        lable_str = 'Milk'
    if (lable==3):
        lable_str = 'Banana'
    if (lable==4):
        lable_str = 'Battery'
    if (lable==5):
        lable_str = 'Tape'
    if (lable==6):
        lable_str = 'Medicine box'
    if (lable==7):
        lable_str = 'Screwdriver'
    if (lable==8):
        lable_str = 'Magic cube'
    if (lable==9):
        lable_str = 'Camera'
    print("real lable:", lable_str)



    lable = predict_y.cpu().numpy()
    if (lable == 0):
        lable_str = 'Tennis'
    if (lable == 1):
        lable_str = 'Mouse'
    if (lable == 2):
        lable_str = 'Milk'
    if (lable == 3):
        lable_str = 'Banana'
    if (lable == 4):
        lable_str = 'Battery'
    if (lable == 5):
        lable_str = 'Tape'
    if (lable == 6):
        lable_str = 'Medicine box'
    if (lable == 7):
        lable_str = 'Screwdriver'
    if (lable == 8):
        lable_str = 'Magic cube'
    if (lable == 9):
        lable_str = 'Camera'
    print("Predict lable:", lable_str)



