# -*- coding: utf-8 -*-
"""
Load data
==============
**Author**: `zhibin Li`
"""
# Create the dataset  Load sensor data
# Sensor serial data reading
# python dataset/dataload.py
####################################################################
import os
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


class Gesture_Dataset(Dataset):
    """ sensors dataset.

    path：./pain0-1
            ./T1 # Different experiments
            ./T2 # Different experiments
            ...

    Parameters
    ----------
    === HDF5 File Structure ===


    """

    def __init__(self, root_dir   = './Alldata/'):



        #get all obj path [0-1]
        self.pain_path = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])


        # save sensor data and joints data path
        self.frame_paths = []  # save data path

        #Traverse all experiments
        for pain_deg in self.pain_path:
            #['1.Tennis', '2.Mouse', '3.Milk', '4.Banana']
            pain_path = os.path.join(root_dir, pain_deg)


            # /root/data/usr/Zhibin/15.96x96_skins/Alldata/Obj_data/1.Tennis
            self.experiments_path = sorted([d for d in os.listdir(pain_path) if
                                     os.path.isdir(os.path.join(pain_path, d))])
            #['T1', 'T2', 'T3', 'T4']

            for exp in self. experiments_path:
                senor_path = os.path.join(pain_path, exp, "Pressure_sensor.h5")


                if os.path.exists(senor_path):
                    with h5py.File(senor_path, 'r') as f:
                        # 假设你知道某个数据集的名称，比如 'data'，可以这样查看长度：
                        Sensor1 = f['Sensor1']     #(n,24,24)
                        Sensor2 = f['Sensor2']
                        pressure_Length = len(Sensor1)

                        #print(Sensor1.shape)
                        # print(Sensor2.shape)


                    sync_frames = pressure_Length

                    for frame_idx in range(sync_frames):
                        self.frame_paths.append({
                            "frame_index": frame_idx,
                            "senor_path": senor_path,
                            "pain_deg": float(pain_deg),

                        })
                else: print('The data path does not exist')

        self.len = len(self.frame_paths)  #sensor length
        print('Total frame length: ', self.len)





    def __len__(self):  # dataset length
        return self.len




    def __getitem__(self, index):  # index get



        frame_info = self.frame_paths[index]
        frame_index = frame_info["frame_index"]
        senor_path = frame_info["senor_path"]
        pain_deg = frame_info["pain_deg"]

        #1、 打开 sensor HDF5 文件
        with h5py.File(senor_path, 'r') as f:
            # 获取数据集
            Sensor1_dataset = f['Sensor1']  # (n,24,24)
            Sensor2_dataset = f['Sensor2']
            # 读取第 frame_index 帧
            Sensor1_frame = Sensor1_dataset[frame_index]
            Sensor2_frame = Sensor2_dataset[frame_index]
            #<class 'numpy.ndarray'> (time, 96,96)



        max_presure = max(np.max(Sensor1_frame),np.max(Sensor2_frame))
        activated_num = max(np.sum(Sensor1_frame > 400), np.sum(Sensor2_frame > 400))

        # 计算梯度 (分别是 y方向, x方向)
        gy, gx = np.gradient(Sensor1_frame)
        # 梯度幅值
        G1 = np.sqrt(gx ** 2 + gy ** 2)
        # 最大梯度
        max_G1 = np.max(G1)

        # 计算梯度 (分别是 y方向, x方向)
        gy, gx = np.gradient(Sensor2_frame)
        # 梯度幅值
        G2 = np.sqrt(gx ** 2 + gy ** 2)
        # 最大梯度
        max_G2 = np.max(G2)

        max_gradient = max(max_G1,max_G2)

        if (activated_num==0): activated_num = 400




        sensor_feature = np.array([max_presure, activated_num, max_gradient])
        pain_deg = np.array(pain_deg)



        return sensor_feature, pain_deg




if __name__ == '__main__':
    Data = Gesture_Dataset(root_dir = '/root/data/usr/Zhibin/15.96x96_skins/Alldata/Pain_data')
    dataset_length = Data.__len__()
    print(Data.__len__())
    #
    sensordata, lable = Data.__getitem__(10500)  #10500
    #
    # print(sensordata.shape)



