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
import dataset.Compress as Compress


import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

class Object_Dataset(Dataset):
    """ sensors dataset.

    path：./Object1-10
            ./T1 # Different experiments
            ./T2 # Different experiments
            ...

    Parameters
    ----------
    === HDF5 File Structure ===


    """

    def __init__(self, root_dir   = './Alldata/', window_length = 1, Compress_Q = 1):

        self.Compress_Model = Compress.IWT_coding()
        self.Compress_Q = Compress_Q

        # time serial length
        self.window_length = window_length

        #get all obj path
        objects_path = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.obj_path = sorted(objects_path, key=lambda x: int(x.split('.')[0]))

        print('Objects:',self.obj_path)
        #['1.Tennis', '2.Mouse', '3.Milk', '4.Banana',]

        # save sensor data and joints data path
        self.frame_paths = []  # save data path

        #Traverse all experiments
        for obj in self.obj_path:
            #['1.Tennis', '2.Mouse', '3.Milk', '4.Banana']
            obj_path = os.path.join(root_dir, obj)

            # /root/data/usr/Zhibin/15.96x96_skins/Alldata/Obj_data/1.Tennis
            self.experiments_path = sorted([d for d in os.listdir(obj_path) if
                                     os.path.isdir(os.path.join(obj_path, d))])
            #['T1', 'T2', 'T3', 'T4']

            for exp in self. experiments_path:
                senor_path = os.path.join(obj_path, exp, "Pressure_sensor.h5")


                if os.path.exists(senor_path):
                    with h5py.File(senor_path, 'r') as f:
                        # 假设你知道某个数据集的名称，比如 'data'，可以这样查看长度：
                        Datasets = f['Data frame']
                        pressure_Length = len(Datasets)




                    sync_frames = pressure_Length

                    for frame_idx in range(self.window_length,sync_frames):
                        self.frame_paths.append({
                            "frame_index": frame_idx,
                            "senor_path": senor_path,
                            "senor_lable": int(obj.split('.')[0]) - 1,

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
        senor_lable = frame_info["senor_lable"]

        #1、 打开 sensor HDF5 文件
        with h5py.File(senor_path, 'r') as f:
            # 获取数据集
            dataset = f['Data frame']
            # 读取第 frame_index 帧
            sensor_frame = dataset[frame_index - self.window_length: frame_index]
            #<class 'numpy.ndarray'> (time, 48,48)



        sensor_frame = np.array(sensor_frame)
        senor_lable = np.array(senor_lable)

        sensor_frame = torch.from_numpy(sensor_frame.astype(np.float32))
        senor_lable = torch.from_numpy(senor_lable)


        #compress_data = self.Compress_Model.compress_out(sensor_frame, Sparsity_Q = self.Compress_Q)


        return sensor_frame, senor_lable



    def getoneframe (self, lable, num):  # index get

        count_num = 0
        for index in range(self.len):
            frame_info = self.frame_paths[index]
            frame_index = frame_info["frame_index"]
            senor_path = frame_info["senor_path"]
            senor_lable = frame_info["senor_lable"]

            if senor_lable == lable:
                if (count_num == num):
                    break
                else: count_num = count_num + 1



        #1、 打开 sensor HDF5 文件
        with h5py.File(senor_path, 'r') as f:
            # 获取数据集
            dataset = f['Data frame']
            # 读取第 frame_index 帧
            sensor_frame = dataset[frame_index - self.window_length: frame_index]
            #<class 'numpy.ndarray'> (time, 48,48)



        sensor_frame = np.array(sensor_frame)
        senor_lable = np.array([senor_lable])

        print(sensor_frame.shape)

        data_matrix = sensor_frame[0]  # 选择第一个矩阵

        # 创建一个热力图
        plt.figure(figsize=(6, 6))  # 设置图形大小
        sns.heatmap(data_matrix, cmap='viridis', cbar=True)

        # 添加标题
        plt.title("Heatmap of 48x48 Matrix")

        # 保存为 PNG 文件
        plt.savefig("heatmap.png")

        sensor_frame =torch.from_numpy(sensor_frame.astype(np.float32))
        senor_lable = torch.from_numpy(senor_lable)

        return sensor_frame, senor_lable




if __name__ == '__main__':
    Data = Object_Dataset(root_dir = '/root/data/usr/Zhibin/15.96x96_skins/Alldata/Obj_data')
    dataset_length = Data.__len__()

    print(Data.__len__())

    sensordata, lable = Data.getoneframe(6,60)

    print(sensordata.shape)



