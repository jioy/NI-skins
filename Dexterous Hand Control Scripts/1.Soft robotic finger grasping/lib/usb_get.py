import serial
import time
import struct
import multiprocessing
from multiprocessing import Queue
import os
import numpy as np
import time
import keyboard
import sys
from PIL import Image
import pyqtgraph as pg
import cv2
import h5py

import argparse
import time
from lib.magiclaw_controller import MagiclawController



class Hand_control:     #USB 400Hz 刷新率

    controller = 0
    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser(description="Test the MagiClaw system.")
        parser.add_argument(
            "--controller_host",
            type=str,
            default="0.0.0.0",
            help="Host address for the controller (default: localhost).",
        )
        parser.add_argument(
            "--magiclaw_host",
            type=str,
            default="192.168.4.1",
            help="Host address for the MagiClaw (default: localhost).",
        )

        args = parser.parse_args()

        # Create a MagiClaw controller
        Hand_control.controller = MagiclawController(
            controller_host=args.controller_host,
            magiclaw_host=args.magiclaw_host,
            loop_rate=100,
        )

        Hand_control.controller.send_commands(
            claw_angle=50,
            motor_speed=50,
        )



class USB_Connect:     #USB 400Hz 刷新率

    def __init__(self):
        super().__init__()


        # 读取参数
        f = open('./lib/coefficient.txt')
        coefficient_data = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
        f.close()  # 关

        # ['Load ratio parameter:\n', '3']
        # 参数设置
        self.coefficient = float(coefficient_data[1])
        print('比例参数为：', self.coefficient)





    def Message_decode(self,data_flag,com):

        try:
            self.com = serial.Serial(com, 2000000)
            print('串口连接成功')

        except Exception as e:
            print("---异常---：", e)
            print("---硬件串口异常---：")
            sys.exit(0)



        while True:
            #包头截取
            dd = self.com.read(1)
            if (dd == b'\xbb'):
                if(self.com.read(1) != b'\xbb'):
                    continue
            else: continue

            while True:
                data = self.com.read((4608+2)*2)

                if(data[0:2] != b'\xaa\xaa' or data[-2:] != b'\xbb\xbb'): #包头，包尾核对
                    print('erro')
                    break
                #print('ok')

                data_flag.put(data)



    def sendMessage(self, message):
        self.tcp_client.write(message.encode(encoding='utf-8'))


    def closeClient(self):
        self.com.close()
        print("串口已关闭")


    #清空队列
    def clear_Queue(self,q):
        res = []
        while q.qsize() > 0:
            res.append(q.get())




    # 膝关节重建 图像获取
    def get_image(self):
        image_dir = r"./lib/AI/joint2.png"
        self.joint = Image.open(image_dir)  # 打开图片
        self.joint = self.joint.convert("RGB") #变RGB,三通道
        self.joint = self.joint.resize((600, 600))
        self.joint = self.joint.rotate(270)  # 逆时针旋转90

        self.joint_h = self.joint.convert('L')
        self.joint = np.asarray(self.joint)  # 转换为矩阵
        self.joint_h = np.asarray(self.joint_h)  # 转换为矩阵

        self.coord_A = [[53,147],[120, 147],[190, 147],
                        [27,206],[88,206],[153,206],[217,206],
                        [28,264],[88,264],[153,264],[217,264],
                        [25,320],[87,320],[153,320],[220,320],
                        [32, 376], [93, 376], [158, 376], [224, 376],
                        [42, 430], [104, 430], [169, 430], [234, 430],
                        [68, 486], [137, 486], [208, 486],
                        ]

        self.coord_B = [[403, 147], [477, 147], [550, 147],
                        [374, 206], [442, 206], [512, 206], [575, 206],
                        [375, 264], [444, 264], [510, 264], [575, 264],
                        [368, 320], [435, 320], [507, 320], [570, 320],
                        [362, 376], [428, 376], [497, 376], [563, 376],
                        [352, 430], [419, 430], [487, 430], [552, 430],
                        [362, 486], [435, 486], [508, 486],
                        ]

        self.white = np.zeros((600, 600,3))
        self.black = np.zeros((600, 600, 3))
        for w in range(600):
            for h in range(600):
                if (self.joint_h[w,h]==255):
                    self.white[w,h,:] = 0
                    self.black[w,h,:] = 255
                else:
                    self.white[w, h, :] = 1
                    self.black[w, h, :] = 0



    def image_reconstructed(self,data_z,image_construct):
        self.get_image()
        self.value_based = 200;

        Sensor_z = np.zeros((48, 48))

        ######################################################
        #机械手初始化
        #######################################################
        parser = argparse.ArgumentParser(description="Test the MagiClaw system.")
        parser.add_argument(
            "--controller_host",
            type=str,
            default="0.0.0.0",
            help="Host address for the controller (default: localhost).",
        )
        parser.add_argument(
            "--magiclaw_host",
            type=str,
            default="192.168.4.1",
            help="Host address for the MagiClaw (default: localhost).",
        )

        args = parser.parse_args()

        # Create a MagiClaw controller
        Hand_control.controller = MagiclawController(
            controller_host=args.controller_host,
            magiclaw_host=args.magiclaw_host,
            loop_rate=100,
        )

        Hand_control.controller.send_commands(
            claw_angle=40,
            motor_speed=200,
        )

        a_nag = 0.5
        theta = 1

        voity = 0
        claw_angle = 50

        key_mode = 3
        pressure_goal = 400   #



        while True:

            get_z = data_z.get(True)[0] #接收数据
            self.clear_Queue(data_z)

            Sensor_z = get_z[:,48:]  #A0:48 B48：
            #Sensor_z = np.rot90(Sensor_z, k=-1)  # k=-1 表示顺时针90度

            Sensor_2 = Sensor_z[:24, :24]
            Sensor_1 = Sensor_z[24:, :24]

            Sensor_1 = np.rot90(Sensor_1, k=1)  # 逆时针旋转 90°
            Sensor_2 = np.rot90(Sensor_2, k=1)  # 逆时针旋转 90°
            #左右对称
            #Sensor_1 = np.flipud(Sensor_1)
            #Sensor_2 = np.flipud(Sensor_2)

            skin1_ave_value = np.mean(Sensor_1)
            skin1_max_value = np.max(Sensor_1)
            skin2_ave_value = np.mean(Sensor_2)
            skin2_max_value = np.max(Sensor_2)



            ####SKIN1
            sensor = cv2.resize(Sensor_1 * self.coefficient, (640, 640), interpolation=cv2.INTER_NEAREST)  #INTER_LINEAR
            sensor *= 255  # 变换为0-255的灰度值
            # 限制数值范围在0-255之间，并转换为uint8类型
            sensor = cv2.convertScaleAbs(sensor)
            # 热力图  避免颜色变化
            sensor[0, 0] = 255
            heat_img = cv2.applyColorMap(sensor, cv2.COLORMAP_CIVIDIS)  # 注意此处的三通道热力图是cv2专有的GBR排列
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
            # 计算每个格子的大小
            grid_size_x = 640 / 24
            grid_size_y = 640 / 24
            # 绘制网格线
            for i in range(1):  # 48个格子需要49条线
                # 计算垂直线的位置
                pos_x = int(round(i * grid_size_x))
                # 计算水平线的位置
                pos_y = int(round(i * grid_size_y))
                # 绘制垂直线
                cv2.line(heat_img, (pos_x, 0), (pos_x, 640), color=(255, 0, 0), thickness=1)  # 红色线条
                # 绘制水平线
                cv2.line(heat_img, (0, pos_y), (640, pos_y), color=(255, 0, 0), thickness=1)  # 红色线条
            Sensor1_map = heat_img


            ####SKIN2
            sensor = cv2.resize(Sensor_2 * self.coefficient, (640, 640),
                                interpolation=cv2.INTER_NEAREST)  # INTER_LINEAR
            sensor *= 255  # 变换为0-255的灰度值
            # 限制数值范围在0-255之间，并转换为uint8类型
            sensor = cv2.convertScaleAbs(sensor)
            # 热力图  避免颜色变化
            sensor[0, 0] = 255
            heat_img = cv2.applyColorMap(sensor, cv2.COLORMAP_CIVIDIS)  # 注意此处的三通道热力图是cv2专有的GBR排列
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
            # 计算每个格子的大小
            grid_size_x = 640 / 24
            grid_size_y = 640 / 24
            # 绘制网格线
            for i in range(1):  # 48个格子需要49条线
                # 计算垂直线的位置
                pos_x = int(round(i * grid_size_x))
                # 计算水平线的位置
                pos_y = int(round(i * grid_size_y))
                # 绘制垂直线
                cv2.line(heat_img, (pos_x, 0), (pos_x, 640), color=(255, 0, 0), thickness=1)  # 红色线条
                # 绘制水平线
                cv2.line(heat_img, (0, pos_y), (640, pos_y), color=(255, 0, 0), thickness=1)  # 红色线条
            Sensor2_map = heat_img


            ################################################

            if keyboard.is_pressed('1'):
                print('1抓取')
                key_mode = 1

            if keyboard.is_pressed('2'):
                print('2抓取')
                key_mode = 2

            if keyboard.is_pressed('3'):
                print('3打开')
                key_mode = 3
                Hand_control.controller.send_commands(
                    claw_angle=40,
                    motor_speed=200,  # 50
                )
                claw_angle = 40


            if (key_mode == 1):
                deta_p  = skin1_max_value + skin2_max_value - pressure_goal
                if(deta_p<=0):

                    claw_angle = claw_angle - 0.3

                    if(claw_angle<=5):
                        claw_angle = 5

                    if (claw_angle >= 40):
                        claw_angle = 40
                #################################################
                #更新控制
                Hand_control.controller.send_commands(
                    claw_angle=claw_angle,
                    motor_speed=50, # 50
                )










            image_construct.put([Sensor1_map, Sensor2_map,
                                 skin1_ave_value, skin1_max_value, skin2_ave_value, skin2_max_value])





    def init_data(self,data_flag):
        init_data_buffer = [0 for i in range(0,4608)]
        data_buffer = {n: [] for n in range(4608)}

        for i in range(30):

            udp_data = data_flag.get(True)  # 接收数据

            for i in range(0, 4608):
                data_analysis = struct.unpack('<h', udp_data[2 + i * 2: 4 + i * 2])  # 元组  2字节
                data_decode = data_analysis[0]

                data_buffer[i].append(data_decode)


        for i in range(4608):
            init_data_buffer[i] = np.max(data_buffer[i])

        return init_data_buffer



    def usb_decode(self,data_flag,data_out,data_z,GUI_order):
        data_buffer = [0] * 4608
        send_savedata = np.zeros((4608+1))

        Sensor_z = np.zeros((48, 96))




        self.send_flag = 0  #装载数据标志位
        strat_time = time.time()  #采样计时

        #初始化，求均值
        init_data = self.init_data(data_flag)



        while True:
            udp_data = data_flag.get(True) #接收数据

            for i in range(0, 4608):
                data_analysis = struct.unpack('<h', udp_data[2 + i * 2: 4 + i * 2])  # 元组  2字节
                data_decode = data_analysis[0] - init_data[i]
                send_savedata[i+1] = data_decode

                if (data_decode < 0):  data_decode = 0
                data_buffer[i] = data_decode



            Sensor_z = np.reshape(data_buffer, (48, 96))




            #Sensor_z[20,:] = 1000;
            #Sensor_z[15, :] = Sensor_z[23, :];
            Sensor_z[23, :] = Sensor_z[22, :] * 0.4;



            data_z.put([Sensor_z,0])



            # 是否保存数据 数据保存
            if (GUI_order.empty() == False):  # 接收到采样标志后开始采样
                get_flag, save_path = GUI_order.get()
                if (get_flag == 'start'):  # 开始采样装入
                    self.clear_Queue(data_out)  # 清空缓存
                    self.savefile_int(save_path)  # 初始化保存路径

                    strat_time = time.time()
                    self.send_flag = 1
                if (get_flag == 'stop'):  # 开始采样装入
                    deta_time = time.time() - strat_time
                    print(deta_time)
                    self.send_flag = 0

            if (self.send_flag == 1):
                timestamp = time.time()  # 时间戳

                # 1、存储压力传感器
                with h5py.File(self.file_names[0], 'a') as hf:
                    time_dataset = hf['Time']
                    sensor1_dataset = hf['Sensor1']
                    sensor2_dataset = hf['Sensor2']

                    # 获取当前数据集长度
                    current_length = time_dataset.shape[0]

                    # 扩展数据集大小以容纳新数据
                    new_length = current_length + 1
                    time_dataset.resize((new_length,))  # (n,)
                    sensor1_dataset.resize((new_length, 24, 24))  # 调整为合适的三维形状 (n, 64)
                    sensor2_dataset.resize((new_length, 24, 24))  # 调整为合适的三维形状 (n, 64)

                    # 添加新数据到数据集末尾
                    time_dataset[current_length] = timestamp
                    get_z = Sensor_z[:, 48:]  # A0:48 B48：
                    sensor1_dataset[current_length, :] = get_z[24:, :24]
                    sensor2_dataset[current_length, :] = get_z[:24, :24]


    def savefile_int(self,save_path):
        # 先创建3个HDF5文件
        self.file_names = [save_path+'/Pressure_sensor.h5', save_path+'/UWB.h5', save_path+'/Depth image.h5']
        self.saveCV_path = save_path+'/cv.mp4'
        #self.Video_saveinit()


        # 1、Pressure sensor.h5
        file_name = self.file_names[0]
        with h5py.File(file_name, 'w') as hf:
            # 创建一个数据集，初始为空
            hf.create_dataset('Time', shape=(0,), maxshape=(None,), dtype='f')
            hf.create_dataset('Sensor1', shape=(0,24,24), maxshape=(None,24,24), dtype='int16')
            hf.create_dataset('Sensor2', shape=(0, 24, 24), maxshape=(None, 24, 24), dtype='int16')






class USB_DataDecode:

    def __init__(self,com):
        super().__init__()

        #self.draw_3d = plot3d.PLOT_3D()
        self.com_num = com

        #多进程
        self.data_flag = Queue() #更新状态
        self.data_out = Queue()  #数据解析结果

        self.data_z = Queue()  # 数据解析结果   //接收解析数据

        self.image_construct = Queue()  #图像重建结果

        self.GUI_order = Queue()  # GUI控制数据解析


        #进程1 接收数据
        self.T = USB_Connect()
        self.thread_getMessage = multiprocessing.Process(target=self.T.Message_decode,args=(self.data_flag,self.com_num))

        #进程2 处理数据
        self.thread_usbdecode = multiprocessing.Process(target=self.T.usb_decode,args=(self.data_flag,self.data_out,self.data_z,self.GUI_order))

        #进程3 图像重建
        self.thread_image_construct = multiprocessing.Process(target=self.T.image_reconstructed, args=(
        self.data_z, self.image_construct))

        #self.thread_key_monitoring = multiprocessing.Process(target=self.key_monitoring, args=())


        #self.eat_process = multiprocessing.Process(target=self.eat, args=(3, "giao"))
        print("主进程ID:", os.getpid())
        self.thread_getMessage.start()
        self.thread_usbdecode.start()
        self.thread_image_construct.start()





    def close_usb(self):
        self.thread_getMessage.terminate()
        self.thread_getMessage.join()

        self.thread_usbdecode.terminate()
        self.thread_usbdecode.join()

        self.thread_image_construct.terminate()
        self.thread_image_construct.join()


    def save(self):
        print('zhibin')


    def key_monitoring(self):
        keyboard.add_hotkey('c', self.save())  # 初始化验证
        keyboard.wait()









if __name__ == '__main__':
    usb = USB_DataDecode()

