import socket
import threading
import struct
import multiprocessing
from multiprocessing import Queue
import os
import numpy as np
import time
import keyboard


class TcpClient:

    def __init__(self):
        super().__init__()
        self.ip = "192.168.1.10"
        self.port = 7
        self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_client.connect((self.ip, self.port))
        self.flag = 0
        self.udp_data = 0



        print('TCP连接成功')

    def getMessage(self):
        while True:
            data, addr = self.tcp_client.recvfrom(808)
            print(data)
            print('\r\n')


    def Message_decode(self,data_flag):
        while True:
            data, addr = self.tcp_client.recvfrom((256+2)*4)

            if(data[0:4] != b'\xaa\xaa\xaa\xaa' or data[-4:] != b'\xbb\xbb\xbb\xbb'): #包头，包尾核对
                #print('erro')
                continue

            #通过包头，包尾验证后：
            data_flag.put(data)
            #print('ok\r\n')

    def sendMessage(self, message):
        self.tcp_client.send(message.encode(encoding='utf-8'))


    def closeClient(self):
        self.tcp_client.close()
        print("客户端已关闭")


    #清空队列
    def clear_Queue(self,q):
        res = []
        while q.qsize() > 0:
            res.append(q.get())



    def init_data(self,data_flag):
        init_data_buffer = [0 for i in range(0,64)]
        data_buffer = {n: [] for n in range(64)}

        for i in range(4):

            udp_data = data_flag.get(True)

            for i in range(256):
                min_num = 6 + i * 4
                max_num = 8 + i * 4
                data_hend = struct.unpack('<H', udp_data[min_num:max_num])[0]
                data_decode = struct.unpack('<H', udp_data[min_num - 2:max_num - 2])[0]
                # print(data_decode)

                data_buffer[data_hend].append(data_decode)

        for i in range(64):
            init_data_buffer[i] = int(np.mean(data_buffer[i]))

        return init_data_buffer


    def udp_decode(self,data_flag,data_out,data_z,GUI_order):
        data_buffer = {n: [] for n in range(64)}

        z = np.zeros((64, 64))
        Sensor = np.zeros((8, 8))

        send_flag = 0  #装载数据标志位
        strat_time = time.time()  #采样计时

        #初始化，求均值
        init_data = self.init_data(data_flag)

        while True:
            udp_data = data_flag.get(True) #接收数据

            for i in range(256):
                min_num = 6+i*4
                max_num = 8+i*4
                data_hend = struct.unpack('<H', udp_data[min_num:max_num])[0]
                data_decode = struct.unpack('<H',udp_data[min_num-2:max_num-2])[0] - init_data[data_hend]
                #print(data_decode)


                if(len(data_buffer[data_hend]) <100 ):
                    data_buffer[data_hend].append(data_decode)
                else:
                    data_buffer[data_hend][:-1] = data_buffer[data_hend][1:]  # shift data in the array one sample left
                    data_buffer[data_hend][-1] = data_decode



            for i in range(8):
                for j in range(8):
                    z[63 - j * 8,63 - i * 8] = data_buffer[i*8 + j][-1]
                    Sensor[7 - j,7 - i] = data_buffer[i*8 + j][-1]




            #data_out.put(Sensor)
            data_z.put([z,0])


            #是否保存数据
            if (GUI_order.empty() == False):   #接收到采样标志后开始采样
                get_flag = GUI_order.get()
                if(get_flag == 'start'): #开始采样装入
                    self.clear_Queue(data_out)  #清空缓存
                    strat_time = time.time()
                    send_flag = 1
                if (get_flag == 'stop'):  # 开始采样装入
                    send_flag = 0

            deta_time = time.time() - strat_time
            if(send_flag == 1 and deta_time>0.002):
                data_out.put(Sensor)
                strat_time = time.time()














class UDP_DataDecode:

    def __init__(self):
        super().__init__()

        #self.draw_3d = plot3d.PLOT_3D()

        #多进程
        self.data_flag = Queue() #更新状态
        self.data_out = Queue()  #数据解析结果

        self.plot_z = Queue()  # 数据解析结果

        self.GUI_order = Queue()  # GUI控制数据解析


        #进程1 接收数据
        self.T = TcpClient()
        self.thread_getMessage = multiprocessing.Process(target=self.T.Message_decode,args=(self.data_flag,))

        #进程2 处理数据
        self.thread_udpdecode = multiprocessing.Process(target=self.T.udp_decode,args=(self.data_flag,self.data_out,self.plot_z,self.GUI_order))

        # 进程3 数据解析
        #self.thread_key_monitoring = multiprocessing.Process(target=self.key_monitoring, args=())


        #self.eat_process = multiprocessing.Process(target=self.eat, args=(3, "giao"))
        print("主进程ID:", os.getpid())
        self.thread_getMessage.start()
        self.thread_udpdecode.start()
        #self.thread_key_monitoring.start()
        #self.thread_plot_3d.run()




    def close_udp(self):
        self.thread_getMessage.terminate()
        self.thread_getMessage.join()

        self.thread_udpdecode.terminate()
        self.thread_udpdecode.join()


    def save(self):
        print('zhibin')


    def key_monitoring(self):
        keyboard.add_hotkey('c', self.save())  # 初始化验证
        keyboard.wait()





