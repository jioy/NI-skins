# import os
# import sys
# sys.path.insert(0, os.getcwd())

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import pyqtgraph.opengl as gl
import numpy as np
import random #随机数生成
import multiprocessing
from multiprocessing import Queue
import lib.usb_get as usb_get
from lib.ui import Ui_Form
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
import sys
import time
import csv
import os
import matplotlib.pyplot as plt



class PLOT_3D(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.setWindowIcon(QIcon('lib/image.png'))


        # 设置串口实例
        self.com = QSerialPort()
        self.port_check()

        self.init() #界面按钮初始化

        ## 创建3D图形窗口
        self.w = gl.GLViewWidget()
        self.w.show()
        self.w.setWindowTitle('3D Map Visualization')
        self.w.setCameraPosition(distance=50, elevation=30, azimuth=45)
        # 设置背景颜色为白色
        self.w.setBackgroundColor('w')  # 使用颜色名称

        ## 创建3D网格
        # 生成网格数据
        map_length = 48
        self.z = np.random.normal(size=(map_length,map_length))
        self.x = np.linspace(-12, 12, map_length)
        self.y = np.linspace(-12, 12, map_length)

        # 获取 Matplotlib 的 'rainbow' colormap
        self.cmap = plt.get_cmap('Purples')   #rainbow Purples   cividis

        # 将 Z 值归一化到 [0, 1] 范围以对应 colormap
        self.rgba_img = self.cmap((self.z - self.z.min()) / (self.z.max() - self.z.min()))


        self.mesh = gl.GLSurfacePlotItem(x=self.x, y=self.y, z=self.z, colors=self.rgba_img, shader='shaded')
        self.mesh.translate(-10, -10, 0)
        self.w.addItem(self.mesh)

        self.verticalLayout_graph.addWidget(self.w)





    """定义信号与槽"""

    def init(self):
        # 打开按钮
        self.open_button.clicked.connect(self.port_open)
        # 关闭按钮
        self.close_button.clicked.connect(self.port_close)

        # 清空
        self.start_measure_button.clicked.connect(self.start_measure)

        # 保存数据
        self.savedata_button.clicked.connect(self.save_data)

        # 退出应用
        self.quit_Button.clicked.connect(self.quit)


    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.serialportnum.clear()
        port_list = QSerialPortInfo.availablePorts()
        # print(self.port_list)
        if len(port_list) == 0:
            pass
            #self.state_label.setText("None")
        else:
            for port in port_list:
                self.com.setPort(port)
                if self.com.open(QSerialPort.ReadWrite):
                    self.serialportnum.addItem('    ' + port.portName())
                    self.com.close()



    # 打开串口
    def port_open(self):
        if self.serialportnum.currentText().strip() == 'None':
            QMessageBox.critical(self, "Port Error", "请选择串口！")
            return None
        print(self.serialportnum.currentText().strip())
        print("开始")

        #初始化数据读取，接收
        self.usbdata = usb_get.USB_DataDecode(self.serialportnum.currentText().strip())
        self.sensor = []

        #画图多线程中断
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(10)


        self.timer_save = QtCore.QTimer()
        self.timer_save.timeout.connect(self.save_timer)


    def start_measure(self):
        print("清空")
        self.sensor = []


    def port_close(self):
        self.usbdata.GUI_order.put(['stop',0])  # 发送结束
        self.timer_save.stop()  # 停止
        self.sensor = []
        self.textBrowser_3.append("已停止")  # 在指定的区域显示提示信息
        print("已停止")

    def save_data(self):
        button_state = self.savedata_button.text()
        int_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

        if (button_state == 'Start Save'):
            os.makedirs('./lib/data_save/' + int_time)
            self.savepath = './lib/data_save/' + int_time
            self.textBrowser.append(self.savepath)  # 在指定的区域显示提示信息

            self.savedata_button.setText("Complete save")
            self.clear_Queue(self.usbdata.data_out)  # 清空队列
            self.usbdata.GUI_order.put(['start', self.savepath])  # 发送开始
            self.textBrowser_3.append("开始检测")  # 在指定的区域显示提示信息da's'fa's'd'f'sa's'd'fa's'd
            self.timer_save.start(10)  # 定时保存时间 ms

        else:
            self.usbdata.GUI_order.put(['stop', 0])  # 发送结束
            self.timer_save.stop()  # 停止
            self.textBrowser_2.append("保存成功")  # 在指定的区域显示提示信息
            self.textBrowser_3.append("保存结束")  # 在指定的区域显示提示信息
            self.savedata_button.setText("Start Save")



    def quit(self):
        try:
            self.usbdata.close_usb()
        except:
            pass
        app.quit()
        print("成功退出")




    def save_data_init(self, filenname):
        data_head = ['timestamp'] + ['sensor' + str(i) for i in range(1, 257)]
        headers =   data_head
        with open(filenname, 'w', newline='') as form:
            writer = csv.writer(form)
            writer.writerow(headers)

    def write_data(self, data):
        filenname = self.savepath
        with open(filenname, 'a', newline='') as form:
            writer = csv.writer(form)
            writer.writerows(data)
        self.sensor = []

    def save_timer(self):
        # 计时显示
        buf_len = len(self.sensor)
        int_time = time.strftime("%Y-%m-%d- %H:%M:%S", time.localtime())

        self.textBrowser_2.append(int_time)  # 在指定的区域显示提示信息

        res = []
        while self.usbdata.data_out.qsize() > 0:
            res.append(self.usbdata.data_out.get())


        #res.reverse()
        self.sensor = self.sensor + res
        if (buf_len >= 20):
            print('Saving')
            self.write_data(self.sensor)


    def clear_Queue(self,q):
        res = []
        while q.qsize() > 0:
            res.append(q.get())








    #绘图
    def update_plot(self):

        self.image,self.ave_value = self.usbdata.image_construct.get(True)
        self.clear_Queue(self.usbdata.image_construct)  # 清空队列


        #self.plotdata = np.zeros((600, 600))

        ## Display the data
        #self.image = np.rot90(self.image, k=-1)  # 顺时针旋转 90°

        #self.z = np.random.normal(size=(48, 48))

        self.image = np.fliplr(self.image)
        rgb_img =self.cmap(self.image)


        self.mesh.setData(z = self.image, colors = rgb_img )


        #self.img.setImage(self.image)

        #3D绘图可视化
        #z = pg.gaussianFilter(z,(4,4))   #高斯平滑

        #显示第一个数值



def closehand():
    print('close')


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = PLOT_3D()
    window.show()
    sys.exit(app.exec_())



