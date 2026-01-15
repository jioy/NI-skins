'''
IWT_coding
==============
**Author**: `zhibin Li`__
'''
#python ./Method/Model.py

import pywt
import zlib
import time
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI环境专用的后端

from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import torch

from skimage.metrics import structural_similarity as compare_ssim



class IWT_coding():
    def __init__(self, Sparsity_Q = 1):  # target_sparsity 稀疏程度
        super(IWT_coding, self).__init__()
        self.Sparsity_Q = Sparsity_Q  # 量化因子

    def integer_wavelet_compress(self, data, wavelet='bior1.3'):
        assert np.issubdtype(data.dtype, np.integer), "输入必须为整数类型"

        # 使用 lifting 小波做多维小波分解
        coeffs = pywt.wavedecn(data, wavelet=wavelet, mode='periodization', axes=(0, 1, 2), level=2)

        # 转为统一数组形式
        arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        # 四舍五入，转为整型（这里才是真正“整数小波系数”）
        arr_int = np.round(arr/self.Sparsity_Q).astype(np.int32)

        return arr_int, coeff_slices, arr.shape


    def integer_wavelet_decompress(self, arr_q, coeff_slices, original_shape, wavelet='bior1.3'):
        #数组以便小波逆变换
        arr_rec = arr_q * self.Sparsity_Q

        # 重构为小波系数结构
        coeffs = pywt.array_to_coeffs(arr_rec.reshape(original_shape), coeff_slices, output_format='wavedecn')

        # 小波逆变换
        data_rec = pywt.waverecn(coeffs, wavelet=wavelet, mode='periodization', axes=(0, 1, 2))

        # 还原成整数
        return np.round(data_rec).astype(np.int16)



    def entropy_encode(self, data):
        flat_bytes = data.tobytes()
        return zlib.compress(flat_bytes)

    def entropy_decode(self, encoded, shape, dtype):
        flat_bytes = zlib.decompress(encoded)
        return np.frombuffer(flat_bytes, dtype=dtype).reshape(shape)


    def compress_decompress(self,original_data, Sparsity_Q = 1):

        self.Sparsity_Q = Sparsity_Q  # 量化因子
        original_data = original_data.astype(np.int16)

        # 压缩流程
        start_time = time.time()
        arr_q, coeff_slices, shape = self.integer_wavelet_compress(original_data)
        compressed = self.entropy_encode(arr_q)
        compress_time = time.time() - start_time

        # 解压还原
        start_time = time.time()
        arr_q_decoded = self.entropy_decode(compressed, shape, arr_q.dtype)
        reconstructed_data = self.integer_wavelet_decompress(arr_q_decoded, coeff_slices, shape)
        decompress_time = time.time() - start_time

        # 评估
        original_size = original_data.nbytes
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size
        max_error = np.max(np.abs(original_data - reconstructed_data))
        is_lossless = np.array_equal(original_data, reconstructed_data)
        rmse = np.sqrt(np.mean((original_data - reconstructed_data) ** 2))
        ssim_value = ssim(original_data, reconstructed_data, data_range=original_data.max())

        return compression_ratio, ssim_value


    def compress_out(self,original_data, Sparsity_Q = 1):

        self.Sparsity_Q = Sparsity_Q  # 量化因子
        original_data = original_data.astype(np.int16)

        # 压缩流程
        start_time = time.time()
        arr_q, coeff_slices, shape = self.integer_wavelet_compress(original_data)
        compressed = self.entropy_encode(arr_q)
        compress_time = time.time() - start_time

        # 解压还原
        start_time = time.time()
        arr_q_decoded = self.entropy_decode(compressed, shape, arr_q.dtype)
        reconstructed_data = self.integer_wavelet_decompress(arr_q_decoded, coeff_slices, shape)
        decompress_time = time.time() - start_time



        return arr_q # reconstructed_data #




    def save_png(self,save_data,output_path = "./result/original_data.png", upscale_size=(640, 640), sigma=3):
        """
            将 96x96 图像升采样为 640x640，并进行高斯平滑后保存为热力图 PNG。

            Parameters:
            - save_data: 2D numpy array (原始图像)
            - output_path: 文件保存路径
            - upscale_size: 目标图像大小 (默认 640x640)
            - sigma: 高斯平滑参数，越大越平滑
            """
        # Step 1: 升维至 640x640
        # 设置裁剪的最大值和最小值
        min_val = np.percentile(save_data, 5)  # 取5%分位数作为下限
        max_val = np.percentile(save_data, 95)  # 取95%分位数作为上限

        # 将数据裁剪到合理范围
        save_data = np.clip(save_data, min_val, max_val)
        upscaled = resize(save_data, upscale_size, order=3, mode='reflect', anti_aliasing=True)

        # Step 2: 高斯平滑
        smoothed = gaussian_filter(upscaled, sigma=sigma)

        # Step 3: 绘图
        fig, ax = plt.subplots(figsize=(12, 12))  # 输出尺寸
        ax.imshow(smoothed, cmap='cividis', interpolation='nearest')
        plt.axis('off')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()




if __name__ == '__main__':
    # 设置系数比例
    target_sparsity = 0
    Comp_Method = IWT_coding(target_sparsity = target_sparsity)

