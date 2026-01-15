# -*- coding: utf-8 -*-
"""
Training network
==============
**Author**: `zhibin Li`__
#python test_violinplot.py

"""
import numpy as np
import pandas as pd
import os
import random

import matplotlib
matplotlib.use("Agg")   # 使用无界面后端
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import r2_score
import seaborn as sns



import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取 Excel 文件
readdf = pd.read_excel("prediction_results.xlsx")

df = pd.DataFrame({
    "True Values": readdf["Y_true"],
    "Predicted Values": readdf["Y_pred"]
})

# ===== 检查分组是否退化（方差=0 或只有1个样本） =====
group_var = df.groupby("True Values")["Predicted Values"].nunique()
degenerate_groups = group_var[group_var <= 1].index.tolist()

print("退化分组（无法绘制violinplot，将改用boxplot/stripplot）：", degenerate_groups)

# ===== 设置画布和风格 =====
plt.figure(figsize=(6, 5))
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams['xtick.major.width'] = 0.3
mpl.rcParams['ytick.major.width'] = 0.3

# Nature 风格颜色（五组刚好用前5个）
nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

order = [0, 0.25, 0.5, 0.75, 1]

# ===== 如果没有退化分组 → 用 violinplot =====
if len(degenerate_groups) == 0:
    sns.violinplot(
        x="True Values", y="Predicted Values", data=df,
        order=order, palette=nature_colors,
        inner=None, linewidth=0.3, bw=0.2  # bw 调节带宽避免报错
    )

    # 叠加箱线图
    sns.boxplot(
        x="True Values", y="Predicted Values", data=df,
        order=order, width=0.15, showcaps=True,
        boxprops={'facecolor':'white', 'linewidth':0.7},
        whiskerprops={'linewidth':0.7},
        medianprops={'color':'black', 'linewidth':1}
    )

# ===== 如果有退化分组 → 改用 boxplot + stripplot =====
else:
    sns.boxplot(
        x="True Values", y="Predicted Values", data=df,
        order=order, width=0.4, showcaps=True,
        boxprops={'facecolor':'white', 'linewidth':0.7},
        whiskerprops={'linewidth':0.7},
        medianprops={'color':'black', 'linewidth':1}
    )
    sns.stripplot(
        x="True Values", y="Predicted Values", data=df,
        order=order, color="0.3", alpha=0.6
    )

# ===== 坐标轴标签 =====
plt.xlabel("True Fx (N)")
plt.ylabel("Predicted Fx (N)")

# ===== 保存为 SVG 高分辨率 =====
plt.savefig("Tz_V1.svg", format="svg", dpi=1200, bbox_inches="tight")
plt.show()










# # 读取 xlsx 文件
# readdf = pd.read_excel('prediction_results.xlsx')
# # 显示 DataFrame 的前几行
# print(readdf.head())
#
# df = pd.DataFrame({'True Values': readdf['Y_true'], 'Predicted Values': readdf['Y_pred']})
#
#
#
# df_sorted = df.sort_values(by='True Values', ascending=True)
# print(df_sorted.head())
# print(df_sorted['True Values'].max())
#
#
# # 截取出第一列在指定范围内的数据段
# df_list = []
# df_0 = df[(df['True Values'] <= -3.5)]
# df_list.append(df_0)
# for i in range(7):
#     df_list.append(df[(df['True Values'] > -3.5 + 1*i) & (df['True Values'] <= -2.5+ 1*i)])
#
# df_list.append(df[(df['True Values'] > 3.5)])
#
# # 设置画布大小
# plt.figure(figsize=(6, 5))
#
#
# # 设置所有文本的字体为Arial
# mpl.rcParams['font.family'] = 'Arial'
# # 设置字体大小
# mpl.rcParams['axes.titlesize'] = 16  # 标题字体大小
# mpl.rcParams['axes.labelsize'] = 10  # 轴标签字体大小
# mpl.rcParams['xtick.labelsize'] = 18  # x轴刻度标签字体大小
# mpl.rcParams['ytick.labelsize'] = 18  # y轴刻度标签字体大小
# # 设置边框线宽
# # 设置边框和刻度线的线宽
# mpl.rcParams['axes.linewidth'] = 0.3  # 边框线宽
# mpl.rcParams['xtick.major.width'] = 0.3  # x轴刻度线宽
# mpl.rcParams['ytick.major.width'] = 0.3  # y轴刻度线宽
# # 定义符合Nature期刊风格的颜色集
# nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
#                  '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#D37295']
# # 使用不同颜色绘制小提琴图
# for i in range(len(df_list)):
#     sns.violinplot(x=-4+i, y='Predicted Values', data=df_list[i], color=nature_colors[i], inner=None, linewidth=0.3)
#
# # plt.ylim(-6, 59)
#
# # 设置标题和标签
# plt.xlabel('True Fx (N)')
# plt.ylabel('Predicted Fx (N)')
# # 显示图像
#
# # 导出为SVG格式
# plt.savefig('Tz_V1.svg', format='svg', dpi=1200)


