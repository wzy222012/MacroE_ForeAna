import numpy as np
import pandas as pd

# 调用各个文件里想用的函数
from EVALU.EVALU_IndProce import pre_quantify_Mix
from EVALU.EVALU_WeightCal import weight_comb
from EVALU.EVALU_WeightCal import weight_add
from EVALU.EVALU_RadarChart import plot_radar

from FORECAST.DBN_revised import forecast_main

'''
----------------------df格式--------------------
            指标1  指标2  指标3  指标4 ... 指标n
    年份1
    年份2
    年份3
    :::
    年份n
              值为各个指标评分,满分为100分
------------------------------------------------
'''

# 评估主函数
def EVALU_main():
    evalu_all = pre_quantify_Mix()
    data2 = weight_add(evalu_all)
    plot_radar(evalu_all.iloc[0:8, :], data2)
    # plot_radar(evalu_all, data2)
    return

# 预测主函数
def FORECAST_main():
    forecast_main()
    return

# 预警主函数
def FOREWARN_main():

    return

# 主函数
if __name__ == '__main__':
    choice = 1

    # 多功能实现
    if choice == 1:
        EVALU_main()
    elif choice == 2:
        FORECAST_main()
    elif choice == 3:
        FOREWARN_main()
    else:
        print("无该功能")
