import numpy as np
import pandas as pd
import openpyxl

# 调用各个文件里想用的函数
from EVALU.EVALU_IndProce import pre_quantify_Mix
from EVALU.EVALU_IndProce import pre_import
from EVALU.EVALU_WeightCal import weight_comb
from EVALU.EVALU_WeightCal import weight_add
from EVALU.EVALU_RadarChart import plot_radar

from FORECAST.DBN_revised import forecast_main
from FORECAST.FORECAST_ExpSmooth import fore_main

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
    # 各指标量化值
    evalu_all = pre_quantify_Mix(pre_import('./data1/EnergyUse.csv'),
                                 pre_import('./data1/EnergyPercon.csv'),
                                 pre_import('./data1/EnergyPergdp.csv'),
                                 pre_import('./data1/EnergyCon.csv'),
                                 pre_import('./data1/EnergyMix.csv'))
    print(evalu_all)
    evalu_all.to_excel('./data1/evalu_all.xlsx')
    # 综合值
    data2 = weight_add(evalu_all)
    data2.to_excel('./data1/data2.xlsx')
    # plot_radar(evalu_all.iloc[0:8, :], data2)
    plot_radar(evalu_all.iloc[-8:, :], data2)
    # plot_radar(evalu_all, data2)
    return evalu_all, data2


# 预测主函数
def FORECAST_main():
    # print(pre_import('./data1/EnergyPergdp.csv'))
    forecast_main()
    # csv_Use_re, csv_Percon_re, csv_Pergdp_re, csv_Con_re, csv_Mix_re = fore_main(pre_import('./data1/EnergyUse.csv'),
    #                                                                              pre_import('./data1/EnergyPercon.csv'),
    #                                                                              pre_import('./data1/EnergyPergdp.csv'),
    #                                                                              pre_import('./data1/EnergyCon.csv'),
    #                                                                              pre_import('./data1/EnergyMix.csv'))
    # evalu_fore_all = pre_quantify_Mix(csv_Use_re, csv_Percon_re, csv_Pergdp_re, csv_Con_re, csv_Mix_re)
    # print(evalu_fore_all)
    # data3 = weight_add(evalu_fore_all)
    # plot_radar(evalu_fore_all.iloc[-8:, :], data3)
    return


# 预警主函数
def FOREWARN_main():
    return


# 主函数
if __name__ == '__main__':
    choice = 2

    # 多功能实现
    if choice == 1:
        EVALU_main()
    elif choice == 2:
        FORECAST_main()
    elif choice == 3:
        FOREWARN_main()
    else:
        print("无该功能")
