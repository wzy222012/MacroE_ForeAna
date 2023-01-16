import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from EVALU.EVALU_IndProce import pre_import


def fore_expsmooth_1(data_primary, para_expsmooth):
    '''
    一次指数平滑函数
    :param data_primary: 原始数据
    :param para_expsmooth: 平滑系数α
    :return: 平滑后数据
    '''
    data_expsmooth_1 = np.array(data_primary[0]).reshape(1)
    for i in range(len(data_primary) - 1):
        data_expsmooth_1 = np.append(data_expsmooth_1, para_expsmooth * data_primary[i + 1]
                                     + (1 - para_expsmooth) * data_expsmooth_1[i])
    data_expsmooth_1.reshape(1, -1)
    return data_expsmooth_1


def fore_expsmooth_3(data_primary, para_expsmooth, para_time):
    '''
    三次指数平滑函数
    :param data_primary: 原始数据
    :param para_expsmooth: 平滑系数
    :param para_time: 间隔周期
    :return: 参数a,b,c 预测值f_t+1
    '''
    # 初始化
    length = len(data_primary)
    data_exp1 = fore_expsmooth_1(data_primary.copy(), para_expsmooth)
    # plt.plot(range(2010, 2020), data_exp1, 'g')
    data_exp2 = fore_expsmooth_1(data_exp1.copy(), para_expsmooth)
    a_double = 2 * data_exp1 - data_exp2  # 计算二次指数平滑的a
    b_double = (para_expsmooth / (1 - para_expsmooth)) * (data_exp1 - data_exp2)  # 计算二次指数平滑的b
    s_pre_double = np.zeros(data_exp2.shape)  # 建立预测轴

    for i in range(1, length):
        s_pre_double[i] = a_double[i - 1] + b_double[i - 1]  # 循环计算每一年的二次指数平滑法的预测值，下面三次指数平滑法原理相同
    pre_next_year = a_double[-1] + b_double[-1] * 1  # 预测下一年
    pre_next_two_year = a_double[-1] + b_double[-1] * 2  # 预测下两年
    insert_year = np.array([pre_next_year, pre_next_two_year])
    s_pre_double = np.insert(s_pre_double, len(s_pre_double), values=np.array([pre_next_year, pre_next_two_year]),
                             axis=0)  # 组合预测值
    # plt.plot(range(2010, 2022), s_pre_double, 'r')

    data_exp3 = fore_expsmooth_1(data_exp2.copy(), para_expsmooth)
    a_triple = 3 * data_exp1 - 3 * data_exp2 + data_exp3
    b_triple = (para_expsmooth / (2 * ((1 - para_expsmooth) ** 2))) * (
            (6 - 5 * para_expsmooth) * data_exp1 - 2 * ((5 - 4 * para_expsmooth) * data_exp2) + (
                4 - 3 * para_expsmooth) * data_exp3)
    c_triple = ((para_expsmooth ** 2) / (2 * ((1 - para_expsmooth) ** 2))) * (data_exp1 - 2 * data_exp2 + data_exp3)

    s_pre_triple = np.zeros(data_exp3.shape)
    # print(s_pre_triple)
    for i in range(1, length):
        s_pre_triple[i] = a_triple[i - 1] + b_triple[i - 1] * 1 + c_triple[i - 1] * (1 ** 2)
    pre_next_year = a_triple[-1] + b_triple[-1] * 1 + c_triple[-1] * (1 ** 2)
    pre_next_two_year = a_triple[-1] + b_triple[-1] * 2 + c_triple[-1] * (2 ** 2)
    insert_year = np.array([pre_next_year, pre_next_two_year])
    print(s_pre_triple, data_primary)
    s_pre_triple = np.insert(s_pre_triple, len(s_pre_triple), values=insert_year,
                             axis=0)
    s_pre_triple[0] = data_primary[0]

    plt.plot(range(2010, 2020), data_primary, 'b', label='real')
    plt.plot(range(2010, 2022), s_pre_triple, 'brown', label='predict')
    plt.legend(['real', 'predict'], bbox_to_anchor=[0.4, 0.5])
    plt.show()

    return data_exp1, data_exp2, data_exp3, s_pre_double, s_pre_triple


def fore_plot(data_primary, data_exp1, data_exp2, data_sec, s_pre_triple):
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8), dpi=500)
    ax = plt.axes()
    # x_tick = Time_list[train_size:]
    # plt.plot(x_tick, pre, 'r', label='prediction')
    # plt.plot(x_tick, y_test, 'b', label='real')
    ax.set_xticks(range(2010, 2022, 2))
    plt.plot(range(2010, 2020), data_primary, 'b', label='real')
    plt.plot(range(2010, 2022), s_pre_triple, 'brown', label='predict')
    # plt.plot(data_Y[train_size: total_size + 1], y_test, 'b', label='real')
    plt.tick_params(axis='y', labelcolor='k', labelsize=5)
    # plt.xticks(range(1, len(x_tick), 5), rotation=45)
    # plt.xticks(range(1, len(data_Y[train_size: total_size + 1]), 5), rotation=45)
    plt.legend(['real', 'predict'], bbox_to_anchor=[0.4, 0.5])
    plt.show()
    return


# 单纯对一个时间序列做平滑并预测
def fore_process(data_primary):
    '''
    :param data_primary: 输入待预测时间序列
    :return:返回指数平滑后的预测值
    '''
    data_sec = np.array(data_primary)[::-1]
    data_exp1, data_exp2, data_fin, s_pre_double, s_pre_triple = fore_expsmooth_3(data_sec, 0.7, 1)
    # fore_plot(data_sec, data_exp1, data_exp2, data_fin, s_pre_triple)
    # print(s_pre_triple)
    return s_pre_triple


def fore_main(csv_Use, csv_Percon, csv_Pergdp, csv_Con, csv_Mix):
    '''
    输入参数：对应数据
    :param csv_Use:总能源消费量
    :param csv_Percon:人均能耗
    :param csv_Pergdp:能源强度
    :param csv_Con:能源消费弹性系数
    :param csv_Mix:能源消费结构
    :return: 五个仅带指数平滑预测值的原始序列，Use，Con等等…………
    '''

    length = (csv_Use.shape[0] + 2, csv_Use.shape[1])
    csv_Use_re = pd.DataFrame(np.zeros(length), index=range(2010, 2022), columns=csv_Use.columns)
    for i in range(0, csv_Use.shape[1]):
        data_fore = np.append(csv_Use.iloc[:, i][::-1], fore_process(csv_Use.iloc[:, i])[-2:])
        csv_Use_re.iloc[:, i] = data_fore

    length = (csv_Use.shape[0] + 2, csv_Percon.shape[1])
    csv_Percon_re = pd.DataFrame(np.zeros(length), index=range(2010, 2022), columns=csv_Percon.columns)
    for i in range(0, csv_Percon.shape[1]):
        data_fore = np.append(csv_Percon.iloc[:, i][::-1], fore_process(csv_Percon.iloc[:, i])[-2:])
        csv_Percon_re.iloc[:, i] = data_fore

    csv_Pergdp = csv_Pergdp.dropna(axis=0)
    length = (csv_Pergdp.shape[0] + 2, csv_Pergdp.shape[1])
    csv_Pergdp_re = pd.DataFrame(np.zeros(length), index=range(2010, 2022), columns=csv_Pergdp.columns)
    for i in range(0, csv_Pergdp.shape[1]):
        data_fore = np.append(csv_Pergdp.iloc[:, i][::-1], fore_process(csv_Pergdp.iloc[:, i])[-2:])
        csv_Pergdp_re.iloc[:, i] = data_fore

    length = (csv_Con.shape[0] + 2, csv_Con.shape[1])
    csv_Con_re = pd.DataFrame(np.zeros(length), index=range(2010, 2022), columns=csv_Con.columns)
    for i in range(0, csv_Con.shape[1]):
        data_fore = np.append(csv_Con.iloc[:, i][::-1], fore_process(csv_Con.iloc[:, i])[-2:])
        csv_Con_re.iloc[:, i] = data_fore

    length = (csv_Mix.shape[0] + 2, csv_Mix.shape[1])
    csv_Mix_re = pd.DataFrame(np.zeros(length), index=range(2010, 2022), columns=csv_Mix.columns)
    for i in range(0, csv_Mix.shape[1]):
        data_fore = np.append(csv_Mix.iloc[:, i][::-1], fore_process(csv_Mix.iloc[:, i])[-2:])
        csv_Mix_re.iloc[:, i] = data_fore

    return csv_Use_re, csv_Percon_re, csv_Pergdp_re, csv_Con_re, csv_Mix_re

# fore_process(pd.read_csv('./data1/EnergyUse.csv').iloc[:, 2])
