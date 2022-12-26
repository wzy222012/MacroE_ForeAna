import numpy as np
import matplotlib.pyplot as plt
from EVALU.EVALU_IndProce import pre_import

def fore_expsmooth_1(data_primary, para_expsmooth):
    '''
    一次指数平滑函数
    :param data_primary: 原始数据
    :param para_expsmooth: 平滑系数α
    :return: 平滑后数据
    '''
    data_expsmooth_1 = [data_primary[0]]
    for i in range(1, len(data_primary)):
        data_expsmooth_1.append(para_expsmooth * data_primary[i]
                                + (1 - para_expsmooth) * data_expsmooth_1[i - 1])
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
    data_exp1 = fore_expsmooth_1(data_primary.copy, para_expsmooth)
    data_exp2 = fore_expsmooth_1(data_primary.copy, para_expsmooth)
    data_exp3 = fore_expsmooth_1(data_primary.copy, para_expsmooth)

    a_tri = [0 for i in range(length)]
    b_tri = [0 for i in range(length)]
    c_tri = [0 for i in range(length)]
    f_tri = [0 for i in range(length)]

    for i in range(length):
        a = 3 * data_exp1[i] - 3 * data_exp2[i] + data_exp3[i]
        b = (para_expsmooth / (2 * ((1 - para_expsmooth) ** 2))) * \
            ((6 - 5 * para_expsmooth) * data_exp1[i] - 2 * ((5 - 4 * para_expsmooth) * data_exp2[i]) + (4 - 3 * para_expsmooth) * data_exp3[i])
        c = ((para_expsmooth ** 2) / (2 * ((1 - para_expsmooth) ** 2))) * (data_exp1[i] - 2 * data_exp2[i] + data_exp3[i])
        f = a + b * para_time + c * para_time
        a_tri[i] = a
        b_tri[i] = b
        c_tri[i] = c
        f_tri[i] = f
    return a_tri, b_tri, c_tri, f_tri

def fore_plot(data_primary, data_sec):

    return

def fore_main(csv_name):
    data_primary = pre_import(csv_name)
    data_sec = fore_expsmooth_3(data_primary)
    print(data_sec)
    # fore_plot(data_primary, data_sec)
    return data_sec

