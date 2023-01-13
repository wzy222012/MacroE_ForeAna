# coding=utf-8
import numpy as np
import pandas as pd
import sklearn.linear_model as lin_mo

'''
说明：1.该子文件实现数据导入、各指标量化
     2.子函数pro_quantify_XXX实现对于某个能源数据的指标量化
       输入变量:csv_name,对应的指标数据csv表名. csv格式参考同文件夹文件'gyreasso.csv'.
       输出变量：evalu_XXX(指标名), 统一为dataframe 类型
     3.测试无误后，参与主文件测试
       调用方法:形如pro_quantify_XXXX(‘csv.name’),计算得到某个指标的分值。
'''

'''-------------------------------------------数据导入子函数------------------------------------------------'''

# 数据导入子函数，将csv转为dataFrame格式类型
def pre_import(csv_name):
    # header=0 表示第一行为表头
    df = pd.read_csv(csv_name, header=0)
    df.index = df.iloc[:, 0]
    df = df.drop(df.columns[0], axis=1)
    return df

'''-------------------------------------------量化用的预备函数------------------------------------------------'''

# 灰色关联法分析
def pre_method_greyasso(df):
    # 自定义+初始化
    line = len(df)
    column = df.shape[1]
    epsilon_line = np.zeros((line, column), dtype=float)
    greyasso_temp = np.empty((column, 1), dtype=float)

    # 作无量纲化处理
    for i in range(1, line):
        df.iloc[i, :] /= df.iloc[0, :]
    df.iloc[0] = np.ones(column)

    # 计算灰色关联系数
    for j in range(column):
        delta_max = 0
        delta_min = 100
        # 算差分的最大值和最小值
        for i in range(line):
            delta_temp = abs(df.iloc[i, 0] - df.iloc[i, j])
            if delta_temp > delta_max:
                delta_max = delta_temp
            if delta_temp < delta_min:
                delta_min = delta_temp
        # 算灰色关联系数
        for i in range(line):
            delta_temp = abs(df.iloc[i, 0] - df.iloc[i, j])
            if 0.5 * delta_max + delta_temp != 0:
                epsilon_line[i][j] = (delta_min + 0.5 * delta_max) / \
                                     (0.5 * delta_max + delta_temp)
            else:
                epsilon_line[i][j] = 0

    epsilon_line = np.delete(epsilon_line, 0, 1)

    # 计算灰色关联度
    line = epsilon_line.shape[0]
    column = epsilon_line.shape[1]
    for j in range(column):
        epsilon = 0
        for i in range(1, line):
            epsilon += epsilon_line[i][j]
        greyasso_temp[j] = epsilon / (line - 1)
    greyasso = pd.DataFrame(np.transpose(np.delete(greyasso_temp, [column])), index=df.columns[1:column+1])

    return greyasso

# 线性回归法分析
def pre_method_lineregre(df):
    # 初始化+定义
    x_data = (df.iloc[:, 1])[:, np.newaxis]
    y_data = (df.iloc[:, 0])[:, np.newaxis]
    lineregre = np.empty((2, 1), dtype=float)

    # 计算线性回归系数
    model = lin_mo.LinearRegression()
    model.fit(x_data, y_data)
    lineregre[0] = model.coef_[0]
    lineregre[1] = model.intercept_

    # 格式化
    lineregre = pd.DataFrame(lineregre, index=['斜率', '截距'])
    return lineregre

'''------------------------------------------能源指标量化计算函数------------------------------------------------'''

# 能耗指标1\能源生产指标2\能源投资总量3
def pre_quantify_EnergySum(df, type):
    # print(df)
    # linereg = pre_method_lineregre(df.copy())
    # line = df.shape[0]
    # data = np.zeros((line, 1))
    #
    # # 代入线性回归函数
    # for i in range(0, line):
    #     data[i] = linereg.loc['斜率'] * df.iloc[i, 1] + linereg.loc['截距']
    #
    # # 百分值化
    # for i in range(0, line):
    #     data[i] /= df.iloc[i, 0]
    # # print(data)
    # data /= max(data) / 100
    line = df.shape[0]
    data_sum = df.iloc[:, 1]
    data = np.zeros((line, 1))

    percon_max = max(data_sum)
    percon_min = min(data_sum)

    for i in range(0, line):
        data[i] = (data_sum.iloc[i] - percon_min) / (percon_max - percon_min) * 30 + 60

    evalu_EnerSum = pd.DataFrame(data, index=df.index, columns=[df.columns[1]])
    return evalu_EnerSum

# 人均能源消费指标
def pre_quantify_EnerPercon(df):
    line = df.shape[0]
    data_percon = df.iloc[:, 1]
    data = np.zeros((line, 1))

    percon_max = max(data_percon)
    percon_min = min(data_percon)

    for i in range(0, line):
        data[i] = (data_percon.iloc[i] - percon_min) / (percon_max - percon_min) * 30 + 60
    evalu_EnerPercon = pd.DataFrame(data, index=df.index, columns=[df.columns[1]])
    return evalu_EnerPercon

def pre_quantify_EnerInte(df):
    line = df.shape[0]
    column = df.shape[1]
    data_percon = df.iloc[:, 0]
    data = np.zeros((line, 1))

    for i in range(0, line):
        data_percon.iloc[i] = 1 / data_percon.iloc[i]

    percon_max = max(data_percon)
    percon_min = min(data_percon)

    for i in range(0, line):
        data[i] = (data_percon.iloc[i] - percon_min) / (percon_max - percon_min) * 30 + 60

    evalu_EnerInte = pd.DataFrame(data, index=df.index, columns=[df.columns[0]])
    return evalu_EnerInte

# 能源消费结构指标
def pre_quantify_EnergyMix(df):
    # 导入数据 初始化
    line = df.shape[0]
    column = df.shape[1]
    evalu_EnerMix = np.zeros((line, 1))
    # print(df)

    # 计算各个能源部分与GDP的关联度
    greyasso = pre_method_greyasso(df.copy())
    # print(greyasso)
    # 计算能源经济结构得分
    for i in range(line):
        score_ene = 0
        for j in range(1, column):
            score_ene += df.iloc[i, j] * greyasso.iloc[j - 1]
        evalu_EnerMix[i] = score_ene / df.iloc[i, 1] * 65
    # evalu_EnerMix = (evalu_EnerMix - 60) * 2.5 + 60
    # print(evalu_EnerMix)

    # 形式化一下
    evalu_EnerMix = pd.DataFrame(evalu_EnerMix, index=df.index, columns=['能源消费结构'])
    # print(evalu_EnerMix)
    return evalu_EnerMix

# 能源消费弹性系数
def pre_quantify_EnergyCon(df):
    df_temp = pow(1/df.iloc[:, 0], 0.05)
    df_temp *= 90 / max(df_temp)

    evalu_EnerCon = pd.DataFrame(df_temp, index=df.index, columns=[df.columns[0]])
    return evalu_EnerCon
'''-------------------------------------------能源指标合并函数------------------------------------------------'''

# 各个指标合成指定的格式Dataframe
def pre_quantify_Mix(csv_Use, csv_Percon, csv_Pergdp, csv_Con, csv_Mix):
    '''
    输入参数：对应数据
    :param csv_Use:总能源消费量
    :param csv_Percon:人均能耗
    :param csv_Pergdp:能源强度
    :param csv_Con:能源消费弹性系数
    :param csv_Mix:能源消费结构
    :return:各个指标量化值
    '''
    # 获取各个指标
    evalu_EnerUse = pre_quantify_EnergySum(csv_Use, 1)
    evalu_EnerPro = pre_quantify_EnerPercon(csv_Percon)
    evalu_EnerInv = pre_quantify_EnerInte(csv_Pergdp)
    evalu_EnerCon = pre_quantify_EnergyCon(csv_Con)
    evalu_EnerMix = pre_quantify_EnergyMix(csv_Mix)

    # 初始化
    length = evalu_EnerMix.shape[0]
    evalu_score = pd.DataFrame(np.zeros((length, 1)), index=range(2009+length, 2009, -1))
    # evalu_score = pd.DataFrame(np.zeros((length, 1)), index=range(2015, 2000, -1))

    # 合并
    evalu_score = pd.merge(evalu_score, evalu_EnerUse, left_index=True, right_index=True)
    evalu_score = pd.merge(evalu_score, evalu_EnerPro, left_index=True, right_index=True)
    evalu_score = pd.merge(evalu_score, evalu_EnerInv, left_index=True, right_index=True)
    evalu_score = pd.merge(evalu_score, evalu_EnerCon, left_index=True, right_index=True)
    evalu_score = pd.merge(evalu_score, evalu_EnerMix, left_index=True, right_index=True)
    '''在这里加合并指标，和上式格式一致----------------------------------------------------------------------'''
    evalu_score = evalu_score.drop([0], axis=1)

    # 设置标签对齐
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 180)  # 设置打印宽度(**重要**)

    return evalu_score

# print(pre_quantify_Mix())
# evalu_EnerInv = pre_quantify_EnergySum('./data/EnergyInv.csv')
# pre_quantify_EnergyCon('./data/EnergyCon.csv')
