import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, normalize

'''
说明：1.该子文件实现各指标权重的计算
     2.可以通过右上角“选择运行/调试配置”选’temp‘实现子文件的直接测试
       输入变量：df，dataFrame类型; 列名称为对应的指标名称; 行名称为对应主体名称
       返回变量：    dataFrame类型；列名称为指标名称和指标权重大小
     3.测试无误后，参与主文件测试
       调用方法：形如data = weight_add（df)，计算加权后的各指标组合权重值 （后续组合权重法完成后不同）
'''

'''------------------------ ---------数据区，后续删除，在主文件中导入数据------------------------------------------'''
df = pd.DataFrame([[100, 90, 50],
                   [80, 80, 80],
                   [70, 70, 90]], columns=['d1', 'd2', 'd3'])

'''------------------------------------------一堆预备函数--------------------------------------------------'''

# --归一化-- #
# 标准化
def normalize_Standard(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns.values)
# 正则化
def normalize_Norm(df):
    scaler = Normalizer(norm='l2')
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns.values)
# MinMax归一化（0-1 归一化）
def normalize_MinMax(df):
    scaler = MinMaxScaler(feature_range=(0, 100))
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns.values)

# --统计量-- #
# 均值
def compute_mat_mean(df):
    return pd.DataFrame(df.mean(), columns=['mean_value'])
# 标准差（使用df.std）
def compute_mat_std(df):
    return pd.DataFrame(df.std(), columns=['std_value'])
# 方差（标准差的平方）
def compute_mat_var(df):
    df_std = pd.DataFrame(df.std(), columns=['std_value'])
    df_std['var_value'] = df_std['std_value'].apply(lambda x: x ** 2)
    return pd.DataFrame(df_std['var_value'], columns=['var_value'])
# 相关矩阵（相关系数矩阵）
def compute_mat_corr(df):
    return pd.DataFrame(df.corr(), columns=df.columns.values)

'''-----------------------------------------数据权重计算函数------------------------------------------------'''

# --熵权法函数-- #
# 计算指标的熵值 （输入:标准化df; 输出:熵值df; ）
def compute_entropy(df):
    col_names = df.columns.values
    df_mid = df[col_names]
    new_col_names = []
    for cc in col_names:
        new_cc = '{}_1'.format(cc)
        new_col_names.append(new_cc)
        num_cc = df_mid[cc].count()
        sum_cc = df_mid[cc].sum()
        df_mid.loc[df_mid[cc] > 0, new_cc] = df_mid.loc[df_mid[cc] > 0, cc] \
            .apply(lambda x: 0 - (x / sum_cc * math.log(x / sum_cc)) / math.log(num_cc))
        df_mid.loc[df_mid[cc] == 0, new_cc] = 0
    df_mid = df_mid[new_col_names]
    df_mid.columns = col_names
    return pd.DataFrame(df_mid.sum(), columns=['etp_value'])
# 根据熵值计算权重 (输入:熵值df; 输出:权重df;)
def compute_entropy_weight(df):
    df_mid = df[df.columns.values]
    num_cc = df_mid.count()
    sum_cc = df_mid.sum()
    df_mid['weight_value'] = df_mid['etp_value'].apply(lambda x: (1 - x) / (num_cc - sum_cc))
    df_mid['p_name'] = df_mid.index.values
    return df_mid[['p_name', 'weight_value']]
# 熵权法主函数 (输入:指标df; 输出:权重df;)
def weight_entropy(df):
    df_mid = compute_entropy_weight(compute_entropy(normalize_MinMax(df)))
    df_mid.index = range(df_mid.shape[0])
    return df_mid

# --CRITIC权重法-- #
def weight_critic(df):
    df_scale = normalize_MinMax(df)
    # 标准差
    df_std = compute_mat_std(df_scale)
    df_std['p_name'] = df_std.index.values
    # 相关系数矩阵
    df_corr = compute_mat_corr(df_scale)
    col_names = df_corr.columns.values
    # 求相关系数和
    df_mid = df_corr[col_names]
    new_col_names = []
    for cc in col_names:
        new_cc = '{}_1'.format(cc)
        new_col_names.append(new_cc)
        df_mid[new_cc] = df_mid[cc].apply(lambda x: 1 - x)
    df_mid = df_mid[new_col_names]
    df_mid = pd.DataFrame(df_mid.sum(), columns=['r_value'])
    df_mid['p_name'] = col_names
    # 标准差与相关系数相乘
    df_mix = pd.merge(df_std, df_mid, on='p_name')
    df_mix['pp_value'] = df_mix.apply(lambda x: x['std_value'] * x['r_value'], axis=1)
    # 最后计算权重值
    sum_pp = df_mix['pp_value'].sum()
    df_mix['weight_value'] = df_mix['pp_value'].apply(lambda x: x / sum_pp)
    return df_mix[['p_name', 'weight_value']]

# --变异系数法-- #
def weight_inform(df):
    df_scale = normalize_MinMax(df)
    df_mid = df_scale[df_scale.columns.values]
    # 计算标准差 和 均值
    df_std = compute_mat_std(df_mid)
    df_std['p_name'] = df_std.index.values
    df_mean = compute_mat_mean(df_mid)
    df_mean['p_name'] = df_mean.index.values
    # 合并两个df
    df_mix = pd.merge(df_std, df_mean, on='p_name')
    # 计算变异系数，再计算权重
    df_mix['cof_value'] = df_mix.apply(lambda x: x['std_value'] / x['mean_value'], axis=1)
    sum_cof = df_mix['cof_value'].sum()
    df_mix['weight_value'] = df_mix['cof_value'].apply(lambda x: x / sum_cof)
    return df_mix[['p_name', 'weight_value']]

'''---------------------------------------------权重组合计算-------------------------------------------------------'''

# 面向最优化目标:需要先建立量化指标才行
# 将多种方法得到的几个指标权重加权，得到指标的综合权重
def weight_comb(df):
    # 指标相关
    line = df.shape[1]  # 指标数
    column = 3  # 赋权方法数
    weight = np.zeros((column))  # 组合权重矩阵 (指标数X指标数X赋权方法数)

    # 获取各个指标权重值
    df_mixed = np.zeros((line, column))
    df_mixed[:, 0] = weight_entropy(df).iloc[:, 1]
    df_mixed[:, 1] = weight_critic(df).iloc[:, 1]
    df_mixed[:, 2] = weight_inform(df).iloc[:, 1]

    # 计算加权权重 (离差最大)
    for t in range(0, column):
        for j in range(0, line):
            for l in range(0, line):
                weight[t] += abs(df_mixed[j][t] - df_mixed[l][t])
    weight_all = 0
    # 将离差值归一化
    weight = pow(weight, 2)
    for t in range(0, column):
        weight_all += weight[t]
    weight /= weight_all
    # weight = np.ones((3, 1)) / 3

    # 各个权重按权累加
    df_mix = pd.DataFrame(np.zeros((line, 2)), columns=['p_name', 'weight_value'])
    df_mix['weight_value'] = (df_mixed[:, 0] * weight[0] +
                              df_mixed[:, 1] * weight[1] +
                              df_mixed[:, 2] * weight[2]
                              )
    df_mix['p_name'] = df.columns
    # print(df_mix)
    return df_mix

# 实现各个指标的累加，得到最后的指标
def weight_add(df):
    # 获取数据 权值
    weight_mix = weight_comb(df)
    # print(weight_mix)
    line = df.shape[0]
    column = df.shape[1]
    df_mix = np.zeros(line)

    # 加权计算
    for i in range(0, line):
        score_weighted = 0
        for j in range(0, column):
            score_weighted += df.iloc[i, j] * weight_mix.iloc[j, 1]
        df_mix[i] = score_weighted

    # 格式化
    evalu_score = pd.DataFrame(df_mix, index=df.index, columns=['经济总得分'])
    return evalu_score

# print(weight_comb(df))
