# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

'''
说明：1.该子文件实现雷达图的展示
     2.子函数radar实现雷达图
       输入变量：df，dataFrame类型; 列名称为对应的指标名称; 行名称为对应主体名称
       无返回值
     3.测试无误后，参与主文件测试
       调用方法：plot_radar(待展示数据)
'''

# 作雷达图
def plot_radar(df, df_score):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')

    # 导入数据
    data = df.values
    feature = df.columns.values

    # 将极坐标根据数据长度进行等分
    angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    plt.figure(figsize=(10, 6), dpi=100)

    # 雷达图
    ax = plt.subplot(121, polar=True)
    # 设置图形的大小
    # 样式
    sam = ['r-', 'b-', 'y-', 'k-', 'w-', 'c-', 'm-', 'g-']

    for i in range(len(data)):
        values = data[i]
        # 雷达图一圈封闭起来
        values = np.concatenate((values, [values[0]]))
        # 绘制折线图
        if i > 7:
            ax.plot(angles, values, sam[((i + 1) % 8) - 1], linewidth=1)
        else:
            ax.plot(angles, values, sam[i], linewidth=1)
        # ax.fill(angles, values, alpha=0.5) # 填充颜色
        ax.set_ylim(auto=True)  # 设置雷达图的范围

    # 设置雷达图中每一项的标签显示
    ang = angles * 180 / np.pi
    ax.set_thetagrids(ang[:-1], feature)
    # 设置雷达图的0度起始位置
    ax.set_theta_zero_location('N')
    # 设置雷达图的坐标刻度范围
    ax.set_rlim(40, 100)
    ax.set_title("中国宏观经济评估雷达图", x=0.5, y=1.15)
    ax.legend(df.index, bbox_to_anchor=[0, 1.25])

    # 柱状图
    bx = plt.subplot(122)
    bx.bar(df_score.index, df_score.iloc[:, 0], color='green')
    bx.set_ylim(60, 85)
    bx.set_xticks(range(2001, 2016, 2))
    bx.set_xlabel("年份")
    bx.set_ylabel("得分")
    bx.set_title("中国经济评估综合得分")
    for x, y in zip(df_score.index, df_score.iloc[:, 0]):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=6)
    print(df_score)

    plt.subplots_adjust(left=None, right=0.95, wspace=0.5)

    plt.show()

