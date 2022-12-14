import numpy as np
import datetime as dt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras import regularizers
from sklearn.neural_network import BernoulliRBM
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


class DBN:
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            hidden_layer,
            learning_rate_rbm=0.0001,
            batch_size_rbm=100,
            n_epochs_rbm=30,
            verbose_rbm=1,
            random_seed_rbm=1300,
            activation_function_nn='relu',
            learning_rate_nn=0.005,
            batch_size_nn=100,
            n_epochs_nn=10,
            verbose_nn=1,
            decay_rate=0):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.hidden_layer = hidden_layer
        self.learning_rate_rbm = learning_rate_rbm
        self.batch_size_rbm = batch_size_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.verbose_rbm = verbose_rbm
        self.random_seed = random_seed_rbm
        self.activation_function_nn = activation_function_nn
        self.learning_rate_nn = learning_rate_nn
        self.batch_size_nn = batch_size_nn
        self.n_epochs_nn = n_epochs_nn
        self.verbose_nn = verbose_nn
        self.decay_rate = decay_rate
        self.weight_rbm = []
        self.bias_rbm = []
        self.test_rms = 0
        self.result = []
        self.model = Sequential()

    def pretraining(self):                                  # 预训练
        input_layer = self.x_train
        for i in range(len(self.hidden_layer)):
            print("DBN Layer {0} Pre-training".format(i + 1))
            rbm = BernoulliRBM(n_components=self.hidden_layer[i],
                               learning_rate=self.learning_rate_rbm,
                               batch_size=self.batch_size_rbm,
                               n_iter=self.n_epochs_rbm,
                               verbose=self.verbose_rbm,
                               random_state=self.verbose_rbm)
            rbm.fit(input_layer)
            # size of weight matrix is [input_layer, hidden_layer]
            self.weight_rbm.append(rbm.components_.T)
            self.bias_rbm.append(rbm.intercept_hidden_)
            input_layer = rbm.transform(input_layer)
        print('Pre-training finish.')

    def finetuning(self):                                # 调参
        print('Fine-tuning start.')

        for i in range(0, len(self.hidden_layer)):
            if i == 0:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function_nn,
                                     input_dim=self.x_train.shape[1]))
            elif i >= 1:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function_nn))
            else:
                pass
            layer = self.model.layers[i]
            layer.set_weights([self.weight_rbm[i], self.bias_rbm[i]])
        if self.y_train.ndim == 1:
            self.model.add(Dense(1, activation=None, kernel_regularizer=regularizers.l2(0.01)))
        else:
            self.model.add(Dense(self.y_train.shape[1], activation=None))

        sgd = SGD(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.model.compile(loss='mse',
                           optimizer=sgd,
                           )
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size_nn,
                       epochs=self.n_epochs_nn, verbose=self.verbose_nn)
        print('Fine-tuning finish.')
        self.test_rms = self.model.evaluate(self.x_test, self.y_test)
        self.result = np.array(self.model.predict(self.x_test))

    def predict(self, series):
        return np.array(self.model.predict(series))          # 预测

    def save(self):
        self.model.save(r'D:\研究生资料\供电服务\供电服务2021\结果图表\trained_DBN.h5')     # 保存


# In[] 生成时间序列
def nday_list(date_start, date_end):
    before_n_days = []
    for i in range((dt.datetime.strptime(date_end, '%Y-%m-%d') - dt.datetime.strptime(date_start, '%Y-%m-%d')).days):
        before_n_days.append(str(dt.datetime.strptime(date_start, '%Y-%m-%d') + dt.timedelta(days=i)))

    return pd.DataFrame(before_n_days)[0].str[:-9]


if __name__ == "__main__":
    date_start, date_end = '2020-01-11', '2021-06-24'
    Time_list = nday_list(date_start, date_end).tolist()
    len_T = -(dt.datetime.strptime(date_start, '%Y-%m-%d') - dt.datetime.strptime(date_end, '%Y-%m-%d')).days
    test_start, test_end = str(dt.datetime.strptime(date_start, '%Y-%m-%d') + dt.timedelta(days=371))[:-9], '2021-06-24'
    Test_list = nday_list(test_start, test_end).tolist()
    len_Test = -(dt.datetime.strptime(test_start, '%Y-%m-%d') - dt.datetime.strptime(test_end, '%Y-%m-%d')).days

    data1 = pd.read_excel('D:/研究生资料/供电服务/供电服务2021/结果图表/异动指标1.xlsx')
    data1 = data1.set_index('Unnamed: 0')
    data2 = pd.read_excel('D:/研究生资料/供电服务/供电服务2021/结果图表/异动指标2.xlsx')
    data2 = data2.set_index('Unnamed: 0')
    data3 = pd.read_excel('D:/研究生资料/供电服务/供电服务2021/结果图表/异动指标3.xlsx')
    data3 = data3.set_index('Unnamed: 0')
    data4 = pd.read_excel('D:/研究生资料/供电服务/供电服务2021/结果图表/异动指标4.xlsx')
    data4 = data4.set_index('Unnamed: 0')
    data_Y = pd.read_excel('D:/研究生资料/供电服务/供电服务2021/数据/工单全业务每日汇总统计3/标签_3340140.xlsx')

    data_la1 = data1.loc[3340140, :]
    data_la2 = data2.loc[3340140, :]
    data_la3 = data3.loc[3340140, :]
    data_la4 = data4.loc[3340140, :]                        # 读取数据

    max_value1 = np.max(data_la1)
    min_value1 = np.min(data_la1)
    data_la1 = (data_la1 - min_value1) / (max_value1 - min_value1)

    max_value2 = np.max(data_la2)
    min_value2 = np.min(data_la2)
    data_la2 = (data_la2 - min_value2) / (max_value2 - min_value2)

    max_value3 = np.max(data_la3)
    min_value3 = np.min(data_la3)
    data_la3 = (data_la3 - min_value3) / (max_value3 - min_value3)

    max_value4 = np.max(data_la4)
    min_value4 = np.min(data_la4)
    data_la4 = (data_la4 - min_value4) / (max_value4 - min_value4)     # 归一化 加快进程

    total_size = np.size(data_Y, axis=1)
    train_size = int(np.size(data_Y, axis=1) * 0.7)                    # 训练与测试集

    x_train = np.zeros([4, train_size])
    y_train = np.zeros([1, train_size])
    x_test = np.zeros([4, total_size - train_size])
    y_test = np.zeros([1, total_size - train_size])
    for i in range(train_size):
        x_train[0, i] = data_la1[i]
        x_train[1, i] = data_la2[i]
        x_train[2, i] = data_la3[i]
        x_train[3, i] = data_la4[i]
        y_train[0, i] = data_Y.iloc[0, i]
    for i in range(total_size - train_size):
        x_test[0, i] = data_la1[train_size + i]
        x_test[1, i] = data_la2[train_size + i]
        x_test[2, i] = data_la3[train_size + i]
        x_test[3, i] = data_la4[train_size + i]
        y_test[0, i] = data_Y.iloc[0, train_size + i]
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    mae_all = []
    mse_all = []                                                        # 误差

    # 对模型进行训练和保存
    #     trained_model, scaler = multioutput(df.values, 0.9, 192, 1, 200, 300, 50)

    dbn1 = DBN(x_train=x_train,
               y_train=y_train,
               x_test=x_test,
               y_test=y_test,
               hidden_layer=[250],
               learning_rate_rbm=0.005,
               batch_size_rbm=150,
               n_epochs_rbm=200,
               verbose_rbm=1,
               random_seed_rbm=500,
               activation_function_nn='tanh',
               learning_rate_nn=0.005,
               batch_size_nn=200,
               n_epochs_nn=300,
               verbose_nn=1,
               decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dbn1.save()

    model = load_model(r'D:\研究生资料\供电服务\供电服务2021\结果图表\trained_DBN.h5')
    pre = model.predict(x_test)
    pre_max = np.max(pre)
    # for i in range(total_size - train_size):
    #     if pre[i] > pre_max / 2:
    #         pre[i] = 1
    #     else:
    #         pre[i] = 0

    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False                                       # 生成图表
    plt.figure(figsize=(24, 8), dpi=500)
    x_tick = Time_list[train_size:]

    plt.plot(x_tick, pre, 'r', label='prediction')
    plt.plot(x_tick, y_test, 'b', label='real')
    plt.tick_params(axis='y', labelcolor='k')
    plt.xticks(range(1, len(x_tick), 5), rotation=45)
    plt.legend(['prediction', 'real'])
    plt.show()



