import numpy as np
import datetime as dt
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import regularizers
from sklearn.neural_network import BernoulliRBM
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    def pretraining(self):
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

    def finetuning(self):
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
        return np.array(self.model.predict(series))

    def save(self):
        self.model.save(r'./data/trained_DBN.h5')


# In[] 生成时间序列
def nday_list(date_start, date_end):
    before_n_days = []
    for i in range((dt.datetime.strptime(date_end, '%Y-%m-%d') - dt.datetime.strptime(date_start, '%Y-%m-%d')).days):
        before_n_days.append(str(dt.datetime.strptime(date_start, '%Y-%m-%d') + dt.timedelta(days=i)))

    return pd.DataFrame(before_n_days)[0].str[:-9]


def model_train(x_train, y_train, x_test, y_test):
    dbn1 = DBN(x_train=x_train,
               y_train=y_train,
               x_test=x_test,
               y_test=y_test,
               hidden_layer=[250],
               learning_rate_rbm=0.0005,
               batch_size_rbm=150,
               n_epochs_rbm=200,
               verbose_rbm=1,
               random_seed_rbm=500,
               activation_function_nn='tanh',
               learning_rate_nn=0.001,
               batch_size_nn=200,
               n_epochs_nn=300,
               verbose_nn=1,
               decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dbn1.save()
    return


def forecast_main():
    date_start, date_end = '2020-01-11', '2021-06-24'
    Time_list = nday_list(date_start, date_end).tolist()
    len_T = -(dt.datetime.strptime(date_start, '%Y-%m-%d') - dt.datetime.strptime(date_end, '%Y-%m-%d')).days
    test_start, test_end = str(dt.datetime.strptime(date_start, '%Y-%m-%d') + dt.timedelta(days=371))[:-9], '2021-06-24'
    Test_list = nday_list(test_start, test_end).tolist()
    len_Test = -(dt.datetime.strptime(test_start, '%Y-%m-%d') - dt.datetime.strptime(test_end, '%Y-%m-%d')).days

    # 导入数据
    data1 = pd.read_excel('./data/test1.xlsx')
    # data1 = data1.set_index('Unnamed: 0')
    # data2 = pd.read_excel('./data/test2.xlsx')
    # # data2 = data2.set_index('Unnamed: 0')
    # data3 = pd.read_excel('./data/test3.xlsx')
    # # data3 = data3.set_index('Unnamed: 0')
    # data4 = pd.read_excel('./data/test4.xlsx')

    # data4 = data4.set_index('Unnamed: 0')
    # data_Y = pd.read_excel('D:/研究生资料/供电服务/供电服务2021/数据/工单全业务每日汇总统计3/标签_3340140.xlsx')

    data_la1 = data1.iloc[:, 6]
    # data_la2 = data2.iloc[:, 6]
    # data_la3 = data3.iloc[:, 6]
    # data_la4 = data4.iloc[:, 6]

    # 归一化
    max_value1 = np.max(data_la1)
    min_value1 = np.min(data_la1)
    data_la1 = (data_la1 - min_value1) / (max_value1 - min_value1)
    data_la_train = data_la1
    # max_value2 = np.max(data_la2)
    # min_value2 = np.min(data_la2)
    # data_la2 = (data_la2 - min_value2) / (max_value2 - min_value2)
    # max_value3 = np.max(data_la3)
    # min_value3 = np.min(data_la3)
    # data_la3 = (data_la3 - min_value3) / (max_value3 - min_value3)
    # max_value4 = np.max(data_la4)
    # min_value4 = np.min(data_la4)
    # data_la4 = (data_la4 - min_value4) / (max_value4 - min_value4)

    data_Y = data1.iloc[:, 0]

    # 各种数据长度计算
    total_size = np.size(data_Y)
    exchange_time = 5                                   # 交叉验证份数
    exchange_size = int(total_size / exchange_time)     # 交叉验证测试集长度
    train_size = total_size - exchange_size             # 交叉验证训练集长度
    rolling_size = 12                                   # 滚动长度

    for k in range(exchange_time):
        # 新建训练集X_train 和 测试集X_test
        x_train = np.zeros([rolling_size, train_size - rolling_size])
        y_train = np.zeros([1, train_size - rolling_size])
        x_test = np.zeros([rolling_size, exchange_size - rolling_size])
        y_test = np.zeros([1, exchange_size - rolling_size])

        # 滚动交叉验证测试集位置
        data_la_train = data_la1.drop(range(exchange_size * k, exchange_size * (k + 1)))
        data_la_train = data_la_train.reset_index(drop=True)
        data_la_test = data_la1[exchange_size * k: exchange_size * (k + 1)]

        # 导入训练集X_train 和 测试集X_test
        for i in range(train_size - rolling_size):
            for j in range(rolling_size):
                x_train[j, i] = data_la_train.iloc[i + j]
            y_train[0, i] = data_la_train.iloc[i + rolling_size]
        for i in range(exchange_size - rolling_size):
            for j in range(rolling_size):
                x_test[j, i] = data_la_test.iloc[i + j]
            y_test[0, i] = data_la_test.iloc[i + rolling_size]

        # 修下格式
        x_train = x_train.T
        y_train = y_train.T
        x_test = x_test.T
        y_test = y_test.T

        # 对模型进行训练和保存
        #     trained_model, scaler = multioutput(df.values, 0.9, 192, 1, 200, 300, 50)
        model_train(x_train, y_train, x_test, y_test)

        # 调用模型计算
        model = load_model(r'./data/trained_DBN.h5')
        pre = model.predict(x_test)
        if k == 0:
            pre_size = pre.size
            predic = np.zeros([pre_size, 1])
        predic += pre
        pre_max = np.max(pre)
    predic /= exchange_time
    # for i in range(total_size - train_size):
    #     if pre[i] > pre_max / 2:
    #         pre[i] = 1
    #     else:
    #         pre[i] = 0

    # 作图
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8), dpi=500)
    # x_tick = Time_list[train_size:]
    # plt.plot(x_tick, pre, 'r', label='prediction')
    # plt.plot(x_tick, y_test, 'b', label='real')
    plt.plot(data_Y[train_size: total_size - rolling_size], pre, 'b', label='predict')
    plt.plot(data_Y[train_size: total_size - rolling_size], predic, 'r', label='prediction')
    plt.plot(data_Y[train_size: total_size - rolling_size], y_test, 'g', label='real')
    # plt.plot(data_Y[train_size: total_size + 1], y_test, 'b', label='real')
    plt.tick_params(axis='y', labelcolor='k')
    # plt.xticks(range(1, len(x_tick), 5), rotation=45)
    # plt.xticks(range(1, len(data_Y[train_size: total_size + 1]), 5), rotation=45)
    plt.legend(['prediction', 'real'], bbox_to_anchor=[0.4, 0.5])
    plt.show()
    return
