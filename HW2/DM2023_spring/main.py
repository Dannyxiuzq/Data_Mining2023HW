"""
author:yqtong@buaa.edu.cn
date:2023-05-04
"""
import numpy as np
from TraceLoader import ObjectTrace
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
random_seed = 2023
np.random.seed(random_seed)
formation_l = {'单机': 0, '横队': 1, '三角': 2, '4': 3, '3': 4, '2': 5, '7': 6, '1': 7, '6': 8, '5': 9}


def extract_intention_data(trace_file):
    traceModel = ObjectTrace()
    with open(trace_file, 'r', encoding='utf-8') as fa:
        line_list = fa.readlines()
    init_timestamp = 0
    curr_time_planes = []
    for idx, line in enumerate(line_list[1:]):
        line_list = line.strip().split(',')
        timestamp = int(float(line_list[0]))
        if init_timestamp == timestamp:
            curr_time_planes.append(line_list)
        else:
            traceModel.parser_line(curr_time_planes)
            curr_time_planes = []
            init_timestamp = timestamp
            curr_time_planes.append(line_list)

    # todo 为了更清楚敌我双方的情况,数据读取完以后是不是可以做一些统计分析？
    trace_data = []
    labels = []
    seq_length = 5
    feature_dim = 4
    p = 0
    hist_trace_dict = traceModel.hist_trace_dict
    for Id, hist_trace in hist_trace_dict.items():
        #plot_3d_scatter(hist_trace['xyz'], hist_trace['Intention'], Id)
        print(hist_trace['Intention'], hist_trace['Formation'], hist_trace['Wingman'], hist_trace['Group'], hist_trace['Model'], hist_trace['Points_num'])
        p += hist_trace['Points_num']
        trace_length = len(hist_trace['Timestamp'])
        trace_intention = hist_trace['Intention'][0]
        xyz = hist_trace['xyz']
        speed = hist_trace['Speed']
        att = hist_trace['Att']
        for idx in range(trace_length):
            start = idx
            end = min(start+seq_length, trace_length)
            features = []
            for idy in range(start, end):
                # todo 能否挖掘出更多特征？哪些是有用特征，哪些是无用的？
                feature = xyz[idy] + [speed[idy]] + att[idy]# + [formation_l[hist_trace['Formation'][0]]]
                features.append(feature)

            # todo 这一步是做什么用？除了padding以外，还能怎么做？
            if end - start < seq_length:
                break
                for _ in range(seq_length - (end - start)):
                    features.append([0] * feature_dim)

            trace_data.append(features)
            labels.append([trace_intention])
    # todo 打印一下这两个array的形状，每一维的大小有什么含义吗？
    trace_data = np.array(trace_data)
    labels = np.array(labels)
    labels = labels.squeeze()
    print(labels.shape)
    print(trace_data.shape)
    print(p)
    return trace_data, labels


def train(train_file, model_path):
    trace_data, labels = extract_intention_data(train_file)
    #print(trace_data, labels)
    sample_num, seq_length, feature_dim = trace_data.shape[0], trace_data.shape[1], trace_data.shape[2]
    le = LabelEncoder()
    le.fit(intention_type)
    trace_labels = le.transform(labels)
    # todo 为啥数据集要打乱？
    shuffled_trace_data, shuffled_trace_labels = shuffle(trace_data, trace_labels)
    scaler = StandardScaler()
    shuffled_trace_data = shuffled_trace_data.reshape(sample_num * seq_length, -1)
    scaler.fit(shuffled_trace_data)
    # todo 标准化有什么作用？标准化处理对所有的模型都有帮助吗？
    shuffled_trace_data = scaler.transform(shuffled_trace_data)
    shuffled_trace_data = shuffled_trace_data.reshape(sample_num, -1)
    # todo 训练集和验证集的统计情况是怎么样的？
    x_train, x_dev, y_train, y_dev = train_test_split(shuffled_trace_data,
                                                      shuffled_trace_labels,
                                                      test_size=0.2,
                                                      random_state=random_seed)
    # todo 对于一个模型，如何寻找最优参数设置？
    clf = RandomForestClassifier(n_estimators=40,
                                 criterion='gini',
                                 max_depth=3,
                                 min_samples_split=2,
                                 bootstrap=True,
                                 random_state=0)

    #knn = KNeighborsClassifier(n_neighbors=5)
    #clf = BaggingClassifier(base_estimator=knn, n_estimators=3, bootstrap=True, random_state=0)
    clf.fit(x_train, y_train)
    y_train_predict = clf.predict(x_train)
    print(y_train_predict)
    y_dev_predict = clf.predict(x_dev)
    train_accuracy_score = accuracy_score(y_true=y_train, y_pred=y_train_predict)
    train_f1_score = f1_score(y_true=y_train, y_pred=y_train_predict, average='macro')
    dev_accuracy_score = accuracy_score(y_dev, y_dev_predict)
    dev_f1_score = f1_score(y_true=y_dev, y_pred=y_dev_predict, average='macro')
    print('Training Finished...')
    # 一般性能指标描述到小数点后4位即可
    print('Training accuracy = {:.4}, f1_score = {:.4}'.format(train_accuracy_score, train_f1_score))
    print('Dev accuracy = {:.4}, f1_score = {:.4}'.format(dev_accuracy_score, dev_f1_score))

    with open(model_path, 'wb') as fa:
        pickle.dump(clf, fa)
    print('Model is saved at {}'.format(os.path.join(os.getcwd(), model_path)))
    return scaler


def test(file_path, model_path, scaler):
    print('Start evaluate...')
    trace_data, labels = extract_intention_data(file_path)
    sample_num, seq_length, feature_dim = trace_data.shape[0], trace_data.shape[1], trace_data.shape[2]
    le = LabelEncoder()
    le.fit(intention_type)
    trace_labels = le.transform(labels)
    # todo 测试的时候是否需要shuffle?
    trace_data = trace_data.reshape(sample_num * seq_length, -1)
    trace_data = scaler.transform(trace_data)
    trace_data = trace_data.reshape(sample_num, -1)
    with open(model_path, 'rb') as fa:
        clf = pickle.load(fa)
    trace_pred = clf.predict(trace_data)
    test_accuracy_score = accuracy_score(y_true=trace_labels, y_pred=trace_pred)
    test_f1_score = f1_score(y_true=trace_labels, y_pred=trace_pred, average='macro')
    # todo 验证集和测试集有什么不同？一般来说，哪个集合上的效果会理想一些？到底应该按哪个指标去选取我们的模型？
    print('Test accuracy = {:.4}, f1_score = {:.4}'.format(test_accuracy_score, test_f1_score))
    # todo 如若效果不理想，有什么改进的思路吗？是否可以分析一下错误的样本，挖掘哪些样本容易被分类器分错？


def plot_3d_scatter(arr, label, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = []
    y = []
    z = []

    for i in range(len(arr)):
        x.append(arr[i][0])
        y.append(arr[i][1])
        z.append(arr[i][2])

    ax.scatter(x, y, z)
    plt.title(label)
    #plt.show()
    plt.savefig(t + '.png')

if __name__ == '__main__':
    train_file_path = 'dataset/BekaaValley_train.csv'
    model_path = 'clf_model.pkl'
    train_scaler = train(train_file_path, model_path)
    test_file_path = 'dataset/BekaaValley_test.csv'
    best_model_path = 'clf_model.pkl'
    test(test_file_path, best_model_path, train_scaler)