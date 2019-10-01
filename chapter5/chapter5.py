

#5.2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import log

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
                ]   
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets,columns=labels)
#计算经验熵
def cal_em_entroy(datasets):
    #样本数量
    data_length = len(datasets)
    #每一类样本所对应的数量
    feature_sets = {}
    for i in range(data_length):
        if datasets[i][-1] not in feature_sets:
            feature_sets[datasets[i][-1]] = 1
        else:
            feature_sets[datasets[i][-1]] +=1
    #经验熵
    em = 0
    for i,v in feature_sets.items():
        em -= v/data_length*math.log(v/data_length,2)

    return em

print(cal_em_entroy(datasets))
#计算经验条件熵
# def cal_em_con_entroy(feature_name,num):
#     print(feature_name)
#     length = len(datasets)
#     feature_len = 0
#     for i in datasets:
#         if i[num] == feature_name:
#             feature_len +=1
#     feature_sets = {}
#     for i in range(length):
#         if datasets[i][num] == feature_name:
#             if datasets[i][-1] not in feature_sets:
#                 feature_sets[datasets[i][-1]] = 1
#             else:
#                 feature_sets[datasets[i][-1]] +=1
#     #条件经验熵
#     em = 0
#     for i,v in feature_sets.items():
#         em -= v/feature_len*math.log(v/feature_len,2)*feature_len/length
#     print(em,feature_sets)
#     return em

#feature name 特征名字
#num 处于第几列
def cal_em_con_entroy(feature_name,num):
    data_length = len(datasets)
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    list1 = list(set(train_data[feature_name]))

    em = 0
    #每个标签下具体的类
    for i in list1:
        feature_sets = {}
        feature_len = 0
        for j in range(data_length):
            if datasets[j][num] == i:
                feature_len+=1
                if datasets[j][-1] not in feature_sets:
                    feature_sets[datasets[j][-1]] = 1
                else:
                    feature_sets[datasets[j][-1]] +=1
        # print(feature_sets)
        for zi,v in feature_sets.items():
            # print(v,feature_len)
            em -= v/feature_len*math.log(v/feature_len,2)*feature_len/data_length
    return em

#计算增益
def cal_info_gain():
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    info_gain_list = []
    for i in range(len(labels)-1):
        info_gain = 0
        info_gain += cal_em_entroy(datasets) - cal_em_con_entroy(labels[i],i)
        info_gain_list.append(info_gain)
    print(info_gain_list)
    max_index = info_gain_list.index(max(info_gain_list))
    print('特征({}) - info_gain - {:.3f}'.format(labels[max_index], info_gain_list[max_index]))

#5.3  
#这个有点迷，需要再看一下啊
#定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        #是否是根节点
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        #格式化字符串
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)
    
class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    #静态方法 -- 只是名义上归属类管理，但是不能使用类变量和实例变量
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        #对字典的数据进行累加
        ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        #每次只传入某一特征的数据
        cond_ent = sum([(len(p)/data_length)*self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        print("best",best_)
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        #y_train 分类结果
        #feature 特征集
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]

    
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        #pandas valuecounts 查看表格中某列有多少个不同的取值
        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        #返回最大信息增益的特征序列号，和对应的值
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        #value_counts 返回类别及其对应的数量
        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)


if __name__ == "__main__":
    # cal_info_gain()
    datasets, labels = create_data()
    data_df = pd.DataFrame(datasets, columns=labels)
    dt = DTree()
    tree = dt.fit(data_df)
    print(tree)
    print(dt.predict(['老年', '否', '否', '好']))

