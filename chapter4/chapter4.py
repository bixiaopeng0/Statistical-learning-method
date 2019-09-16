import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
#Counter计算值出现的次数
#defaultdict提供了默认值功能，是dict的plus版本
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
def create_data():
    """
    测试集x特征应该和样本中的特征一致，因为iris里特征为小数，所以这里不采用了
    """
    # iris = load_iris()
    # df = pd.DataFrame(iris.data,columns=iris.feature_names)
    # df['label'] = iris.target
    # df.columns = ['sepal length','sepal width','petal length','petal width','label']
    # data = np.array(df.iloc[:100,[0,1,-1]])
    data = np.array([[1, 0, -1], [1, 1, -1], [1, 1, 1], [1, 0, 1],
                    [1, 0, -1], [2, 0, -1], [2, 1, -1], [2, 1, 1],
                    [2, 2, 1], [2, 2, 1], [3, 2, 1], [3, 1, 1],
                    [3, 1, 1], [3, 2, 1], [3, 2, -1]])
    true_data_1 = []
    true_data_2 = []
    false_data_1 = []
    false_data_2 = []
    true_data,false_data = [],[]
    for i in data:
        if i[2] == -1:
            true_data_1.append(i[0])
            true_data_2.append(i[1])
        else:
            false_data_1.append(i[0])
            false_data_2.append(i[1])
    true_data.append( true_data_1)
    true_data.append(true_data_2)
    false_data.append(false_data_1)
    false_data.append(false_data_2)
    print(true_data,false_data)
    return data[:,:-1],data[:,-1],true_data,false_data

class NativeBayes:
    def __init__(self,wg=1):
        self.wg = wg
        self.p_prior = {} #先验概率
        self.p_condition = {} #条件概率

    def fit(self,x_data,y_data):
        #所有的实例数量
        n = y_data.shape[0]
        #每一类实例的数量
        c_y = Counter(y_data)
        k = len(c_y)
        #遍历字典
        for key,val in c_y.items():
            #先验概率的贝叶斯估计
            self.p_prior[key] = (val+self.wg)/(n+k*self.wg)
        for d in range(x_data.shape[1]):
            #初始化为0
            d_dict = defaultdict(int)
            vector = x_data[:,d]
            k1 = len(Counter(vector))
            for xd,y in zip(vector,y_data):
                d_dict[(xd,y)] += 1
            for key,val in d_dict.items():
                #(d,key[0],key[1]) 键值
                #条件概率的贝叶斯估计
                self.p_condition[(d,key[0],key[1])] = (val+self.wg)/(c_y[key[1]]+k1*self.wg)
    #确定实例x的类
    def predict(self,x):
        p_post = defaultdict()
        for y,py in self.p_prior.items():
            p_joint = py
            for d,xd in enumerate(x):
                p_joint *= self.p_condition[(d,xd,y)]
            p_post[y] = p_joint
        #.get返回相应的键
        return max(p_post,key = p_post.get)

                        


if __name__ == "__main__":
    x_data,y_data,true_data,false_data = create_data()
    plt.scatter(true_data[0],true_data[1],label='feature0')
    plt.scatter(false_data[0],false_data[1],label='feature1')
    plt.scatter(1, 2, color='r', label='test')
    plt.xlabel('feature0')
    plt.ylabel('feature1')
    plt.legend()

    model = NativeBayes(1)
    model.fit(x_data,y_data)
    print(model.p_prior,'\n',model.p_condition)
    print(model.predict(np.array([1,2])))
    plt.show()