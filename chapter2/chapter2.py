#pandas 数据分析包
import numpy as np
import pandas as pd
#sklearn提供了机器学习算法和数据集
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#导入数据集
iris = load_iris()
#pandas数据结构.二维表
#dataframe(数据，行名，列名)
df = pd.DataFrame(iris.data,columns=iris.feature_names)
#target目标值
df['label'] = iris.target
#每一列的标签，可以打印df查看
df.columns = ['sepal length','sepal width','petal length','petal width','label']
#切片
data = np.array(df.iloc[:100,[0,1,-1]])
#:-1 不包含-1  ;,-1所有的行，最后一列
x,y = data[:,:-1],data[:,-1]
#y输出空间 {+1，-1}正样本和负样本
y = np.array([1 if i==1 else -1 for i in y])

class Model:
    def __init__(self):
        #生成len(data[0])-1个一维向量1
        #权重w长度跟x一致
        self.w = np.ones(len(x[0]),dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self,x,w,b):
        #处理一维数组返回的是内积
        y = np.dot(x,w) + b
        if y>=0:
            y=1
        else:
            y=-1
        return y

    def fit(self,x_train,y_train):
        is_wrong = False
        while is_wrong == False:
            wrong_cnt = 0
            for d in range(len(x_train)):
                x = x_train[d]
                y = y_train[d]
                #梯度下降
                y_predict = self.sign(x,self.w,self.b)
                if y*y_predict <= 0:
                    self.w = self.w + self.l_rate*np.dot(y,x)
                    self.b = self.b + self.l_rate*y
                    wrong_cnt += 1
            if wrong_cnt == 0:
                is_wrong = True

#首值，尾值，数量
x_points = np.linspace(4,7,10)
perceptron = Model()
perceptron.fit(x, y)
#0 = w[0]x[0]+w[1]x[1]+b
y_fit =  -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'],label='0')
plt.scatter(df[100:150]['sepal length'],df[100:150]['sepal width'],label='2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.plot(x_points, y_fit,color='r')
plt.show()