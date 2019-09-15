
import math
#eg3.1

def get_dis(p,x1,x2):
    dis = 0
    for i in range(len(x1)):
        #math.pow(底数，幂)
        dis += math.pow(abs(x1[i]-x2[i]),p)
    dis = math.pow(dis,1/p)
    return dis

def run_example_31():
    x1 = [1,1]
    x2 = [5,1]
    x3 = [4,4]
    #p 1~9
    for p in range(1,10):
        x12_dis = get_dis(p,x1,x2)
        x13_dis = get_dis(p,x1,x3)
        if x12_dis > x13_dis:
            print(p,"x13",x13_dis)
        else:
            print(p,"x12",x12_dis)




#KNN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
#新加一列
df['label'] = iris.target
df.columns = ['sepal length','sepal width','petal length', 'petal width','label']
# plt.scatter(x,y,...)
plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'],label='0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'],label='1')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

#0~99行 sepal length sepal width label列
data = np.array(df.iloc[:100,[0,1,-1]])
x,y = data[:,:-1],data[:,-1]
#将数据划分为训练集和测试集，test_size = 0.2测试集占20%
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

class KNN:
    def __init__(self,x,y,test,k,p):
        self.test = test
        self.k = k
        self.p = p
        self.dis = []
        self.x = x
        self.y = y
    def predict(self):
        for i in range(len(self.x)):
            self.dis.append(get_dis(self.p,self.x[i],self.test))
        label_dis  = list(zip(y,self.dis))
        k_dis = label_dis[:self.k]
        for i in label_dis:
            for num,j in enumerate(k_dis):
                if i[1] < j[1]:
                    k_dis[num] = i
                    break
        return k_dis

if __name__ == '__main__':
    # run_example_31()
    test = [6,3]
    plt.scatter(test[0], test[1], color='r', label='test')
    pre_dict = KNN(x,y,test,10,2)
    l = pre_dict.predict()
    zero_class = 0
    for i in l:
        if i[0] == 0:
            zero_class += 1
    if zero_class > 4:
        print("predict 0")
    else:
        print("predict 1")
    plt.show()


