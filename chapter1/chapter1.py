#例1.1
import numpy as np
#scipy数学计算包
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

#目标函数
def real_fun(x):
    return np.sin(2*np.pi*x)

#多项式
def ploy_fun(p,x):
    #ploy1d[1,2,3] 1x^2+2x^1+3
    f = np.poly1d(p)
    return f(x)

#误差函数
def residuals_fun(p,x,y):
    ret = ploy_fun(p,x) - y
    return ret


x = np.linspace(0,1,10)
y_ = real_fun(x)
#normal 均值0 标准差0.1
y = [np.random.normal(0,0.1)+y1 for y1 in y_]

def fitting(N=0):
    #随机产生初始值 rand(n) 产生n个服从0~1均匀分布的随机值
    p_init = np.random.rand(N+1)
    #最小二乘法  (误差函数，参数，数据点)
    lsq = leastsq(residuals_fun,p_init,args=(x,y))
    print("best parameters",lsq[0])
    x_points = np.linspace(0,1,1000)
    plt.plot(x_points,real_fun(x_points),label = 'real')
    plt.plot(x_points,ploy_fun(lsq[0],x_points),label = 'fit curve')
    plt.plot(x,y,'bo',label='noise')
    #给图加上图例
    plt.legend()
    plt.show()

#三次函数拟合
fitting(3)