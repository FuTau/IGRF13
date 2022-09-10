import numpy as np

a = 6371.2    # 地球半径
ntheta = 100
nphi = 200
N1 = 195
N2 = 26
t = 180 / np.pi

theta = np.linspace(0, np.pi, ntheta) 
#将纬度180度划分为100等份
phi = np.linspace(-np.pi, np.pi, nphi)
#将精度360度划分为200等份
[Phi, Theta] = np.meshgrid(phi, theta)
#phi为经度数组，theta为纬度数组，meshgrid将其组成了一个矩阵
shapex = list(Phi.shape)
#list()将任何可迭代数据转换为列表类型，并返回转换后的列表
#print(shapex)
#print(Phi.shape)
#运行结果：
#[100, 200]        列表
#(100, 200)        元组
shapex.append(N2)
#print(shapex)
#运行结果：
#[100, 200, 26]

def factorial(N):
    # 阶乘函数

    a = 1
    if N == 0:
        a == 1
    else:
        for i in range(1, N+1):
            a = a * i
    return a
