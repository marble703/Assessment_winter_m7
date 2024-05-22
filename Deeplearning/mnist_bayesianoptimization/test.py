# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:11:21 2024

@author: admin
"""

# 导入所需库
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

# 定义一个黑盒函数进行优化
def f(x):
    return -np.sin(3*x) - x**2 + 0.7*x

# 定义输入域的边界
bounds = np.array([[-1.0, 2.0]])

# 定义高斯过程的核函数
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

# 创建高斯过程回归模型
gp = GaussianProcessRegressor(kernel=kernel)

# 定义采集函数
def acquisition(x):
    # 预测目标函数的均值和标准差
    mu, sigma = gp.predict(x, return_std=True)
    # 计算期望改进
    ei = mu - np.max(mu) + sigma
    # 返回负的期望改进
    return -ei

# 初始化输入点和输出值
X = np.array([[-0.9], [1.1]])
Y = f(X)

# 迭代5步
for i in range(5):
    # 将高斯过程拟合到观察数据
    gp.fit(X, Y)
    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, ))
    # 使用采集函数找到下一个最佳采样点
    x_next = minimize(acquisition, x0 = x0.reshape(1, -1)).x
    # 在下一个点处评估目标函数
    y_next = f(x_next)
    # 更新输入点和输出值
    X = np.append(X, x_next, axis=0)
    Y = np.append(Y, y_next, axis=0)
    # 打印当前迭代和迄今为止找到的最佳值
    print("迭代次数:", i+1, "最佳值:", np.max(Y))
