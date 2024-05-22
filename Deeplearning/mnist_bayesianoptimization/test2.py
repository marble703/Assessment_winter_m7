# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:16:38 2024

@author: admin
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

init_points = 5 #初始探索点数量，建议5
n_iter = 20 #迭代次数，建议20
total_epoch = init_points + n_iter #总超参数寻找轮数

pbounds = { 'learning_rate_log': (-5, -1),  
            'beta1': (0.9, 0.999), #控制一阶矩（梯度的指数加权平均）的衰减速度,其实我也不知道这是啥
            'beta2': (0.9, 0.999), #控制二阶矩（梯度平方的指数加权平均）的衰减速度,但是它总有存在的意义
            'weight_decay_log': (-5,-3) } #正则化项


def t_t(learning_rate_log, beta1, beta2, weight_decay_log):
    X = torch.rand(100, 100, dtype=torch.float64)
    Y = torch.rand(100, 100, dtype=torch.float64)
    
    target = beta1 * beta2
    return target

bayesian_optimizer = BayesianOptimization(
    f = t_t,
    pbounds = pbounds,
    verbose = 2) # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent

bayesian_optimizer.maximize(
    init_points = init_points,
    n_iter = n_iter)

print("""\nbest params combo:
         learning_rate: {};
         beta1:{};
         beta2:{};
         weight_decay:{};\n
         accuracy:{}""".format(10 ** bayesian_optimizer.max["params"]["learning_rate_log"],
                               bayesian_optimizer.max["params"]["beta1"],
                               bayesian_optimizer.max["params"]["beta2"],
                               10 ** bayesian_optimizer.max["params"]["weight_decay_log"],
                               bayesian_optimizer.max["target"]))