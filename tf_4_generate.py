#coding:utf-8
import numpy as np

#生成输入数据与标准值
def generate(size, seed = 1):
    rdm = np.random.RandomState(seed)
    X = rdm.randn(size, 2)

    Y_ = [[int(x0 * x0 + x1 * x1 < 2)] for (x0, x1) in X]
    Y_c = [['red' if y[0] else 'blue'] for y in Y_] 

    return X, Y_, Y_c


