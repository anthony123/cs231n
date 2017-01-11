#coding=utf-8
import numpy as np

#为了提高速度，接口稍有不同
def eval_numberical_gradient(f,data, label,x):
    """
    数值型梯度的一个简单实现
    -f 函数，其输入参数的个数为1个
    -x 目标点的坐标
    """
    fx = f(data, label,x)   #计算x处的函数值

    grad = np.zeros(x.shape)
    h = 0.0001

    #迭代x中所有的索引值
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        #计算在x+h处的函数值
        ix = it.multi_index
        #print ix
        old_value = x[ix]
        x[ix] = old_value+h
        fxh = f(data, label,x)
        x[ix] = old_value

        #计算偏导数
        grad[ix] = (fxh - fx)/h
        it.iternext()

    return grad
