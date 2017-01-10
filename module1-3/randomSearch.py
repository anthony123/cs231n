#coding=utf-8
import numpy as np
import loss
import load_full_CIFAR10 as load


X_train, Y_train, X_test, Y_test = load.load_CIFAR10('cifar-10-batches-py/')

X_train = X_train.T;
#增加一行，处理偏置值
bias = np.ones((1, X_train.shape[1]))

X_train = np.vstack((X_train, bias))


bestloss = float('inf') #初始化为最大值

for num in xrange(1000):
    W = np.random.randn(10, 3073)*0.000001
    lossValue = loss.L(X_train, Y_train, W);
    print bestloss
    if lossValue < bestloss:
        bestloss = lossValue;
        bestW = W

    #print 'in attempt %d the loss was %f, best %f' %(num, lossValue, bestloss)
