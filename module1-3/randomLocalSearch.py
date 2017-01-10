#coding=utf-8
import numpy as np
import loss
import load_full_CIFAR10 as load

X_train, Y_train, X_test, Y_test = load.load_CIFAR10('cifar-10-batches-py/')
X_train = X_train.T

bias = np.ones((1,X_train.shape[1]))

X_train = np.vstack((X_train, bias))

W = np.random.randn(10,3073)*0.001      #随机生成一个W
bestloss = float("inf")

for i in xrange(1000):
    step_size = 0.0001
    Wtry = W + np.random.randn(10,3073)*step_size
    l = loss.L(X_train, Y_train, Wtry)
    if l < bestloss:
        W = Wtry
        bestloss = l

    print 'iter %d loss is %f' %(i, bestloss)
