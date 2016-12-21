#coding=utf-8
import numpy as np
import load_full_CIFAR10 as load
import loss


trainData, trainLabel, testData, testLabel = load.load_CIFAR10('cifar-10-batches-py/')


trainData = np.transpose(trainData)

#增加一行，处理偏置值
bias = np.ones((1, trainData.shape[1]))

trainData = np.vstack((trainData, bias))
#print trainData.shape,trainLabel.shape

#trainData: (3073, 45000), trainLabel: (45000,1)

#产生一个随机的权重W
W = np.random.randn(10,3073)*0.0001

#计算损失值
l = loss.L(trainData, trainLabel, W)
print l
