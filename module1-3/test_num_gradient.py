#coding=utf-8
import numpy as np
import eval_numberical_gradient as num_grad
import loss
import load_full_CIFAR10 as load

def CIFAR_loss_fun(X_train,Y_train,W):
    lossValue = loss.L(X_train, Y_train, W);
    return lossValue

X_train, Y_train, X_test, Y_test = load.load_CIFAR10('cifar-10-batches-py/')
X_train = X_train.T;
#增加一行，处理偏置值
bias = np.ones((1, X_train.shape[1]))

X_train = np.vstack((X_train, bias))
W = np.random.randn(10, 3073)*0.001     #随机初始化权重向量
df = num_grad.eval_numberical_gradient(CIFAR_loss_fun, X_train,Y_train, W)

loss_original = CIFAR_loss_fun(X_train,Y_train, W)   #计算原始损失
#print 'compute original gradient done'
print 'original loss: %f'  %(loss_original,)

#看不同的步长产生的效果
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10**step_size_log
    W_new = W - step_size*df    #更新后的权重值
    loss_new = CIFAR_loss_fun(X_train,Y_train, W_new)
    print 'for step size %f new loss: %f' %(step_size, loss_new)
