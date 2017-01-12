#coding=utf-8
import math

#计算函数 f(w,x) = 1/(1+exp(-(w0x0+w1x1+w2)))

 #随机的梯度和数据
w = [2,-3,-3]
x = [-1,-2]

#前向传播
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0/(1+math.exp(-dot))

ddot = (1-f)*f      #直接使用公式计算梯度
dx = [w[0]*ddot, w[1]*ddot]     #反向传播到x
dw = [x[0]*ddot, x[1]*ddot, 1.0*ddot]   #反向传播到w

print dx, dw
