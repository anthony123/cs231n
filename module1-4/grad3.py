#coding=utf-8
import math

#计算函数f(x,y) = (x+σ(y))/(σ(x) + （x+y)^2) 的梯度

#example values
x = 3
y = -4

#forward pass
sigy = 1.0 / (1+math.exp(-y))   #分子中的sigmoid函数  (1)
num = x + sigy      #整个分子部分                     (2)

sigx = 1.0 / (1+math.exp(-x)) #分母中的sigmoid函数    (3)
xpy = x + y                 #                        (4)
xpysqr = xpy ** 2           #                        (5)
den = sigx + xpysqr #分母                            (6)

invden = 1.0/den    #                                (7)
f = num*invden      #                                (8)


#反向传播

#反向传播 f = num*invden
dnum = invden                                       #(8)
dinvden = num                                       #(8)

#反向传播 invden=1.0/den
dden = (-1.0/(den**2))*dinvden                      #(7)


#反向传播 den = sigx + xpysqr
dsigx = dden                                        #(6)
dxpysqr = dden                                      #(6)

#反向传播 xpysqr = xpy**2
dxpy = 2*xpy*dxpysqr                                #(5)

#反向传播 xpy = x + y
dx = dxpy                                           #(4)
dy = dxpy                                           #(4)

#反向传播 sigx = 1.0 / (1+math.exp(-x))
dx += ((1-sigx)*sigx)*dsigx     #主要是 “+=”        #(3)

#反向传播 num = x + sigy
dx += dnum                                         #(2)
dsigy = dnum                                       #(2)

#反向传播 sigy = 1.0 / (1+math.exp(-y))
dy += (dsigy) * ((1-sigy)*sigy)                    #(1)

grad = [dx, dy]
print grad
