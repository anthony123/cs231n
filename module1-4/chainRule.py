#coding=utf-8

#f(x,y,z) = (x+y)*z

#设置输入值
x = -2; y = 5; z = -4

#计算前向过程
q = x+y
f = q*z

#逆序求出反向过程，也就是先求出 f=q*z
dfdz = q
dfdq = z

dqdx = 1
dqdy = 1

#现在通过q=x+y反向传播
dfdx = dfdq*dqdx
dfdy = dfdq*dqdy

grad = [dfdx, dfdy, dfdz]
print grad
