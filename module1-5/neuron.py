#coding=utf-8

class Neuron(object):

    def forward(inputs):
        #假设输入和权重都是一维数组，偏置值是一个数字
        cell_body_sum = np.sum(inputs*self.weights) + self.bias

        #sigmoid 激活函数
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))

        return firing_rate
