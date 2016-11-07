import load_CIFAR10 as load
import numpy as np

def load_CIFAR10(dir):
    path1 = dir + 'data_batch_1'
    path2 = dir + 'data_batch_2'
    path3 = dir + 'data_batch_3'
    path4 = dir + 'data_batch_4'
    path5 = dir + 'data_batch_5'
    
    Xtr1, Ytr1, Xte1, Yte1 = load.load_CIFAR10(path1)
    Xtr2, Ytr2, Xte2, Yte2 = load.load_CIFAR10(path2)
    Xtr3, Ytr3, Xte3, Yte3 = load.load_CIFAR10(path3)
    Xtr4, Ytr4, Xte4, Yte4 = load.load_CIFAR10(path4)
    Xtr5, Ytr5, Xte5, Yte5 = load.load_CIFAR10(path5)

    trainData = np.concatenate((Xtr1,Xtr2,Xtr3,Xtr4,Xtr5))
    trainLabels = np.concatenate((Ytr1,Ytr2,Ytr3,Ytr4,Ytr5))
    testData = np.concatenate((Xte1,Xte2,Xte3,Xte4,Xte5))
    testLabels = np.concatenate((Yte1,Yte2,Yte3,Yte4,Yte5))

    return trainData, trainLabels, testData, testLabels
