import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, knums):
        num_test = X.shape[0]
        Ypred = np.zeros([knums.shape[0], num_test], dtype = type(self.ytr))

        for k in xrange(knums.shape[0]):
            print 'k = %d' %(knums[k])
            for i in xrange(num_test):
                distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
                temp = np.argpartition(distances,knums[k])
                min_indices = temp[:knums[k]]

                #record the min nums
                temp = np.partition(distances, knums[k])
                min_nums = temp[:knums[k]]

                #find the most frequent element in the min_indices
                min_nums = min_nums.astype(np.int32)
                counts = np.bincount(min_nums)
                num = np.argmax(counts)
                bool_indices = (min_nums == num)
                result_indices = min_indices[bool_indices]
                Ypred[k][i] = result_indices[0]
        return Ypred
