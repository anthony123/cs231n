import numpy as np
import load_CIFAR10 as load
import KNearestNeighbor as KNN

#Xtr: training data
#Ytr: training labels
#Xte: test data
#Yte: test label

Xtr, Ytr, Xte, Yte = load.load_CIFAR10('data_batch_1')

#flatten out all images to be one-dimensional
#training data
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)

#test data
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

#create a Nearest Neighbor classifier class
knn = KNN.KNearestNeighbor()

#train the classifier on the training images
knn.train(Xtr_rows, Ytr)

#k numbers
knums = np.array([1,3,5,10,20,50,100])

#predict labels on the test images
Yte_predict = knn.predict(Xte_rows, knums)

#print the result
for k in xrange(knums.shape[0]):
    print 'k = %d' %(knums[k])
    print 'accuracy: %f' %(np.mean(Yte_predict[k] == Yte))
