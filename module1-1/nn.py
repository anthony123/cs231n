import load_full_CIFAR10 as load
import NearestNeighbor as NN
import numpy as np

#Xtr: training data
#Ytr: training labels
#Xte: test data
#Yte: test label

Xtr, Ytr, Xte, Yte = load.load_CIFAR10('cifar-10-batches-py/')

#flatten out all images to be one-dimensional
#training data
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)

#test data
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

#create a Nearest Neighbor classifier class
nn = NN.NearestNeighbor()

#train the classifier on the training images
nn.train(Xtr_rows, Ytr)

#predict labels on the test images
Yte_predict = nn.predict(Xte_rows)

#print the result
print 'accuracy: %f' %(np.mean(Yte_predict == Yte))
