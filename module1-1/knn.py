import numpy as np
import load_full_CIFAR10 as load
import KNearestNeighbor as KNN

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

Xval_rows = Xtr_rows[:1000, :]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:,:]
Ytr = Ytr[1000:]

validation_accuracies = []


#create a Nearest Neighbor classifier class
knn = KNN.KNearestNeighbor()

#train the classifier on the training images
knn.train(Xtr_rows, Ytr)

#k numbers
knums = np.array([1,3,5,10,20,50,100])

#predict labels on the test images
Yte_predict = knn.predict(Xval_rows, knums)

#print the result
for k in xrange(knums.shape[0]):
    print 'k = %d' %(knums[k])
    acc = np.mean(Yte_predict[k] == Yval)
    validation_accuracies.append(acc)
    print 'accuracy: %f' %(acc)

#find the best k
max_index = np.argmax(validation_accuracies)
best_k = knums[max_index]

#get the final result
Yte_final_predict = knn.predict(Xte_rows, np.array[best_k])
acc = np.mean(Yte_final_predict == Yte)
print 'acc = %f' %(acc)
