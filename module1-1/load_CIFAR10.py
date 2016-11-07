
import cPickle
def load_CIFAR10(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()

  trainData = dict['data'][1000:, :]
  trainLabels = dict['labels'][1000:]

  testData = dict['data'][:1000,:]
  testLabels = dict['labels'][:1000]


  return trainData, trainLabels, testData, testLabels
