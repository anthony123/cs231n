import numpy as np

def L_i(x,y,W):
    #x: [3073 x 1]
    #y: an integer indicates the index of correct class
    #W: [10 x 3073]
    delta = 1.0
    scores = W.dot(x)
    correct_class_score = scores[y]
    D = W.shape[0]
    loss_i = 0.0

    for j in xrange(D):
        if j == y:
            continue

        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def L_i_vectorized(x,y,W):
    #half-vectorized implementation
    delta = 1.0
    scores = W.dot(x)


    margins = np.maximum(0, scores - scores[y] + delta)

    margins[y] = 0
    loss_i = np.sum(margins)

    return loss_i

def L(X,y,W):
    #X: all training examples [3073x45000]
    #y: [50000x1]
    #W: [10x3073]
    delta = 1.0
    scores = W.dot(X) #scores: 10 x 45000

    margins = np.maximum(0, scores - scores[y, np.arange(scores.shape[1])] + delta)
    margins[y,np.arange(y.shape[0]) ] = 0;

    loss = np.sum(margins)

    return loss
