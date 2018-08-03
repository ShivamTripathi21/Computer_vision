import numpy as np

def L(x,y,w):
    y=np.transpose(y)
    scores=np.dot(w,x)
    margins=np.maximum(0, scores - scores[y] + 1)
    margins[y]=0
    loss_i=np.sum(margins)
    return loss_i

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

x_train = np.transpose(unpickle('cifar-10/data_batch_1')['data'])
y_train = [unpickle('cifar-10/data_batch_1')['labels']]

bestloss=float('inf')

for num in range(2):
    W = np.random.randn(10, 3072) * 0.0001
    
    loss=L(x_train,y_train,W)
    if loss < bestloss :
        bestloss = loss
        bestw = W
    
    print 'in attempt %d the loss was %f , best %f' % (num,loss,bestloss)


