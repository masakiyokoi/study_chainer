from chainer import iterators
from chainer.datasets import mnist,split_dataset_random

def load_data():
    train_val,test = mnist.get_mnist(withlabel = True,ndim = 1)
    print(train_val[0][0].shape,test.shape[0][0].shape)

    train,valid = split_dataset_random(train_val,50000,seed = 0)

    batchsize = 128
    train_iter = iterators.SerialIterator(train,batchsize)
    valid_iter = iterators.SerialIterator(valid,batchsize,repeat=False,shuffle=False)
    test_iter = iterators.SerialIterator(test,batchsize,repeat=Fale,shuffle=False)

    return train_iter,valid_iter,test_iter
