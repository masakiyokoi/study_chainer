
from predict_from_my_mnist_model import MLP

import random
import numpy
import chainer
import sys
from chainer import serializers
from load_data import load_data
from chainer.cuda import to_cpu

def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def main():
    

    reset_seed(0)
    
    test,train_iter,valid_iter,test_iter=load_data()
    infer_net = MLP()
    serializers.load_npz('mnist_result/snapshot_epoch-10',infer_net, path='updater/model:main/predictor/')

    gpu_id = 0
    if gpu_id >= 0:
        infer_net.to_gpu(gpu_id)

    x, t = test[0]
    #plt.imshow(x.reshape(28, 28), cmap='gray')
    #plt.show()

    x = infer_net.xp.asarray(x[None, ...])
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)
    y = to_cpu(y.array)

    print('予測ラベル:', y.argmax(axis=1)[0])

if __name__ == '__main__':
    main()
