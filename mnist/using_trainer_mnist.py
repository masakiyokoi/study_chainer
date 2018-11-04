from test_model_mnist import MLP
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from load_data import load_data
import numpy as np
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu,to_gpu
from chainer import training
from chainer.training import extensions
import numpy as np
def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

    return reset_seed(0)
def main():

    gpu_id = 0  # CPUを使いたい場合は-1を指定してください

    
    test,train_iter,valid_iter,test_iter = load_data()

    net = MLP()
    # ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
    net = L.Classifier(net)

    # 最適化手法の選択
    optimizer = optimizers.SGD(lr=0.01).setup(net)

    # UpdaterにIteratorとOptimizerを渡す
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    max_epoch = 10

    # TrainerにUpdaterを渡す
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
    trainer.extend(extensions.ParameterStatistics(net.predictor.l1, {'std': np.std}))
    #trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
    #trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    #trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()
    test_evaluator = extensions.Evaluator(test_iter, net, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])
if __name__ == '__main__':
    main()

