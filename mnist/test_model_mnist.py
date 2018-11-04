import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
import numpy as np
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
from chainer import serializers
#Linksはパラメータありの関数、Functionはなし

from chainer import iterators
from chainer.datasets import mnist,split_dataset_random

def load_data():
    train_val,test = mnist.get_mnist(withlabel = True,ndim = 1)
    #print(train_val[0][0].shape,test.shape[0][0].shape)

    train,valid = split_dataset_random(train_val,50000,seed = 0)

    batchsize = 128
    train_iter = iterators.SerialIterator(train,batchsize)
    valid_iter = iterators.SerialIterator(valid,batchsize,repeat=False,shuffle=False)
    test_iter = iterators.SerialIterator(test,batchsize,repeat=False,shuffle=False)

    return train,valid,train_iter,valid_iter,test_iter
#モデル
class MLP(chainer.Chain):
    def __init__(self,n_mid_units = 100,n_out = 10):
        super(MLP,self).__init__()

        #パラメータを持つ層の登録
        with self.init_scope():
            #各層ごとの定義:L.Linearは全結合を意味する
            self.l1 = L.Linear(None,n_mid_units)
            self.l2 = L.Linear(n_mid_units,n_mid_units)
            self.l3 = L.Linear(n_mid_units,n_out)
    def forward(self,x):
        #forward計算
        h1 = F.relu(self.l1(x)) #l1層の計算(relu関数)
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def main():

    train,valid,train_iter,valid_iter,test_iter=load_data()

    gpu_id = 0 #cpuの場合は-1

    net = MLP()

    if gpu_id >= 0:
        net.to_gpu(gpu_id)

    optimizer = optimizers.SGD(lr=0.01).setup(net)

    max_epoch = 50

    while train_iter.epoch < max_epoch: #train_iterで取得したtrainデータセット(128で1epoch)を100epoch繰り返す。


        #-------学習の1イテレーション---------
        train_batch=train_iter.next() #train_batchに新たなの128枚を代入
        x,t = concat_examples(train_batch, gpu_id) #concat_examplesはタプルをデータとラベルに分解

        #予測値の計算
        y = net(x) #モデルにデータを代入

        #ロスの計算
        loss = F.softmax_cross_entropy(y,t) #モデルの出力とラベルとのlossを算出

        #勾配の計算
        net.cleargrads()
        loss.backward() #誤差逆伝播法で勾配の算出

        #パラメータの更新
        optimizer.update() #updateメソッドでbackward()で求めた勾配を更新
        #---------ここまで---------

        # 1エポック終了ごとにValidationデータに対する予測精度を測って、
        # モデルの汎化性能が向上していることをチェック

        if train_iter.is_new_epoch: #1epochが終わったら

            #ロスの表示
            print('epoch:{:02d} train_loss:{:.04f}'.format(train_iter.epoch,float(to_cpu(loss.data))),end = '')
            #format(今のepoch数,cpuで計算するloss,改行なし)
            valid_losses = []
            valid_accurcies = []

            #validation
            while True:
                valid_batch = valid_iter.next() #valid_uterで128枚の画像を取得
                x_valid,t_valid = concat_examples(valid_batch,gpu_id) #タプルであるvalid_batchをtrainデータとラベルデータに分解する。

                #varidationデータをforward
                with chainer.using_config('train',False),chainer.using_config('enable_backprop',False):#Falseは学習はしないという意味
                    y_valid = net(x_valid) #validationデータを学習させる


                #ロスを計算
                loss_valid = F.softmax_cross_entropy(y_valid,t_valid)
                valid_losses.append(to_cpu(loss_valid.array))

                #精度を計算
                accuracy = F.accuracy(y_valid,t_valid)
                accuracy.to_cpu()
                valid_accurcies.append(accuracy.array)

                if valid_iter.is_new_epoch: #1epoch終えたら
                    valid_iter.reset() #valid_iterを初期化
                    break
            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(np.mean(valid_losses),np.mean(valid_accurcies)))

    #テストデータでの評価

    test_accuracies = []
    while True:
        test_batch = test_iter.next()
        x_test,t_test = concat_examples(test_batch,gpu_id)

        #テストデータをforward
        with chainer.using_config('train',False),chainer.using_config('enable_backprop',False):#学習はしないという意味
            y_test = net(x_test)

            #精度を計算
            accuracy = F.accuracy(y_test,t_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.array)

            if test_iter.is_new_epoch:
                test_iter.reset()
                break
    print('test_accuracy:{:.04f}'.format(np.mean(test_accuracies)))
    serializers.save_npz('my_mnist.model', net)

if __name__ == '__main__':
    main()
