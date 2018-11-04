import chainer
import chainer.links as L
import chainer.functions as F
import chainer
#Linksはパラメータありの関数、Functionはなし
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

gpu_id = 0 #cpuの場合は-1

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)
