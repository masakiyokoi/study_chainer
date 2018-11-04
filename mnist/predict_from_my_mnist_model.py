from test_model_mnist import MLP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import chainer
from chainer import serializers
from load_data import load_data
import numpy as np
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
def main():
    # まず同じネットワークのオブジェクトを作る
    infer_net = MLP()

    # そのオブジェクトに保存済みパラメータをロードする
    serializers.load_npz('my_mnist.model', infer_net)

    gpu_id = 0  # CPUで計算をしたい場合は、-1を指定してください

    if gpu_id >= 0:
        infer_net.to_gpu(gpu_id)
    test,train_iter,valid_iter,test_iter = load_data()
    # 1つ目のテストデータを取り出します
    x, t = test[0]  #  tは使わない

    # どんな画像か表示してみます
    #plt.imshow(x.reshape(28, 28), cmap='gray')
    #plt.show()

    # ミニバッチの形にする（複数の画像をまとめて推論に使いたい場合は、サイズnのミニバッチにしてまとめればよい）
    print('元の形：', x.shape, end=' -> ')

    x = x[None, ...]

    print('ミニバッチの形にしたあと：', x.shape)

    # ネットワークと同じデバイス上にデータを送る
    x = infer_net.xp.asarray(x)

    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)

    # Variable形式で出てくるので中身を取り出す
    y = y.array

    # 結果をCPUに送る
    y = to_cpu(y)

    # 予測確率の最大値のインデックスを見る
    pred_label = y.argmax(axis=1)

    print('ネットワークの予測:', pred_label[0])

if __name__ == '__main__':
    main()
