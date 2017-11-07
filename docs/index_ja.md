# WebDNNの基本的な使い方の演習
この記事は、Webブラウザ上でDeep Neural Networkを実行できるフレームワークWebDNNの基本的な使い方を演習を通して解説するものです。

演習を最後まで進めると、次のような画面が得られます。画面上の黒い領域にマウスで数字を描くと、それがどの数字なのかを認識して表示します。

![Screenshot of complete application]({{ site.baseurl }}/images/complete_screenshot.png)

動作するデモが[ここ](https://milhidaka.github.io/webdnn-exercise/answer/)から見られます。

WebDNNのリポジトリ: [https://github.com/mil-tokyo/webdnn](https://github.com/mil-tokyo/webdnn)

## 対象読者
この演習では、深層学習(Deep Learning)自体の説明は行いません。Pythonの基本的な文法およびKerasまたはChainerで簡単な画像識別モデルを学習させる方法についての知識があることを仮定しています。JavaScriptについても知識があると望ましいですが、なくてもかまいません。

## 演習の流れ
以下の流れで演習が進みます。
1. 環境構築
2. WebDNNのインストール
3. DNNモデルの学習
4. DNNモデルの変換
5. HTML/JavaScriptの実装
6. 動作確認

いくつかのコードでは、WebDNNに関する処理が穴埋め問題になっており、テキストエディタで編集しながら進めます。

Python 3.6がインストール可能なGUI環境であれば、ほとんどの環境で演習を行えます。深層学習フレームワークは、Keras・Chainerに対応します。NVIDIA GPUは不要です。

# 環境構築
`webdnn-exercise`リポジトリをまだダウンロードしていない場合は、次のコマンドで行います。

```
git clone https://github.com/milhidaka/webdnn-exercise
```

以下、カレントディレクトリが`webdnn-exercise`になっているものとします。

Python環境は、Python 3.6, numpy, Keras 2.0 or Chainer 2.0が使えればOKです。Python 3.5以下では文法エラーとなり動作しませんので注意してください。

anacondaで仮想環境を構築する場合は、以下のコマンドで行えます。

```
conda env create -f environment.yml
```

この場合、名前`webdnn`で仮想環境が作られるので、

```
source activate webdnn
```

で仮想環境に入れます。

JavaScriptについては、モダンなWebブラウザのみ必要です。Firefox, Chrome, Edgeで動作確認しています。Safariについては注意事項があるので、「Safariについて」の章を読んでください。
node.jsは不要です。

# WebDNNのインストール
WebDNNをインストールします。この中には、モデルの変換に利用するPython製のプログラムと、Webブラウザ上でモデルを実行する際のJavaScript製のライブラリが入っています。

```
git clone https://github.com/mil-tokyo/webdnn
cd webdnn
python setup.py install
cd ..
```

(執筆時点commit 519bf7c)

# DNNモデルの学習
これ以降、`exercise`ディレクトリで作業します。`answer`ディレクトリに同じファイル構成で答えと計算済み出力ファイルが入っていますので、うまくいかない場合は参照してください。

```
cd exercise
```

まず、Webブラウザ上で動かしたいモデルを学習します。この演習では、MNISTデータセットを用いた文字認識Convolutional Neural NetworkをKerasまたはChainerを用いて学習します。MNISTデータセットは0から9の手書き数字を識別するためのデータセットです。

## Kerasを利用する場合
次のコマンドでモデルを学習します。この時点では、WebDNNに依存する処理はまだ含まれていません。

```
python train_model_keras.py
```

モデルが学習され、`keras_model.h5`ファイルが生成されます。

学習されるモデルの入力は、28px * 28px、1チャンネルの画像です。値域は0から1(0から255ではない)で、前景が1で背景が0です。出力は、各クラス（数字）に対応するsoftmax確率を表す10次元のベクトルです。

## Chainerを利用する場合
次のコマンドでモデルを学習します。この時点では、WebDNNに依存する処理はまだ含まれていません。

```
python train_model_chainer.py
```

モデルが学習され、`chainer_output/chainer_model.npz`ファイルが生成されます。

学習されるモデルの入力は、28px * 28px、1チャンネルの画像です。値域は0から1(0から255ではない)で、前景が1で背景が0です。出力は、各クラス（数字）に対応するモデルの反応(softmaxを適用する前の値)を表す10次元のベクトルです。

softmax cross entropy lossを用いて学習するにあたり、モデルの定義に注意が必要です。この演習ではChainerの標準的な利用方法に従い、多クラス識別モデルを次のように実装します。
```python
class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.conv1 = L.Convolution2D(None, 8, ksize=3)
            self.conv2 = L.Convolution2D(None, 16, ksize=3)
            self.l3 = L.Linear(None, 32)
            self.l4 = L.Linear(None, 10)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)

model = chainer.links.Classifier(CNN())
```

すなわち、`CNN()`はsoftmaxを適用する前のモデルを表し、`chainer.links.Classifier`がsoftmaxおよび損失計算を担っています。このことはモデルの変換時に注意事項として出てきます。

# DNNモデルの変換
WebDNNを用いて、Keras・Chainerモデルを、Webブラウザで読み込める形式(WebDNNではgraph descriptorと呼ぶ)に変換します。

## Kerasを利用する場合
次のコマンドで変換を行います。

```
python ../webdnn/bin/convert_keras.py --backend webgl --input_shape '(1,28,28,1)' keras_model.h5
```

Webブラウザ上で数値計算を高速に行うために利用出来る規格がいくつかあります。WebDNNではこれをバックエンドと呼び、WebGPU・WebGL・WebAssemblyに対応しています。ここでは環境構築が容易なWebGLを利用することとし、オプションに`--backend webgl`を指定します。環境が整っていれば、`--backend webgpu,webgl,webassembly`のように複数のバックエンドを指定することが可能です。この場合、Webブラウザ側ではそのブラウザが対応しているバックエンドを自動的に選択して読み込むようになっています。

また、モデルへの入力配列の形状を指定する必要があります。画像を扱う場合、バッチサイズ、高さ、幅、チャンネル数の4次元の情報を指定することになります。この4つの情報の順序はデータオーダーと呼ばれ、メモリ上の1次元的な並び方と関わっています。Kerasのデフォルトでは「バッチサイズ、高さ、幅、チャンネル数」の順です[^keras_order]。MNISTの画像は28px * 28px、チャンネル数1（モノクロ）のため、`--input_shape '(1,28,28,1)'`を指定します。先頭の1(バッチサイズ)を他の数値に変更すれば、複数枚の画像を同時に識別させることも可能です。

[^keras_order]: `keras.backend.image_data_format() == "channels_last"`なら「バッチサイズ、高さ、幅、チャンネル数」の順、`"channels_first"`なら「バッチサイズ、チャンネル数、高さ、幅」となります。WebDNNは後者に対応していません。

`webdnn_graph_descriptor`ディレクトリが作成され、中に`graph_webgl_16384.json`などのファイルが出来ます。これがgraph descriptorで、あとでWebブラウザから読み込むことになります。

## Chainerを利用する場合
次のコマンドで変換を行います。

```
python convert_model_chainer.py
```

コードは短いですが、重要な部分について説明を加えておきます。

ChainerはDefine-by-Run方式をとっているため、モデルを実際に動作させて計算グラフを取得し、それをWebDNNに与えてgraph descriptorを生成させるという手順になっています。

モデルオブジェクトの生成および学習したパラメータの読み込みを行います。
```python
model = chainer.links.Classifier(CNN())
chainer.serializers.load_npz('chainer_output/chainer_model.npz', model)
```

ダミーの入力変数`input_variable`を作成し、モデルを動作させます。配列の形状`(1, 1, 28, 28)`は、入力画像の「バッチサイズ、チャンネル数、高さ、幅」を表しています。`model.predictor==CNN()`です。
```python
input_variable = chainer.Variable(np.zeros((1, 1, 28, 28), dtype=np.float32))
prediction_raw_variable = model.predictor(input_variable)
```

softmax関数によって確率に変換します。chainer.links.Classifierの中では、softmaxとcross entropyが同時に計算されてしまい損失しか得ることができません。
```python
prediction_with_softmax_variable = chainer.functions.softmax(prediction_raw_variable)
```

入出力変数(計算履歴情報を保持している)を用いて、DNNの計算グラフをWebDNNの中間表現に変換します。
```python
graph = ChainerConverter().convert([input_variable], [
    prediction_with_softmax_variable])
```

Webブラウザ上で数値計算を高速に行うために利用出来る規格がいくつかあります。WebDNNではこれをバックエンドと呼び、WebGPU・WebGL・WebAssemblyに対応しています。ここでは環境構築が容易なWebGLを利用することとし、オプションに`webgl`を指定して`generate_descriptor`を呼び出し、生成されたgraph descriptorを保存します。環境が整っていれば、`webgpu`, `webassembly`等を指定して同様にgraph descriptorを生成・保存することができます。この場合、Webブラウザ側ではそのブラウザが対応しているバックエンドを自動的に選択して読み込むようになっています。
```python
backend = 'webgl'
exec_info = generate_descriptor(backend, graph)
exec_info.save('webdnn_graph_descriptor')
```

`webdnn_graph_descriptor`ディレクトリが作成され、中に`graph_webgl_16384.json`などのファイルが出来ます。これがgraph descriptorで、あとでWebブラウザから読み込むことになります。

# JavaScriptの実装
## WebDNN JavaScriptのコピー
先ほど生成したgraph descriptorをWebブラウザ上で読み込んで動作させるためのJavaScriptライブラリを、以前cloneしたWebDNNリポジトリからコピーします。

```
cp ../webdnn/dist/webdnn.js webdnn.js
```

(もし`convert_keras.py`で`--encoding eightbit`を指定してモデルを圧縮した場合は、`../webdnn/lib/inflate.min.js`も必要)

## HTML/JavaScriptの実装
Webアプリケーション本体の実装を行います。HTMLでページの視覚的構造を記述し、JavaScriptで動作のロジックを記述します。

アプリケーションの大部分はすでに記述されており、WebDNNに関する部分を編集することで完成するようになっています。わからない部分は`answer`ディレクトリの中を見てコピペしてください。

ページの構造は次のようになっています。必要な処理は、`"draw"`に描かれた画像をモデルに与えて、識別結果を`"result"`に表示することです。マウスで画像を描く部分については、`paint.js`に実装済みです。

![Page structure]({{ site.baseurl }}/images/page_structure.png)

編集すべきファイルは`index.html`、`script.js`の2つです。編集すべき箇所には`FIXME`というキーワードが書かれています。

`index.html`では、以下の箇所を編集します。

`script`タグで、先ほどコピーした`webdnn.js`を読み込みます。
```html
  <meta charset="utf-8">
  <!-- FIXME: load webdnn.js -->
  <script src="paint.js"></script>
```

`script.js`では、以下の箇所を編集します。

`initialize`関数内

```javascript
// Load WebDNN model
webdnn_runner = /* FIXME: uncomment -> await WebDNN.load('./webdnn_graph_descriptor'); */
// Get view object for input / output variable of the model
// There can be multiple variables for input / output, but there is only one for this model
webdnn_input_view = /* FIXME: uncomment -> webdnn_runner.getInputViews()[0]; */
webdnn_output_view = /* FIXME: uncomment -> webdnn_runner.getOutputViews()[0]; */
```

1. 先ほど生成したgraph descriptorの読み込み
  `WebDNN.load()`により、graph descriptorを読み込みます。これにより、runnerと呼ばれるオブジェクトが生成されます。runnerは、モデルの実行を制御する中心的なオブジェクトです。なお、`WebDNN.load()`は非同期関数のため、`await`キーワードで結果を取り出します([参考](https://developer.mozilla.org/ja/docs/Web/JavaScript/Reference/Operators/await))。

2. 入出力ビューの取得
  runnerから、モデルの入出力データを操作するためのviewと呼ばれるオブジェクトを取得します。

`calculate`関数内

```javascript
// set input data by scaling and flattening the image in canvas
// https://mil-tokyo.github.io/webdnn/docs/api_reference/descriptor-runner/modules/webdnn_image.html
var canvas_draw = document.getElementById('draw'); // canvas object that contains input image
webdnn_input_view.set(await WebDNN.Image.getImageArray(canvas_draw, {
  /* convert image on canvas into input array (Float32Array, 28px*28px*1ch=784 dimension, black=0.0 and white=1.0) */
  dstH: /* FIXME */, dstW: /* FIXME */,
  order: WebDNN.Image.Order.HWC, // for Keras
  // order: WebDNN.Image.Order.CHW, // for Chainer (but the same as HWC because channel=1 in MNIST)
  color: /* FIXME */,
  bias: /* FIXME */,
  scale: /* FIXME */
}));

// run the DNN
/* FIXME: uncomment -> await webdnn_runner.run(); */

// get model output as Float32Array
var probabilities = /* FIXME: webdnn_output_view.toActual(); */
```

1. canvas上の画像を変換し、入力変数に書き込み

canvasオブジェクト`canvas_draw`に、ユーザが描いた画像を表すデータが入っています。これをモデルに入力するためには所定の形式に変換する必要があります。`canvas_draw.getContext('2d').getImageData`メソッドにより生のピクセル値を取り出すことはできますが、モデルの入力として用いるためにはいくつかの変換が必要です。

|項目|canvasデータ|モデルが受け付ける形式|
|---|---|---|
|ピクセル数|256px * 256px|28px * 28px|
|色|RGBA(カラー+透明度)|モノクロ|
|値域|0~255|0~1|
|データオーダー|HWC|HWC(Kerasモデルの場合)、CHW(Chainerモデルの場合)|

データオーダーとは画像を1次元の配列で表現した時のピクセルの並び順のことです。ピクセルの明るさを関数`I(y, x, channel)`で表現した場合、HWC(height, width, channel)なら`[I(0, 0, 0), I(0, 0, 1), I(0, 0, 2), I(0, 1, 0)...]`となり、CHW(chanel, height, width)なら`[I(0, 0, 0), I(0, 1, 0), I(0, 2, 0), ..., I(0, W-1, 0), I(1, 0, 0), ...]`というように順序が定まります。

これらの変換を簡単に行うためのメソッドが`WebDNN.Image.getImageArray`です。[マニュアル](https://mil-tokyo.github.io/webdnn/docs/api_reference/descriptor-runner/modules/webdnn_image.html)を見ながらFIXMEとなっているところを埋めましょう。

2. モデルを実行

`webdnn_runner.run()`でモデルを実行できます。

3. 出力変数から計算結果を取り出し

`webdnn_output_view.toActual()`で、モデルの計算結果(各クラスの確率)を取り出すことができます。後続の処理は、これを加工して画面に表示しています。

# 動作確認
## テスト用HTTPサーバの実行
作成したページをWebブラウザで開けるようにするため、HTTPサーバを実行します。
```
python -m http.server
```

アクセス待機状態になるので、次に進みます。HTTPサーバを終了するときは、Ctrl-Cを押します。

## Webブラウザでアクセス
作成したページにWebブラウザでアクセスします。

[http://localhost:8000/](http://localhost:8000/)

黒いボックス上でマウスをクリック・ドラッグすることで数字(0~9)を描きます。正しい識別結果がボックスの下に表示されれば成功です。

データセットの性質上、ボックス中央に小さめに数字を描いた方が認識されやすいようです。

うまくいかない場合、開発用コンソールでエラーメッセージを見るなどしながらデバッグします。

なお、次のようなエラーは出ていても構いません。このエラーは、WebAssemblyバックエンドに対応したgraph descriptorが存在しないために出ています。
```
GET http://localhost:8000/webdnn_graph_descriptor/graph_webassembly.json 404 (File not found)
```

# Safariについて
WebDNNはMac OS標準のWebブラウザSafariにも対応していますが、本演習の手順では他のブラウザと同様に動作させることができません。本演習ではWebDNNの計算バックエンドとしてWebGLを使用していますが、このバックエンドはSafariと互換性がないためです。SafariではWebGPUまたはWebAssemblyバックエンドが使用できます。WebGPUバックエンドはブラウザの設定が必要(→ [https://mil-tokyo.github.io/webdnn/docs/tips/enable_webgpu_macos.html](https://mil-tokyo.github.io/webdnn/docs/tips/enable_webgpu_macos.html) )です。WebAssemblyバックエンドはブラウザの設定は不要ですが、環境構築に若干時間がかかります。

Safariで演習を行う場合、`convert_keras.py`で作成されたディレクトリ`webdnn_graph_descriptor`に、リポジトリ直下にある`webdnn_graph_descriptor_for_all_backend`の中身を上書きしてください。全てのバックエンドに対応したgraph descriptorが同梱されているため、WebGPU(有効な場合)およびWebAssemblyを利用してSafariでのWebDNN実行が可能となります。
