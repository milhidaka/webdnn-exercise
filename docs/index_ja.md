# WebDNNの基本的な使い方の演習
この記事は、Webブラウザ上でDeep Neural Networkを実行できるフレームワークWebDNNの基本的な使い方を演習を通して解説するものです。

演習を最後まで進めると、次のような画面が得られます。画面上の黒い領域にマウスで数字を描くと、それがどの数字なのかを認識して表示します。

![Screenshot of complete application](https://raw.githubusercontent.com/milhidaka/webdnn-exercise/master/images/complete_screenshot.png)

動作するデモが[ここ](https://milhidaka.github.io/webdnn-exercise/answer/)から見られます。

WebDNNのリポジトリ: https://github.com/mil-tokyo/webdnn

## 対象読者
この演習では、深層学習(Deep Learning)自体の説明は行いません。Pythonの基本的な文法およびKerasまたはChainerで簡単な画像識別モデルを学習させる方法についての知識があることを仮定しています。JavaScriptについても知識があると望ましいですが、なくてもかまいません。

## 演習の流れ
以下の流れで演習が進みます。
1. 環境構築
2. WebDNNのインストール
3. DNNモデルの学習
4. DNNモデルの変換
5. JavaScriptの実装
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

学習されるモデルの入力は、28px * 28px、1チャンネルの画像です。値域は0から1(0から255ではない)で、前景が1で背景が0です。出力は、各クラス（数字）に対応するsoftmax確率を表す10次元のベクトルです。

TODO; `L.Classifier`の扱い

# DNNモデルの変換
WebDNNを用いて、Keras・Chainerモデルを、Webブラウザで読み込める形式(WebDNNではgraph descriptorと呼ぶ)に変換します。

## Kerasを利用する場合
```
python ../webdnn/bin/convert_keras.py --backend webgl --input_shape '(1,28,28,1)' keras_model.h5
```

Webブラウザ上で数値計算を高速に行うために利用出来る規格がいくつかあります。WebDNNではこれをバックエンドと呼び、WebGPU・WebGL・WebAssemblyに対応しています。ここでは環境構築が容易なWebGLを利用することとし、オプションに`--backend webgl`を指定します。環境が整っていれば、`--backend webgpu,webgl,webassembly`のように複数のバックエンドを指定することが可能です。この場合、Webブラウザ側ではそのブラウザが対応しているバックエンドを自動的に選択して読み込むようになっています。

TODO: input shape

`webdnn_graph_descriptor`ディレクトリが作成され、中に`graph_webgl_16384.json`などのファイルが出来ます。これがgraph descriptorで、あとでWebブラウザから読み込むことになります。

## Chainerを利用する場合
```
python convert_model_chainer.py
```

`webdnn_graph_descriptor`ディレクトリが作成され、中に`graph_webgl_16384.json`などのファイルが出来ます。これがgraph descriptorで、あとでWebブラウザから読み込むことになります。

# JavaScriptの実装
## WebDNN JavaScriptのコピー
TODO
```
cp ../webdnn/dist/webdnn.js webdnn.js
```

(もし`convert_keras.py`で`--encoding eightbit`を指定してモデルを圧縮した場合は、`../webdnn/lib/inflate.min.js`も必要)

## JavaScriptの実装
TODO

# 動作確認
## テスト用HTTPサーバの実行
作成したページをWebブラウザで開けるようにするため、HTTPサーバを実行します。
```
python -m http.server
```

アクセス待機状態になるので、次に進みます。HTTPサーバを終了するときは、Ctrl-Cを押します。

## Webブラウザでアクセス
作成したページにWebブラウザでアクセスします。

http://localhost:8000/

黒いボックス上でマウスをクリック・ドラッグすることで数字(0~9)を描きます。正しい識別結果がボックスの下に表示されれば成功です。

データセットの性質上、ボックス中央に小さめに数字を描いた方が認識されやすいようです。

うまくいかない場合、開発用コンソールでエラーメッセージを見るなどしながらデバッグします。

なお、次のようなエラーは出ていても構いません。このエラーは、WebAssemblyバックエンドに対応したgraph descriptorが存在しないために出ています。
```
GET http://localhost:8000/webdnn_graph_descriptor/graph_webassembly.json 404 (File not found)
```

# Safariについて
WebDNNはMac OS標準のWebブラウザSafariにも対応していますが、本演習の手順では他のブラウザと同様に動作させることができません。本演習ではWebDNNの計算バックエンドとしてWebGLを使用していますが、このバックエンドはSafariと互換性がないためです。SafariではWebGPUまたはWebAssemblyバックエンドが使用できます。WebGPUバックエンドはブラウザの設定が必要(→ https://mil-tokyo.github.io/webdnn/docs/tips/enable_webgpu_macos.html )です。WebAssemblyバックエンドはブラウザの設定は不要ですが、環境構築に若干時間がかかります。

Safariで演習を行う場合、`convert_keras.py`で作成されたディレクトリ`webdnn_graph_descriptor`に、リポジトリ直下にある`webdnn_graph_descriptor_for_all_backend`の中身を上書きしてください。全てのバックエンドに対応したgraph descriptorが同梱されているため、WebGPU(有効な場合)およびWebAssemblyを利用してSafariでのWebDNN実行が可能となります。
