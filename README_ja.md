# WebDNNの基本的な使い方の演習
この記事は、Webブラウザ上でDeep Neural Networkを実行できるフレームワークWebDNNの基本的な使い方を演習を通して解説するものです。

WebDNNのリポジトリ: https://github.com/mil-tokyo/webdnn

# 環境構築
python環境は、python 3.6, numpy, Keras 2.0が使えればOKです。

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

# モデルの学習
これ以降、`exercise`ディレクトリで作業します。`answer`ディレクトリに同じファイル構成で答えと計算済み出力ファイルが入っています。

まず、Webブラウザ上で動かしたいモデルを学習します。この演習では、MNISTデータセットを用いた文字認識Convolutional Neural NetworkをKerasを用いて学習します。

```
python train_model_keras.py
```

TODO: モデルの入出力についての説明

モデルが学習され、`keras_model.h5`ファイルが生成されます。この時点では、WebDNNに依存する処理はまだ含まれていません。

# モデルの変換
WebDNNを用いて、Kerasモデルを、Webブラウザで読み込める形式(WebDNNではgraph descriptorと呼ぶ)に変換します。

```
python ../webdnn/bin/convert_keras.py --backend webgl --input_shape '(1,28,28,1)' keras_model.h5
```

Webブラウザ上で数値計算を高速に行うために利用出来る規格がいくつかあります。WebDNNではこれをバックエンドと呼び、WebGPU・WebGL・WebAssemblyに対応しています。ここでは環境構築が容易なWebGLを利用することとし、オプションに`--backend webgl`を指定します。環境が整っていれば、`--backend webgpu,webgl,webassembly`のように複数のバックエンドを指定することが可能です。この場合、Webブラウザ側ではそのブラウザが対応しているバックエンドを自動的に選択して読み込むようになっています。

TODO: input shape

`webdnn_graph_descriptor`ディレクトリが作成され、中に`graph_webgl_16384.json`などのファイルが出来ます。これがgraph descriptorで、あとでWebブラウザから読み込むことになります。

# WebDNN JavaScriptのコピー
TODO
```
cp ../webdnn/dist/webdnn.js webdnn.js
```

(もし`convert_keras.py`で`--encoding eightbit`を指定してモデルを圧縮した場合は、`../webdnn/lib/inflate.min.js`も必要)

# JavaScriptの実装
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
