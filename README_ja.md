# 環境構築
python環境は、python 3.6, numpy, keras 2.0が使えればOKです。

anacondaで仮想環境を構築する場合は、以下のコマンドで行えます。

```
conda env create -f environment.yml
```

この場合、名前`webdnn`で仮想環境が作られるので、

```
source activate webdnn
```

で仮想環境に入れます。

JavaScriptについては、モダンなWebブラウザのみ必要です。node.jsは不要です。

# WebDNNのインストール
```
git clone https://github.com/mil-tokyo/webdnn
cd webdnn
python setup.py install
cd ..
```

(執筆時点commit 519bf7c)

# モデルの学習
これ以降、`exercise`ディレクトリで作業します。`answer`ディレクトリに同じファイル構成で答えと計算済み出力ファイルが入っています。

```
python train_model_keras.py
```

モデルが学習され、`keras_model.h5`ファイルが生成されます。

# モデルの変換
Kerasモデルを、Webブラウザで読み込める形式(WebDNNではgraph descriptorと呼ぶ)に変換します。

```
python ../webdnn/bin/convert_keras.py --backend webgl --input_shape '(1,28,28,1)' keras_model.h5
```

# WebDNN JavaScriptのコピー
```
cp ../webdnn/dist/webdnn.js webdnn.js
```

(もし`convert_keras.py`で`--encoding eightbit`を指定してモデルを圧縮した場合は、`../webdnn/lib/inflate.min.js`も必要)

# 動作確認
## テスト用HTTPサーバの実行
```
python -m http.server
```

