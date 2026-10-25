# photographer

ノートPCのカメラ映像から、人物・顔・ピースサインをリアルタイムで検出するOpenCVアプリです。

## 環境構築 (apt)
```bash
sudo apt install vlc v4l-utils cheese
v4l2-ctl --list-devices # check camera num
cheese # camera test
```


## 環境構築 (venv)

プロジェクトのルートディレクトリで、Pythonの仮想環境を作成します。

```bash
python3 -m venv camera_env
source camera_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

仮想環境を終了するときは、次を実行します。

```bash
deactivate
```

## 環境構築 (conda)

```bash
conda create -n camera310 python=3.10
conda activate camera310
conda install -c conda-forge xcb-util-cursor
python -m pip install -r requirements.txt
```

GUIには `PySide6==6.9.3` を使用します。PySide6は `requirements.txt` から、
Linux版Qtの起動に必要な `xcb-util-cursor` はCondaからインストールされます。

仮想環境を終了するときは、次を実行します。

```bash
conda deactivate
```

## モデルファイル

`models/` に次のファイルが必要です。このリポジトリにファイルが含まれていない場合は、以下のコマンドで取得してください。

```bash
mkdir -p models
curl -fL \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx \
  -o models/yolo11n.onnx
curl -fL \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  -o models/hand_landmarker.task
curl -fL \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite \
  -o models/blaze_face_short_range.tflite
```

## 実行方法

```bash
source camera_env/bin/activate
```

Qt GUIは次のコマンドで起動します。

```bash
conda activate camera310
python -m src.gui --instax-address FA:AB:BC:XX:XX:XX
```

GUIはスタート画面から始まり、開始ボタンを押すと認識・撮影画面へ移ります。
手動撮影またはピースサインによる自動撮影が完了すると撮影画像を表示し、
「印刷する」を選ぶとスタート画面へ戻ります。「印刷しない」を選ぶと、
新しい認識・撮影セッションへ戻ります。
スタート画面の画像は `assets/start_screen.jpg` に置くか、次のように指定できます。

```bash
python -m src.gui --start-image path/to/start-image.jpg
```

起動スクリプトを使用すると、`camera310` の有効化とプロジェクトディレクトリへの移動も自動で行われます。

```bash
./script/run_photographer.sh --instax-address FA:AB:BC:XX:XX:XX
```

別名のConda環境を使う場合は、`PHOTOGRAPHER_CONDA_ENV` で指定できます。

```bash
PHOTOGRAPHER_CONDA_ENV=my-camera-env ./script/run_photographer.sh
```

`--instax-address` を省略すると、名前が `INSTAX-` で始まるプリンターを自動検索します。
環境変数 `INSTAX_DEVICE_ADDRESS` でもBluetoothアドレスを指定できます。

`xcb` プラットフォームプラグインのエラーが表示されるLinux環境では、
`xcb-util-cursor` がインストールされていることを確認してください。

> [!WARNING]
> GUIの確認画面で「印刷する」を押すと、接続したInstaxで実際に印刷され、フィルムを1枚使用します。
> `python -m src.main` で起動した場合は、撮影後すぐに印刷されます。

## Capture test

```bash
python test/face_recognition.py
```

人物検出の感度は変更できます。値を大きくすると誤検出が減り、小さくすると検出されやすくなります。

```bash
python test/face_recognition.py --person-confidence 0.5
```

別のカメラを使う場合は、カメラ番号を指定します。

```bash
python test/face_recognition.py --camera 1
```

### 操作

- `s`: 検出結果をJPEGで `captured_images/` に保存
- `q` または `Esc`: 終了

### Result
![sample](./sample_data/sample_data.jpg)

## Instax Link 3 test

Instaxプリンターの電源を入れ、PCのBluetoothを有効にします。すでにOSから接続またはペアリング済みの場合も、その機器を検出して利用します。

最初にプリンターのBluetoothアドレスを確認します。

```bash
source camera_env/bin/activate
python test/bluetooth_test.py
```

表示されたアドレスを、`test/instax_test.py` の `device_address` に設定します。

```python
instax = InstaxBLE(
    device_address="FA:AB:BC:XX:XX:XX",
    verbose=True,
)
```

印刷するJPEG画像を次の場所へ配置します。

```text
test/captured_images/test_image.jpg
```

テストは必ずプロジェクトのルートディレクトリから実行します。

```bash
source camera_env/bin/activate
python test/instax_test.py
```

プリンターの検出、接続、画像変換、転送が順番に実行されます。横向き画像は90度回転し、縦横比を維持したままプリンターの画像サイズに合わせて中央をトリミングします。元画像は変更されません。

> [!WARNING]
> `instax_test.py` は `enable_printing()` を呼び出すため、実行すると実際にフィルムを1枚使用して印刷します。

処理を中断する場合は `Ctrl+C` を押します。接続中であれば、終了処理でプリンターから切断されます。
