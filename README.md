# photographer

ノートPCのカメラ映像から、人物・顔・ピースサインをリアルタイムで検出するOpenCVアプリです。

## 環境構築

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

## 操作

- `s`: 検出結果をJPEGで `captured_images/` に保存
- `q` または `Esc`: 終了
