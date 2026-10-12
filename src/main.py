import argparse
import os
import threading
from pathlib import Path
from queue import Queue

if __package__:
    from .camera_detection_worker import (
        COUNTDOWN_SECONDS,
        DETECTION_INTERVAL,
        INPUT_SIZE,
        CameraDetectionWorker,
        CameraSettings,
    )
    from .instax_print_worker import InstaxPrintWorker
    from .job import PrintJob
else:  # python src/main.py で直接実行する場合
    from camera_detection_worker import (
        COUNTDOWN_SECONDS,
        DETECTION_INTERVAL,
        INPUT_SIZE,
        CameraDetectionWorker,
        CameraSettings,
    )
    from instax_print_worker import InstaxPrintWorker
    from job import PrintJob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="カメラ映像から人物、顔、ピースサインをリアルタイム検出します。")
    parser.add_argument("--camera", type=int, default=0, help="カメラ番号（デフォルト: 0）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captured_images"),
        help="画像の保存先（デフォルト: captured_images）",
    )
    parser.add_argument(
        "--person-confidence",
        type=float,
        default=0.35,
        help="人物検出の信頼度しきい値（デフォルト: 0.35）",
    )
    parser.add_argument(
        "--countdown-seconds",
        type=float,
        default=COUNTDOWN_SECONDS,
        help="ピースを連続検出してから保存するまでの秒数（デフォルト: 3.0）",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=INPUT_SIZE,
        help="YOLOの入力サイズ（デフォルト: 416、小さいほど高速）",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=DETECTION_INTERVAL,
        help="検出を行うフレーム間隔（デフォルト: 2）",
    )
    parser.add_argument(
        "--instax-address",
        default=os.environ.get("INSTAX_DEVICE_ADDRESS"),
        help="InstaxのBluetoothアドレス（省略時: 自動検索）",
    )
    parser.add_argument(
        "--instax-connect-timeout",
        type=int,
        default=10,
        help="Instax検索のタイムアウト秒数（デフォルト: 10）",
    )
    parser.add_argument(
        "--instax-verbose",
        action="store_true",
        help="Instax通信の詳細ログを表示",
    )
    parser.add_argument(
        "--start-image",
        type=Path,
        help="GUIのスタート画面に表示する画像",
    )
    return parser.parse_args()


def validate_settings(settings: CameraSettings) -> None:
    if settings.countdown_seconds < 0:
        raise ValueError("--countdown-seconds は 0 以上にしてください。")
    if settings.input_size < 160 or settings.input_size % 32 != 0:
        raise ValueError("--input-size は160以上かつ32の倍数にしてください。")
    if settings.detection_interval < 1:
        raise ValueError("--detection-interval は1以上にしてください。")


def main() -> None:
    args = parse_args()
    settings = CameraSettings.from_args(args)
    validate_settings(settings)

    stop_event = threading.Event()
    print_queue: Queue[PrintJob | None] = Queue()
    camera_worker = CameraDetectionWorker(settings, print_queue, stop_event)
    instax_worker = InstaxPrintWorker(
        print_queue,
        device_address=args.instax_address,
        connect_timeout=args.instax_connect_timeout,
        verbose=args.instax_verbose,
    )

    instax_worker.start()
    camera_worker.start()
    try:
        while camera_worker.is_alive():
            camera_worker.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n終了しています...")
        stop_event.set()
    finally:
        camera_worker.join()
        print_queue.put(None)
        instax_worker.join()

    if camera_worker.error is not None:
        raise RuntimeError("カメラスレッドでエラーが発生しました。") from camera_worker.error


if __name__ == "__main__":
    main()
