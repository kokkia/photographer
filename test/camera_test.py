import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCVでカメラ画像を取り込みます。")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="使用するカメラ番号（デフォルト: 0）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captured_images"),
        help="画像の保存先（デフォルト: captured_images）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"カメラ {args.camera} を開けませんでした。接続状態やカメラ番号を確認してください。"
        )

    print("s: 画像を保存 / q または Esc: 終了")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("カメラから画像を取得できませんでした。")
                break

            cv2.imshow("Laptop Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = args.output_dir / f"capture_{timestamp}.jpg"
                if cv2.imwrite(str(output_path), frame):
                    print(f"保存しました: {output_path}")
                else:
                    print(f"保存に失敗しました: {output_path}")
            elif key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
