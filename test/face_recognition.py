import argparse
from datetime import datetime
from pathlib import Path

import cv2


PERSON_COLOR = (0, 200, 0)
FACE_COLOR = (255, 120, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="カメラ映像から人物と顔をリアルタイム検出します。"
    )
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


def create_detectors() -> tuple[cv2.HOGDescriptor, cv2.CascadeClassifier]:
    person_detector = cv2.HOGDescriptor()
    person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError(f"顔検出モデルを読み込めませんでした: {cascade_path}")

    return person_detector, face_detector


def detect_and_draw(
    frame,
    person_detector: cv2.HOGDescriptor,
    face_detector: cv2.CascadeClassifier,
) -> tuple[int, int]:
    people, _ = person_detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    for x, y, width, height in people:
        # HOGの検出枠は少し大きいため、表示枠を人物側へ寄せます。
        x += round(width * 0.1)
        width = round(width * 0.8)
        y += round(height * 0.07)
        height = round(height * 0.85)
        cv2.rectangle(frame, (x, y), (x + width, y + height), PERSON_COLOR, 2)
        cv2.putText(
            frame,
            "Person",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PERSON_COLOR,
            2,
        )

    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), FACE_COLOR, 2)
        cv2.putText(
            frame,
            "Face",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            FACE_COLOR,
            2,
        )

    cv2.putText(
        frame,
        f"People: {len(people)}  Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return len(people), len(faces)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    person_detector, face_detector = create_detectors()

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

            detect_and_draw(frame, person_detector, face_detector)
            cv2.imshow("Person and Face Detection", frame)
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
