import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
YOLO_MODEL = MODEL_DIR / "yolo11n.onnx"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"

PERSON_COLOR = (0, 200, 0)
FACE_COLOR = (255, 120, 0)
HAND_COLOR = (180, 80, 255)
PEACE_COLOR = (0, 255, 255)
SAVED_COLOR = (120, 120, 120)
INPUT_SIZE = 640
COUNTDOWN_SECONDS = 3.0

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int, float]
    missed_frames: int = 0


class PersonTracker:
    """検出枠の位置を使った軽量な人物IDトラッカー。"""

    def __init__(self, max_missed_frames: int = 20) -> None:
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    @staticmethod
    def _iou(first, second) -> float:
        ax, ay, aw, ah = first[:4]
        bx, by, bw, bh = second[:4]
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center_distance(first, second) -> float:
        ax, ay, aw, ah = first[:4]
        bx, by, bw, bh = second[:4]
        return math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))

    def update(self, detections) -> list[Track]:
        unmatched_tracks = set(self.tracks)
        unmatched_detections = set(range(len(detections)))
        candidates = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                iou = self._iou(track.box, detection)
                distance_between = self._center_distance(track.box, detection)
                size = max(track.box[2], track.box[3], detection[2], detection[3])
                if iou >= 0.10 or distance_between <= size * 0.35:
                    # IoUを優先し、近い中心位置を補助的に使います。
                    candidates.append((iou - distance_between / max(size, 1) * 0.1, track_id, detection_index))

        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self.tracks[track_id].box = detections[detection_index]
            self.tracks[track_id].missed_frames = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in unmatched_tracks:
            self.tracks[track_id].missed_frames += 1

        for detection_index in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(track_id, detections[detection_index])

        expired = [track_id for track_id, track in self.tracks.items() if track.missed_frames > self.max_missed_frames]
        for track_id in expired:
            del self.tracks[track_id]

        return [track for track in self.tracks.values() if track.missed_frames == 0]


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
        help="ピース検出から保存までの秒数（デフォルト: 3.0）",
    )
    return parser.parse_args()


def require_models() -> None:
    missing = [path for path in (YOLO_MODEL, HAND_MODEL, FACE_MODEL) if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"モデルファイルがありません: {names}")


def create_detectors():
    require_models()
    person_detector = cv2.dnn.readNetFromONNX(str(YOLO_MODEL))

    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.5,
    )
    hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    face_detector = mp.tasks.vision.FaceDetector.create_from_options(face_options)
    return person_detector, face_detector, hand_detector


def detect_people(frame: np.ndarray, detector, confidence: float) -> list[tuple[int, int, int, int, float]]:
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    detector.setInput(blob)
    output = detector.forward()[0].T

    boxes: list[list[int]] = []
    scores: list[float] = []
    scale_x = width / INPUT_SIZE
    scale_y = height / INPUT_SIZE

    # COCOデータセットではクラス0がperson。YOLO11の出力は xywh + 80クラス。
    for detection in output:
        score = float(detection[4])
        if score < confidence:
            continue
        center_x, center_y, box_width, box_height = detection[:4]
        x = int((center_x - box_width / 2) * scale_x)
        y = int((center_y - box_height / 2) * scale_y)
        boxes.append([x, y, int(box_width * scale_x), int(box_height * scale_y)])
        scores.append(score)

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.45)
    return [(*boxes[int(index)], scores[int(index)]) for index in indices]


def angle(a, b, c) -> float:
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
    return math.degrees(math.acos(cosine))


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def is_peace_sign(landmarks) -> bool:
    wrist = landmarks[0]

    def extended(mcp: int, pip: int, tip: int) -> bool:
        return angle(landmarks[mcp], landmarks[pip], landmarks[tip]) > 155 and (
            distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist)
        )

    def folded(mcp: int, pip: int, tip: int) -> bool:
        return angle(landmarks[mcp], landmarks[pip], landmarks[tip]) < 145 or (
            distance(landmarks[tip], wrist) < distance(landmarks[pip], wrist)
        )

    index_up = extended(5, 6, 8)
    middle_up = extended(9, 10, 12)
    ring_down = folded(13, 14, 16)
    pinky_down = folded(17, 18, 20)
    fingers_apart = distance(landmarks[8], landmarks[12]) > distance(landmarks[5], landmarks[17]) * 0.30
    return index_up and middle_up and ring_down and pinky_down and fingers_apart


def draw_people(frame: np.ndarray, tracks: list[Track], saved_ids: set[int]) -> None:
    for track in tracks:
        x, y, width, height, score = track.box
        saved = track.track_id in saved_ids
        color = SAVED_COLOR if saved else PERSON_COLOR
        state = "SAVED" if saved else "READY"
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            f"ID {track.track_id} {state} {score:.0%}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_faces(frame: np.ndarray, detections) -> None:
    for detection in detections:
        box = detection.bounding_box
        x, y, width, height = box.origin_x, box.origin_y, box.width, box.height
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


def draw_hands(frame: np.ndarray, hands) -> int:
    height, width = frame.shape[:2]
    peace_count = 0
    for landmarks in hands:
        points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
        peace = is_peace_sign(landmarks)
        color = PEACE_COLOR if peace else HAND_COLOR
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2)
        for point in points:
            cv2.circle(frame, point, 3, color, -1)

        if peace:
            peace_count += 1
            label_x = min(x for x, _ in points)
            label_y = max(min(y for _, y in points) - 12, 25)
            cv2.putText(
                frame,
                "PEACE!",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                PEACE_COLOR,
                2,
            )
    return peace_count


def detect_and_draw(
    frame,
    person_detector,
    face_detector,
    hand_detector,
    tracker,
    saved_ids,
    confidence,
    timestamp_ms,
):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    people = detect_people(frame, person_detector, confidence)
    tracks = tracker.update(people)
    faces = face_detector.detect_for_video(mp_image, timestamp_ms).detections
    hands = hand_detector.detect_for_video(mp_image, timestamp_ms).hand_landmarks

    draw_people(frame, tracks, saved_ids)
    draw_faces(frame, faces)
    peace_count = draw_hands(frame, hands)
    active_ids = {track.track_id for track in tracks}
    ready = bool(active_ids - saved_ids)
    status = "READY" if ready else "WAITING FOR NEW PERSON"
    cv2.putText(
        frame,
        f"People: {len(tracks)}  Faces: {len(faces)}  Peace: {peace_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        status,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        PERSON_COLOR if ready else SAVED_COLOR,
        2,
    )
    return active_ids, peace_count


def save_frame(frame: np.ndarray, output_dir: Path, prefix: str) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{prefix}_{timestamp}.jpg"
    if cv2.imwrite(str(output_path), frame):
        print(f"保存しました: {output_path}")
        return True
    print(f"保存に失敗しました: {output_path}")
    return False


def draw_countdown(frame: np.ndarray, remaining_seconds: float) -> None:
    height, width = frame.shape[:2]
    count = max(1, math.ceil(remaining_seconds))
    text = f"{count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 4
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max((width - text_width) // 2, 10)
    y = max((height + text_height) // 2, text_height + baseline + 10)
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.countdown_seconds < 0:
        raise ValueError("--countdown-seconds は 0 以上にしてください。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    person_detector, face_detector, hand_detector = create_detectors()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        face_detector.close()
        hand_detector.close()
        raise RuntimeError(f"カメラ {args.camera} を開けませんでした。接続状態やカメラ番号を確認してください。")

    tracker = PersonTracker()
    saved_ids: set[int] = set()
    countdown_start: float | None = None
    countdown_ids: set[int] = set()

    print("ピースで自動保存 / s: 手動保存 / q または Esc: 終了")
    start_time = time.monotonic()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("カメラから画像を取得できませんでした。")
                break

            raw_frame = frame.copy()
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            active_ids, peace_count = detect_and_draw(
                frame,
                person_detector,
                face_detector,
                hand_detector,
                tracker,
                saved_ids,
                args.person_confidence,
                timestamp_ms,
            )

            unsaved_active_ids = active_ids - saved_ids
            now = time.monotonic()

            # 未保存の人物が画面内にいて、少なくとも一人がピースしたら保存カウントダウンを開始。
            if countdown_start is None and peace_count > 0 and unsaved_active_ids:
                countdown_start = now
                countdown_ids = set(unsaved_active_ids)

            if countdown_start is not None:
                if not (countdown_ids & active_ids):
                    countdown_start = None
                    countdown_ids.clear()
                else:
                    remaining_seconds = args.countdown_seconds - (now - countdown_start)
                    if remaining_seconds <= 0:
                        if save_frame(raw_frame, args.output_dir, "peace"):
                            saved_ids.update(countdown_ids)
                        countdown_start = None
                        countdown_ids.clear()
                    else:
                        draw_countdown(frame, remaining_seconds)

            cv2.imshow("Person, Face and Peace Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                save_frame(raw_frame, args.output_dir, "capture")
            elif key in (ord("q"), 27):
                break
    finally:
        camera.release()
        face_detector.close()
        hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
