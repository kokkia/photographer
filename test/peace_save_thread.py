import argparse
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue

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
INPUT_SIZE = 416
DETECTION_INTERVAL = 2
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


@dataclass(frozen=True)
class CameraSettings:
    camera_index: int
    output_dir: Path
    person_confidence: float
    countdown_seconds: float
    input_size: int
    detection_interval: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CameraSettings":
        return cls(
            camera_index=args.camera,
            output_dir=args.output_dir,
            person_confidence=args.person_confidence,
            countdown_seconds=args.countdown_seconds,
            input_size=args.input_size,
            detection_interval=args.detection_interval,
        )


@dataclass(frozen=True)
class PrintJob:
    """カメラスレッドからInstaxスレッドへ渡す不変の印刷要求。"""

    image_path: Path
    created_at: datetime
    person_ids: tuple[int, ...] = ()


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


def detect_people(
    frame: np.ndarray,
    detector,
    confidence: float,
    input_size: int,
) -> list[tuple[int, int, int, int, float]]:
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (input_size, input_size), swapRB=True, crop=False)
    detector.setInput(blob)
    output = detector.forward()[0].T

    # COCOデータセットではクラス0がperson。YOLO11の出力は xywh + 80クラス。
    # 数千候補をPythonで1件ずつ処理せず、NumPyでまとめて絞り込みます。
    scores_array = output[:, 4]
    selected = output[scores_array >= confidence]
    if selected.size == 0:
        return []

    scores = selected[:, 4].astype(float).tolist()
    xywh = selected[:, :4].copy()
    xywh[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * (width / input_size)
    xywh[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * (height / input_size)
    xywh[:, 2] *= width / input_size
    xywh[:, 3] *= height / input_size
    boxes = xywh.astype(np.int32).tolist()

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
    input_size,
    run_detection,
    cached_tracks,
    cached_faces,
    cached_hands,
):
    if run_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        people = detect_people(frame, person_detector, confidence, input_size)
        tracks = tracker.update(people)
        faces = face_detector.detect_for_video(mp_image, timestamp_ms).detections
        hands = hand_detector.detect_for_video(mp_image, timestamp_ms).hand_landmarks
    else:
        tracks, faces, hands = cached_tracks, cached_faces, cached_hands

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
    return active_ids, peace_count, tracks, faces, hands


def save_frame(frame: np.ndarray, output_dir: Path, prefix: str) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{prefix}_{timestamp}.jpg"
    if cv2.imwrite(str(output_path), frame):
        print(f"保存しました: {output_path}")
        return output_path
    print(f"保存に失敗しました: {output_path}")
    return None


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


class InstaxPrintWorker(threading.Thread):
    """印刷要求を順番に受け取るInstax用ワーカースレッド。"""

    def __init__(self, print_queue: Queue[PrintJob | None]) -> None:
        super().__init__(name="instax-print-worker")
        self.print_queue = print_queue

    def run(self) -> None:
        while True:
            job = self.print_queue.get()
            try:
                if job is None:
                    return
                self.print_image(job)
            finally:
                self.print_queue.task_done()

    def print_image(self, job: PrintJob) -> None:
        """後からInstax接続・印刷処理を実装するための差し替えポイント。"""
        print("Instax印刷要求を受信しました（印刷処理は未実装）")
        print(f"  image_path: {job.image_path}")
        print(f"  created_at: {job.created_at.isoformat(timespec='milliseconds')}")
        print(f"  person_ids: {job.person_ids}")


class CameraDetectionWorker(threading.Thread):
    """カメラ取得、推論、描画、画像保存を単一スレッドで実行する。"""

    WINDOW_NAME = "Person, Face and Peace Detection"

    def __init__(
        self,
        settings: CameraSettings,
        print_queue: Queue[PrintJob | None],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="camera-detection-worker")
        self.settings = settings
        self.print_queue = print_queue
        self.stop_event = stop_event
        self.error: BaseException | None = None

        # これらの状態はカメラスレッドだけが更新する。
        self.tracker = PersonTracker()
        self.saved_ids: set[int] = set()
        self.countdown_start: float | None = None
        self.countdown_ids: set[int] = set()
        self.cached_tracks: list[Track] = []
        self.cached_faces = []
        self.cached_hands = []
        self.frame_index = 0
        self.fps = 0.0
        self.fps_started = 0.0
        self.fps_frames = 0
        self.started_at = 0.0

        # MediaPipeのオブジェクトはrun()内で生成し、同じスレッドからのみ使う。
        self.person_detector = None
        self.face_detector = None
        self.hand_detector = None

    def run(self) -> None:
        try:
            self._run_camera_loop()
        except BaseException as error:
            self.error = error
        finally:
            self.stop_event.set()

    def _run_camera_loop(self) -> None:
        cv2.setUseOptimized(True)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.person_detector, self.face_detector, self.hand_detector = create_detectors()

        camera = cv2.VideoCapture(self.settings.camera_index)
        if not camera.isOpened():
            camera.release()
            self.face_detector.close()
            self.hand_detector.close()
            raise RuntimeError(
                f"カメラ {self.settings.camera_index} を開けませんでした。"
                "接続状態やカメラ番号を確認してください。"
            )

        self.started_at = time.monotonic()
        self.fps_started = self.started_at
        print("ピースで自動保存 / s: 手動保存 / q または Esc: 終了")

        try:
            while not self.stop_event.is_set():
                success, captured_frame = camera.read()
                if not success:
                    print("カメラから画像を取得できませんでした。")
                    break

                display_frame, raw_frame = self.process_frame(captured_frame)
                cv2.imshow(self.WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    save_frame(raw_frame, self.settings.output_dir, "capture")
                elif key in (ord("q"), 27):
                    break
        finally:
            camera.release()
            if self.face_detector is not None:
                self.face_detector.close()
            if self.hand_detector is not None:
                self.hand_detector.close()
            cv2.destroyAllWindows()

    def process_frame(self, captured_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1フレームの推論、状態更新、描画を行う。"""
        frame = cv2.flip(captured_frame, 1)  # ミラー表示
        raw_frame = frame.copy()
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000)

        (
            active_ids,
            peace_count,
            self.cached_tracks,
            self.cached_faces,
            self.cached_hands,
        ) = detect_and_draw(
            frame,
            self.person_detector,
            self.face_detector,
            self.hand_detector,
            self.tracker,
            self.saved_ids,
            self.settings.person_confidence,
            timestamp_ms,
            self.settings.input_size,
            self.frame_index % self.settings.detection_interval == 0,
            self.cached_tracks,
            self.cached_faces,
            self.cached_hands,
        )
        self.frame_index += 1
        self._update_fps()
        self._update_countdown(frame, raw_frame, active_ids, peace_count)

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return frame, raw_frame

    def _update_fps(self) -> None:
        self.fps_frames += 1
        now = time.monotonic()
        elapsed = now - self.fps_started
        if elapsed >= 0.5:
            self.fps = self.fps_frames / elapsed
            self.fps_frames = 0
            self.fps_started = now

    def _update_countdown(
        self,
        display_frame: np.ndarray,
        raw_frame: np.ndarray,
        active_ids: set[int],
        peace_count: int,
    ) -> None:
        now = time.monotonic()
        unsaved_active_ids = active_ids - self.saved_ids

        if self.countdown_start is None and peace_count > 0 and unsaved_active_ids:
            self.countdown_start = now
            self.countdown_ids = set(unsaved_active_ids)

        if self.countdown_start is None:
            return

        if peace_count == 0 or not (self.countdown_ids & active_ids):
            self.countdown_start = None
            self.countdown_ids.clear()
            return

        remaining_seconds = self.settings.countdown_seconds - (now - self.countdown_start)
        if remaining_seconds > 0:
            draw_countdown(display_frame, remaining_seconds)
            return

        output_path = save_frame(raw_frame, self.settings.output_dir, "peace")
        if output_path is not None:
            self.saved_ids.update(self.countdown_ids)
            self._request_print(output_path, self.countdown_ids)
        self.countdown_start = None
        self.countdown_ids.clear()

    def _request_print(self, image_path: Path, person_ids: set[int]) -> None:
        """保存済み画像の印刷要求をInstaxスレッドへ送る。"""
        job = PrintJob(
            image_path=image_path,
            created_at=datetime.now(),
            person_ids=tuple(sorted(person_ids)),
        )
        self.print_queue.put(job)
        print("Instax印刷要求を送信しました")
        print(f"  image_path: {job.image_path}")
        print(f"  created_at: {job.created_at.isoformat(timespec='milliseconds')}")
        print(f"  person_ids: {job.person_ids}")


def validate_settings(settings: CameraSettings) -> None:
    if settings.countdown_seconds < 0:
        raise ValueError("--countdown-seconds は 0 以上にしてください。")
    if settings.input_size < 160 or settings.input_size % 32 != 0:
        raise ValueError("--input-size は160以上かつ32の倍数にしてください。")
    if settings.detection_interval < 1:
        raise ValueError("--detection-interval は1以上にしてください。")


def main() -> None:
    settings = CameraSettings.from_args(parse_args())
    validate_settings(settings)

    stop_event = threading.Event()
    print_queue: Queue[PrintJob | None] = Queue()
    camera_worker = CameraDetectionWorker(settings, print_queue, stop_event)
    instax_worker = InstaxPrintWorker(print_queue)

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
