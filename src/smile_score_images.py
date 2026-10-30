import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp


ROOT_DIR = Path(__file__).resolve().parents[1]
FACE_LANDMARKER_MODEL = ROOT_DIR / "models" / "face_landmarker.task"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_COLOR = (255, 120, 0)
SMILE_COLOR = (0, 220, 255)
NO_FACE_COLOR = (80, 80, 255)
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14
FACE_LEFT_INDEX = 234
FACE_RIGHT_INDEX = 454


@dataclass
class SmileResult:
    score: float
    box: tuple[int, int, int, int]
    mouth_left: tuple[int, int]
    mouth_right: tuple[int, int]
    smile_left: float
    smile_right: float
    mouth_width_score: float
    mouth_lift_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="captured_images内の顔から笑顔スコアを計算し、点数入り画像として保存します。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("captured_images"),
        help="入力画像ディレクトリ（デフォルト: captured_images）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("smile_scored_images"),
        help="スコア付き画像の保存先（デフォルト: smile_scored_images）",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=5,
        help="1枚の画像から検出する顔の最大数（デフォルト: 5）",
    )
    parser.add_argument(
        "--smile-low",
        type=float,
        default=0.02,
        help="0点に近い笑顔係数（デフォルト: 0.02）",
    )
    parser.add_argument(
        "--smile-high",
        type=float,
        default=0.90,
        help="強い笑顔として扱う笑顔係数（デフォルト: 0.90）",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=1.0,
        help="顔検出できた画像の最低点（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--score-ceiling",
        type=float,
        default=99.0,
        help="顔検出できた画像の最高点（デフォルト: 99.0）",
    )
    return parser.parse_args()


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return clamp((value - low) / (high - low))


def compress_high_scores(
    value: float,
    threshold: float = 0.75,
    compressed_ceiling: float = 0.94,
    power: float = 1.7,
) -> float:
    if value <= threshold:
        return value
    input_range = 1.0 - threshold
    output_range = compressed_ceiling - threshold
    compressed = ((value - threshold) / input_range) ** power
    return threshold + compressed * output_range


def clipped_box(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = max(x, 0)
    top = max(y, 0)
    right = min(x + width, image_width)
    bottom = min(y + height, image_height)
    return left, top, max(right - left, 0), max(bottom - top, 0)


def landmark_point(landmarks, index: int, image_width: int, image_height: int) -> tuple[int, int]:
    landmark = landmarks[index]
    return int(landmark.x * image_width), int(landmark.y * image_height)


def landmark_xy(landmarks, index: int, image_width: int, image_height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x * image_width, landmark.y * image_height


def point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def landmark_box(landmarks, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    xs = [int(landmark.x * image_width) for landmark in landmarks]
    ys = [int(landmark.y * image_height) for landmark in landmarks]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    padding_x = int((right - left) * 0.08)
    padding_y = int((bottom - top) * 0.08)
    return clipped_box(
        left - padding_x,
        top - padding_y,
        (right - left) + padding_x * 2,
        (bottom - top) + padding_y * 2,
        image_width,
        image_height,
    )


def blendshape_score(blendshapes, name: str) -> float:
    for category in blendshapes:
        if category.category_name == name:
            return float(category.score)
    return 0.0


def calculate_smile_score(
    landmarks,
    blendshapes,
    image_width: int,
    image_height: int,
    smile_low: float,
    smile_high: float,
    score_floor: float,
    score_ceiling: float,
) -> SmileResult:
    smile_left = blendshape_score(blendshapes, "mouthSmileLeft")
    smile_right = blendshape_score(blendshapes, "mouthSmileRight")
    smile_average = (smile_left + smile_right) / 2
    symmetry_penalty = abs(smile_left - smile_right) * 0.18

    mouth_left = landmark_xy(landmarks, MOUTH_LEFT_INDEX, image_width, image_height)
    mouth_right = landmark_xy(landmarks, MOUTH_RIGHT_INDEX, image_width, image_height)
    upper_lip = landmark_xy(landmarks, UPPER_LIP_INDEX, image_width, image_height)
    lower_lip = landmark_xy(landmarks, LOWER_LIP_INDEX, image_width, image_height)
    face_left = landmark_xy(landmarks, FACE_LEFT_INDEX, image_width, image_height)
    face_right = landmark_xy(landmarks, FACE_RIGHT_INDEX, image_width, image_height)

    face_width = max(point_distance(face_left, face_right), 1.0)
    mouth_width_ratio = point_distance(mouth_left, mouth_right) / face_width
    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    mouth_corner_y = (mouth_left[1] + mouth_right[1]) / 2
    mouth_lift_ratio = (mouth_center_y - mouth_corner_y) / face_width

    blend_value = max(smile_average - symmetry_penalty, 0.0)
    blend_score = normalize(math.sqrt(blend_value), math.sqrt(smile_low), math.sqrt(smile_high))
    mouth_width_score = normalize(mouth_width_ratio, 0.33, 0.50)
    mouth_lift_score = normalize(mouth_lift_ratio, -0.03, 0.045)

    combined = blend_score * 0.58 + mouth_lift_score * 0.27 + mouth_width_score * 0.15
    combined = compress_high_scores(combined)
    score = score_floor + combined * (score_ceiling - score_floor)

    return SmileResult(
        score=round(max(score_floor, min(score_ceiling, score)), 1),
        box=landmark_box(landmarks, image_width, image_height),
        mouth_left=landmark_point(landmarks, MOUTH_LEFT_INDEX, image_width, image_height),
        mouth_right=landmark_point(landmarks, MOUTH_RIGHT_INDEX, image_width, image_height),
        smile_left=smile_left,
        smile_right=smile_right,
        mouth_width_score=mouth_width_score,
        mouth_lift_score=mouth_lift_score,
    )


def draw_smile_results(image, results: list[SmileResult]) -> None:
    for index, result in enumerate(results, start=1):
        x, y, width, height = result.box
        cv2.rectangle(image, (x, y), (x + width, y + height), FACE_COLOR, 2)
        cv2.line(image, result.mouth_left, result.mouth_right, SMILE_COLOR, 3)
        cv2.circle(image, result.mouth_left, 4, SMILE_COLOR, -1)
        cv2.circle(image, result.mouth_right, 4, SMILE_COLOR, -1)

        label = f"Smile {result.score:04.1f}"
        if len(results) > 1:
            label = f"Face {index} {label}"
        label_y = max(y - 10, 25)
        cv2.putText(
            image,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            SMILE_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_no_face(image) -> None:
    cv2.putText(
        image,
        "No face: Smile 000",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        NO_FACE_COLOR,
        2,
        cv2.LINE_AA,
    )


def format_score_for_filename(score: float) -> str:
    return f"{score:05.1f}".replace(".", "_")


def output_path_for(input_path: Path, output_dir: Path, score: float, face_found: bool) -> Path:
    face_part = "face" if face_found else "noface"
    score_text = format_score_for_filename(score)
    base_name = f"smile_{score_text}_{face_part}_{input_path.stem}{input_path.suffix.lower()}"
    output_path = output_dir / base_name
    counter = 2
    while output_path.exists():
        output_path = output_dir / (
            f"smile_{score_text}_{face_part}_{input_path.stem}_{counter}{input_path.suffix.lower()}"
        )
        counter += 1
    return output_path


def create_face_landmarker(max_faces: int):
    if not FACE_LANDMARKER_MODEL.is_file():
        raise FileNotFoundError(f"顔ランドマークモデルがありません: {FACE_LANDMARKER_MODEL}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def score_image(
    input_path: Path,
    output_dir: Path,
    face_landmarker,
    smile_low: float,
    smile_high: float,
    score_floor: float,
    score_ceiling: float,
) -> bool:
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"読み込みに失敗しました: {input_path}")
        return False

    height, width = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    face_result = face_landmarker.detect(mp_image)
    results = [
        calculate_smile_score(
            landmarks,
            blendshapes,
            width,
            height,
            smile_low,
            smile_high,
            score_floor,
            score_ceiling,
        )
        for landmarks, blendshapes in zip(
            face_result.face_landmarks,
            face_result.face_blendshapes,
        )
    ]

    if results:
        draw_smile_results(image, results)
        image_score = max(result.score for result in results)
    else:
        draw_no_face(image)
        image_score = 0

    output_path = output_path_for(input_path, output_dir, image_score, bool(results))
    if cv2.imwrite(str(output_path), image):
        print(f"{input_path.name}: {image_score:04.1f} -> {output_path}")
        return True

    print(f"保存に失敗しました: {output_path}")
    return False


def iter_images(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"入力ディレクトリがありません: {args.input_dir}")
    if args.smile_low < 0:
        raise ValueError("--smile-low は 0 以上にしてください。")
    if args.smile_high <= args.smile_low:
        raise ValueError("--smile-high は --smile-low より大きい値にしてください。")
    if not 0 <= args.score_floor < args.score_ceiling <= 100:
        raise ValueError("--score-floor と --score-ceiling は 0以上100以下、かつ floor < ceiling にしてください。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_images(args.input_dir))
    if not image_paths:
        print(f"画像が見つかりませんでした: {args.input_dir}")
        return

    face_landmarker = create_face_landmarker(args.max_faces)
    try:
        success_count = 0
        for input_path in image_paths:
            if score_image(
                input_path,
                args.output_dir,
                face_landmarker,
                args.smile_low,
                args.smile_high,
                args.score_floor,
                args.score_ceiling,
            ):
                success_count += 1
    finally:
        face_landmarker.close()

    print(f"完了: {success_count}/{len(image_paths)} 件を保存しました。")


if __name__ == "__main__":
    main()
