import ctypes
import os
import sys
import threading
from pathlib import Path
from queue import Queue

import cv2
from PySide6.QtCore import QLibraryInfo, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .camera_detection_worker import CameraDetectionWorker, CameraSettings
    from .instax_print_worker import InstaxPrintWorker
    from .job import PrintJob
    from .main import parse_args, validate_settings
else:  # python src/gui.py で直接実行する場合
    from camera_detection_worker import CameraDetectionWorker, CameraSettings
    from instax_print_worker import InstaxPrintWorker
    from job import PrintJob
    from main import parse_args, validate_settings


_xcb_cursor_library = None
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_START_IMAGE = ROOT_DIR / "assets" / "start_screen.jpg"


def configure_qt_runtime() -> None:
    """PySide6のQtプラグインとConda内のxcb依存ライブラリを使用する。"""
    global _xcb_cursor_library

    xcb_cursor_path = Path(sys.prefix) / "lib" / "libxcb-cursor.so.0"
    if sys.platform.startswith("linux") and xcb_cursor_path.is_file():
        _xcb_cursor_library = ctypes.CDLL(
            str(xcb_cursor_path),
            mode=ctypes.RTLD_GLOBAL,
        )

    plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path


configure_qt_runtime()


class CameraWindow(QMainWindow):
    """スタート画面と認識・撮影画面を提供するメインウィンドウ。"""

    def __init__(
        self,
        camera_worker: CameraDetectionWorker,
        instax_worker: InstaxPrintWorker,
        print_queue: Queue[PrintJob | None],
        stop_event: threading.Event,
        start_image_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.camera_worker = camera_worker
        self.instax_worker = instax_worker
        self.print_queue = print_queue
        self.stop_event = stop_event
        self.shutting_down = False
        self.worker_error_shown = False
        self.start_image_path = start_image_path or DEFAULT_START_IMAGE
        self.start_image_pixmap = QPixmap()
        self.review_pixmap = QPixmap()
        self.pending_print_job: PrintJob | None = None

        self.setWindowTitle("Photographer")
        # self.resize(1100, 850)
        self.resize(600, 1024)
        self.min_image_height = 480
        self.min_image_width = 360

        self.pages = QStackedWidget()
        self.start_page = self._create_start_page()
        self.camera_page = self._create_camera_page()
        self.review_page = self._create_review_page()
        self.pages.addWidget(self.start_page)
        self.pages.addWidget(self.camera_page)
        self.pages.addWidget(self.review_page)
        self.setCentralWidget(self.pages)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_preview)
        self.frame_timer.start(30)

    def _create_start_page(self) -> QWidget:
        self.start_image = QLabel()
        self.start_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.start_image.setMinimumSize(640, 480)
        self.start_image.setMinimumSize(self.min_image_height, self.min_image_width)
        self.start_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.start_image.setStyleSheet(
            "background-color: #f3f0ea; color: #777; font-size: 18px; "
            "border: 2px dashed #c8c0b4; border-radius: 12px;"
        )
        self._load_start_image()

        self.start_status = QLabel("ボタンを押すと、認識と撮影を開始します")
        self.start_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_status.setStyleSheet("font-size: 18px;")

        self.start_button = QPushButton("認識・撮影をスタート")
        self.start_button.setMinimumSize(260, 64)
        self.start_button.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_capture_session)

        start_exit_button = QPushButton("終了")
        start_exit_button.setMinimumSize(140, 52)
        start_exit_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addSpacing(20)
        button_layout.addWidget(start_exit_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.addWidget(self.start_image, 1)
        layout.addSpacing(12)
        layout.addWidget(self.start_status)
        layout.addSpacing(12)
        layout.addLayout(button_layout)

        page = QWidget()
        page.setLayout(layout)
        return page

    def _create_camera_page(self) -> QWidget:
        self.preview = QLabel("カメラを起動しています...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(self.min_image_height, self.min_image_width)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("background-color: #111; color: #ddd; font-size: 18px;")

        self.status = QLabel("準備中")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.capture_button = QPushButton("撮影")
        self.capture_button.setMinimumSize(160, 52)
        self.capture_button.clicked.connect(self.capture)

        self.exit_button = QPushButton("終了")
        self.exit_button.setMinimumSize(160, 52)
        self.exit_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.capture_button)
        button_layout.addSpacing(24)
        button_layout.addWidget(self.exit_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.status)
        layout.addLayout(button_layout)

        page = QWidget()
        page.setLayout(layout)
        return page

    def _create_review_page(self) -> QWidget:
        title = QLabel("撮影した写真")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.review_image = QLabel("撮影画像を読み込んでいます...")
        self.review_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.review_image.setMinimumSize(self.min_image_height, self.min_image_width)
        self.review_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.review_image.setStyleSheet("background-color: #111; color: #ddd; font-size: 18px;")

        self.review_status = QLabel("この写真を印刷しますか？")
        self.review_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.review_status.setStyleSheet("font-size: 18px;")

        print_button = QPushButton("印刷する")
        print_button.setMinimumSize(200, 60)
        print_button.setStyleSheet("font-size: 20px; font-weight: bold;")
        print_button.clicked.connect(self.print_captured_photo)

        skip_button = QPushButton("印刷しない")
        skip_button.setMinimumSize(200, 60)
        skip_button.setStyleSheet("font-size: 20px;")
        skip_button.clicked.connect(self.skip_printing)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(print_button)
        button_layout.addSpacing(24)
        button_layout.addWidget(skip_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(title)
        layout.addWidget(self.review_image, 1)
        layout.addWidget(self.review_status)
        layout.addLayout(button_layout)

        page = QWidget()
        page.setLayout(layout)
        return page

    def _load_start_image(self) -> None:
        if self.start_image_path.is_file():
            self.start_image_pixmap = QPixmap(str(self.start_image_path))
        if self.start_image_pixmap.isNull():
            self.start_image.setText(
                "スタート画面の画像\n\n"
                f"{self.start_image_path}\n"
                "に画像を置くか、--start-image で指定してください"
            )
            return
        self._scale_start_image()

    def _scale_start_image(self) -> None:
        if self.start_image_pixmap.isNull():
            return
        self.start_image.setPixmap(
            self.start_image_pixmap.scaled(
                self.start_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _scale_review_image(self) -> None:
        if self.review_pixmap.isNull():
            return
        self.review_image.setPixmap(
            self.review_pixmap.scaled(
                self.review_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_start_image()
        self._scale_review_image()

    def start_capture_session(self) -> None:
        if not self.camera_worker.is_alive():
            self.start_status.setText("カメラを開始できませんでした")
            return
        self.preview.setPixmap(QPixmap())
        self.preview.setText("カメラを準備しています...")
        self.status.setText("準備中")
        self.start_status.setText("ボタンを押すと、認識と撮影を開始します")
        self.camera_worker.start_capture_session()
        self.pages.setCurrentWidget(self.camera_page)

    def capture(self) -> None:
        if self.camera_worker.is_alive():
            self.camera_worker.request_capture()
            self.status.setText("撮影を要求しました")

    def show_capture_review(self, job: PrintJob) -> None:
        self.pending_print_job = job
        self.review_pixmap = QPixmap(str(job.image_path))
        self.review_image.setPixmap(QPixmap())
        if self.review_pixmap.isNull():
            self.review_image.setText(f"撮影画像を表示できませんでした\n{job.image_path}")
        else:
            self.review_image.setText("")
        self.review_status.setText("この写真を印刷しますか？")
        self.pages.setCurrentWidget(self.review_page)
        QTimer.singleShot(0, self._scale_review_image)

    def print_captured_photo(self) -> None:
        if self.pending_print_job is None:
            return
        job = self.pending_print_job
        self.pending_print_job = None
        self.print_queue.put(job)
        self._return_to_start(f"印刷を開始しました: {job.image_path.name}")

    def skip_printing(self) -> None:
        if self.pending_print_job is None:
            return
        self.pending_print_job = None
        self.review_pixmap = QPixmap()
        self.start_capture_session()

    def _return_to_start(self, message: str) -> None:
        self.review_pixmap = QPixmap()
        self.pages.setCurrentWidget(self.start_page)
        self.start_status.setText(message)

    def update_preview(self) -> None:
        completed_capture = self.camera_worker.consume_completed_capture()
        if completed_capture is not None:
            self.show_capture_review(completed_capture)
            return

        if self.pages.currentWidget() is not self.camera_page:
            self._check_worker_error()
            return

        frame = self.camera_worker.get_latest_display_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            image = QImage(
                rgb_frame.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
            pixmap = QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview.setPixmap(pixmap)
            self.status.setText("認識中")

        self._check_worker_error()

    def _check_worker_error(self) -> None:
        if not self.camera_worker.is_alive() and not self.worker_error_shown:
            self.worker_error_shown = True
            self.capture_button.setEnabled(False)
            self.start_button.setEnabled(False)
            if self.camera_worker.error is None:
                message = "カメラ処理が終了しました"
            else:
                message = f"カメラエラー: {self.camera_worker.error}"
            self.preview.setPixmap(QPixmap())
            self.preview.setText(message)
            self.status.setText(message)
            self.start_status.setText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        if self.shutting_down:
            return
        self.shutting_down = True
        self.frame_timer.stop()
        self.capture_button.setEnabled(False)
        self.stop_event.set()
        self.camera_worker.join()
        self.print_queue.put(None)
        self.instax_worker.join()


def main() -> int:
    args = parse_args()
    settings = CameraSettings.from_args(args)
    validate_settings(settings)

    app = QApplication(sys.argv)
    stop_event = threading.Event()
    print_queue: Queue[PrintJob | None] = Queue()
    camera_worker = CameraDetectionWorker(
        settings,
        print_queue,
        stop_event,
        show_window=False,
        start_paused=True,
    )
    instax_worker = InstaxPrintWorker(
        print_queue,
        device_address=args.instax_address,
        connect_timeout=args.instax_connect_timeout,
        verbose=args.instax_verbose,
    )
    window = CameraWindow(
        camera_worker,
        instax_worker,
        print_queue,
        stop_event,
        start_image_path=args.start_image,
    )

    instax_worker.start()
    camera_worker.start()
    window.show()
    # window.showFullScreen()
    exit_code = app.exec()
    window.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
