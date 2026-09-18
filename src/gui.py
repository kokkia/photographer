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
    """認識映像と撮影操作を提供するメインウィンドウ。"""

    def __init__(
        self,
        camera_worker: CameraDetectionWorker,
        instax_worker: InstaxPrintWorker,
        print_queue: Queue[PrintJob | None],
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self.camera_worker = camera_worker
        self.instax_worker = instax_worker
        self.print_queue = print_queue
        self.stop_event = stop_event
        self.shutting_down = False
        self.worker_error_shown = False

        self.setWindowTitle("Photographer")
        self.resize(1100, 850)

        self.preview = QLabel("カメラを起動しています...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 480)
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

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_preview)
        self.frame_timer.start(30)

    def capture(self) -> None:
        if self.camera_worker.is_alive():
            self.camera_worker.request_capture()
            self.status.setText("撮影を要求しました")

    def update_preview(self) -> None:
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

        if not self.camera_worker.is_alive() and not self.worker_error_shown:
            self.worker_error_shown = True
            self.capture_button.setEnabled(False)
            if self.camera_worker.error is None:
                message = "カメラ処理が終了しました"
            else:
                message = f"カメラエラー: {self.camera_worker.error}"
            self.preview.setPixmap(QPixmap())
            self.preview.setText(message)
            self.status.setText(message)

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
    )
    instax_worker = InstaxPrintWorker(
        print_queue,
        device_address=args.instax_address,
        connect_timeout=args.instax_connect_timeout,
        verbose=args.instax_verbose,
    )
    window = CameraWindow(camera_worker, instax_worker, print_queue, stop_event)

    instax_worker.start()
    camera_worker.start()
    window.show()
    exit_code = app.exec()
    window.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
