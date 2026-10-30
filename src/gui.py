import ctypes
import os
import sys
import threading
from pathlib import Path
from queue import Queue

import cv2
from PySide6.QtCore import QLibraryInfo, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent, QColor, QImage, QImageReader, QKeySequence,
    QLinearGradient, QPainter, QPixmap, QShortcut,
)
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
NATURAL_STYLE = """
QMainWindow, QStackedWidget, QWidget#page { background: #fbf7f2; }
QWidget {
    color: #6c5547;
    font-family: "Noto Sans CJK JP", "Yu Gothic", sans-serif;
    font-size: 16px; font-weight: 300;
}
QLabel { background: transparent; }
QLabel#boothTitle {
    font-family: "Noto Serif CJK JP", "Yu Mincho", "Georgia", serif;
    color: #625044; font-size: 50px; font-weight: 300;
}
QLabel#coverIntro { color: #725e50; font-size: 15px; }
QLabel#coverStatus {
    color: #625044; background: rgba(255, 250, 244, 225);
    border-radius: 6px; padding: 10px 14px; font-size: 14px;
}
QPushButton#coverButton {
    background: rgba(255, 250, 244, 235); color: #6c5547;
    border: 1px solid #c7b6a4; border-radius: 8px;
}
QPushButton#coverButton:hover { background: #fffaf4; border-color: #987d68; }
QPushButton#coverButton:pressed { background: #e9dfd4; }
QPushButton#coverButton:focus { border: 2px solid #987d68; }
QPushButton#coverButton:disabled { background: #e9dfd4; color: #a89583; }
QLabel#title {
    font-family: "Noto Serif CJK JP", "Yu Mincho", serif;
    font-size: 28px; font-weight: 300;
}
QLabel#intro { color: #8d7868; font-size: 18px; }
QLabel#status { color: #8d7868; font-size: 14px; padding: 4px 0; }
QLabel#image {
    background: #fffcf8; color: #8d7868;
    border: 1px solid #ead8cd; border-radius: 4px; padding: 8px;
}
QPushButton {
    background: #80624e; color: #fffaf4; border: 1px solid #80624e;
    border-radius: 6px; padding: 14px 20px; font-size: 18px; font-weight: 400;
}
QPushButton:hover { background: #71533f; border-color: #71533f; }
QPushButton:pressed { background: #614531; border-color: #614531; }
QPushButton:focus { border: 2px solid #cba99a; }
QPushButton[secondary="true"] { background: #fcf5ef; color: #7b604c; border-color: #ddc5b7; }
QPushButton[secondary="true"]:hover { background: #f5e7df; }
QPushButton[secondary="true"]:pressed { background: #ebd5ca; }
QPushButton:disabled { background: #e9dfd4; color: #a89583; border-color: #e9dfd4; }
QStatusBar { background: #f4ece3; border-top: 1px solid #e9dbcc; }
QStatusBar::item { border: none; }
QLabel#bluetoothStatus { font-size: 14px; color: #8d7868; padding: 4px 16px; }
QLabel#bluetoothStatus[connected="true"] { color: #78614e; }
"""


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


class PhotoCoverPage(QWidget):
    """写真の縦横比を保ち、画面の中央を基準に背景いっぱいに描く。"""

    def __init__(self, pixmap: QPixmap) -> None:
        super().__init__()
        self.pixmap = pixmap

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fbf7f2"))
        if self.pixmap.isNull():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scale = max(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        width, height = self.width() / scale, self.height() / scale
        source = QRectF(
            (self.pixmap.width() - width) / 2,
            (self.pixmap.height() - height) / 2,
            width, height,
        )
        painter.drawPixmap(QRectF(self.rect()), self.pixmap, source)
        # 空の明るさをわずかに整え、文字の読みやすさを保つ。
        wash = QLinearGradient(0, 0, 0, self.height() * 0.5)
        wash.setColorAt(0, QColor(255, 250, 244, 65))
        wash.setColorAt(1, QColor(255, 250, 244, 0))
        painter.fillRect(self.rect(), wash)


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
        self.start_image_path = start_image_path or Path(__file__).with_name("wedding.JPG")
        self.start_image_pixmap = QPixmap()
        self.review_pixmap = QPixmap()
        self.pending_print_job: PrintJob | None = None

        self.setWindowTitle("Photographer")
        self.resize(600, 1024)
        self.setMinimumSize(540, 800)
        self.setStyleSheet(NATURAL_STYLE)
        self.min_image_height = 360
        self.min_image_width = 360

        self.pages = QStackedWidget()
        self.start_page = self._create_start_page()
        self.camera_page = self._create_camera_page()
        self.review_page = self._create_review_page()
        self.pages.addWidget(self.start_page)
        self.pages.addWidget(self.camera_page)
        self.pages.addWidget(self.review_page)
        self.setCentralWidget(self.pages)

        for key in ("Q", "Esc"):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(self.close)

        self.bluetooth_status = QLabel("Bluetooth: 未接続  |  残り: --枚")
        self.bluetooth_status.setObjectName("bluetoothStatus")
        self.bluetooth_status.setProperty("connected", False)
        self.bluetooth_status.setToolTip("接続状態と最後に取得したフィルム枚数（待機中は5秒ごとに確認）")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setFixedHeight(44)
        self.statusBar().addPermanentWidget(self.bluetooth_status, 1)
        self._bluetooth_connected = False
        self._remaining_photos = None

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_preview)
        self.frame_timer.start(30)

    def _create_start_page(self) -> QWidget:
        self._load_start_image()
        page = PhotoCoverPage(self.start_image_pixmap)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 100, 36, 20)
        layout.setSpacing(0)

        title = QLabel("Photo Booth")
        title.setObjectName("boothTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setMinimumHeight(76)
        layout.addWidget(title)
        layout.addSpacing(8)

        intro = QLabel("今日の思い出を一枚に")
        intro.setObjectName("coverIntro")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(intro)

        layout.addStretch(1)

        self.start_button = QPushButton("撮影をはじめる")
        self.start_button.setObjectName("coverButton")
        self.start_button.setMinimumSize(300, 64)
        self.start_button.clicked.connect(self.start_capture_session)
        layout.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(20)

        # 通常は空欄にし、印刷開始やエラーなど必要な通知だけを表示する。
        self.start_status = QLabel("")
        self.start_status.setObjectName("coverStatus")
        self.start_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_status.setWordWrap(True)
        self.start_status.setVisible(False)
        layout.addWidget(self.start_status)
        return page

    def _create_page_layout(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 40, 32, 28)
        layout.setSpacing(16)
        return page, layout

    def _add_heading(self, layout: QVBoxLayout, title: str, subtitle: str) -> None:
        for text, name in ((title, "title"), (subtitle, "intro")):
            label = QLabel(text)
            label.setObjectName(name)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def _create_camera_page(self) -> QWidget:
        page, layout = self._create_page_layout()
        self._add_heading(layout, "笑顔で！はいピース！", "ピースを認識すると自動で撮影します")
        self.preview = QLabel("カメラを起動しています...")
        self.preview.setObjectName("image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(self.min_image_width, self.min_image_height)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.status = QLabel("準備中")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.capture_button = QPushButton("手動で撮影")
        self.capture_button.setMinimumSize(160, 52)
        self.capture_button.clicked.connect(self.capture)

        self.back_button = QPushButton("戻る")
        self.back_button.setProperty("secondary", True)
        self.back_button.setMinimumSize(120, 52)
        self.back_button.clicked.connect(self.cancel_capture_session)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.capture_button, 1)

        layout.addWidget(self.preview, 1)
        layout.addWidget(self.status)
        layout.addLayout(button_layout)
        return page

    def _create_review_page(self) -> QWidget:
        page, layout = self._create_page_layout()
        self._add_heading(layout, "お気に入りの一枚をチェキに！", "写真を確認してよろしければ印刷してください")

        self.review_image = QLabel("撮影画像を読み込んでいます...")
        self.review_image.setObjectName("image")
        self.review_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.review_image.setMinimumSize(self.min_image_width, self.min_image_height)
        self.review_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.review_status = QLabel("この写真を印刷しますか？")
        self.review_status.setObjectName("status")
        self.review_status.setWordWrap(True)
        self.review_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        print_button = QPushButton("印刷する")
        print_button.setMinimumHeight(60)
        print_button.clicked.connect(self.print_captured_photo)

        skip_button = QPushButton("撮り直す")
        skip_button.setProperty("secondary", True)
        skip_button.setMinimumHeight(60)
        skip_button.clicked.connect(self.skip_printing)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addWidget(skip_button, 1)
        button_layout.addWidget(print_button, 1)

        layout.addWidget(self.review_image, 1)
        layout.addWidget(self.review_status)
        layout.addLayout(button_layout)
        return page

    def _load_start_image(self) -> None:
        reader = QImageReader(str(self.start_image_path))
        reader.setAutoTransform(True)
        self.start_image_pixmap = QPixmap.fromImage(reader.read())

    def _set_start_status(self, message: str) -> None:
        self.start_status.setText(message)
        self.start_status.setVisible(bool(message))

    def _scale_review_image(self) -> None:
        if self.review_pixmap.isNull():
            return
        self.review_image.setPixmap(
            self.review_pixmap.scaled(
                self.review_image.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_review_image()

    def start_capture_session(self) -> None:
        if not self.camera_worker.is_alive():
            self._set_start_status("カメラを開始できませんでした")
            return
        self.preview.setPixmap(QPixmap())
        self.preview.setText("カメラを準備しています...")
        self.status.setText("準備中")
        self._set_start_status("")
        self.camera_worker.start_capture_session()
        self.pages.setCurrentWidget(self.camera_page)

    def cancel_capture_session(self) -> None:
        self.camera_worker.stop_capture_session()
        self.pending_print_job = None
        self._return_to_start("")

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
        self._set_start_status(message)

    def update_preview(self) -> None:
        connected = self.instax_worker.is_connected
        remaining = self.instax_worker.remaining_photos if connected else None
        if connected != self._bluetooth_connected or remaining != self._remaining_photos:
            self._bluetooth_connected = connected
            self._remaining_photos = remaining
            connection_text = "接続済み" if connected else "未接続"
            remaining_text = "--" if remaining is None else str(remaining)
            self.bluetooth_status.setText(f"Bluetooth: {connection_text}  |  残り: {remaining_text}枚")
            self.bluetooth_status.setProperty("connected", connected)
            self.bluetooth_status.style().unpolish(self.bluetooth_status)
            self.bluetooth_status.style().polish(self.bluetooth_status)

        completed_capture = self.camera_worker.consume_completed_capture()
        if completed_capture is not None and self.pages.currentWidget() is self.camera_page:
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
                self.preview.contentsRect().size(),
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
            self._set_start_status(message)

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
    # 600×1024の実画面では、ウィンドウ枠やデスクトップのバーの分も考慮する。
    available = window.screen().availableGeometry()
    if available.width() <= window.width() or available.height() <= window.height():
        window.showMaximized()
    else:
        window.show()
    exit_code = app.exec()
    window.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
