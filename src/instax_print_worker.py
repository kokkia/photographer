import os
import sys
import threading
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from time import monotonic

from PIL import Image, ImageOps

if __package__:
    from .job import PrintJob
else:  # python src/main.py で直接実行する場合
    from job import PrintJob

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# SimplePyBLEが存在しないMiniconda側のDBusソケットを参照する場合があるため、
# InstaxBLEをimportする前にLinuxのシステムDBusソケットを明示する。
SYSTEM_DBUS_SOCKET = Path("/run/dbus/system_bus_socket")
if sys.platform.startswith("linux") and SYSTEM_DBUS_SOCKET.exists():
    os.environ["DBUS_SYSTEM_BUS_ADDRESS"] = f"unix:path={SYSTEM_DBUS_SOCKET}"

from InstaxBLE.InstaxBLE import InstaxBLE


class InstaxPrintWorker(threading.Thread):
    """印刷要求を順番に処理し、Instaxプリンターへ送信する。"""

    CONNECTION_CHECK_INTERVAL = 5.0

    def __init__(
        self,
        print_queue: Queue[PrintJob | None],
        device_address: str | None = None,
        connect_timeout: int = 10,
        verbose: bool = False,
        print_enabled: bool = True,
    ) -> None:
        super().__init__(name="instax-print-worker")
        self.print_queue = print_queue
        self.device_address = device_address
        if connect_timeout < 0:
            raise ValueError("Instaxの接続タイムアウトは0以上にしてください。")
        self.connect_timeout = connect_timeout
        self.verbose = verbose
        self.print_enabled = print_enabled
        self.instax: InstaxBLE | None = None
        self.error: Exception | None = None
        self._connected = threading.Event()

    @property
    def is_connected(self) -> bool:
        """ワーカーが最後に確認した接続状態（GUIから安全に参照可能）。"""
        return self._connected.is_set()

    @property
    def remaining_photos(self) -> int | None:
        """最後に受信した残り枚数。未取得・切断時はNone。通信は行わない。"""
        instax = self.instax
        if not self.is_connected or instax is None:
            return None
        return instax.remaining_photos

    @staticmethod
    def prepare_image_for_instax(
        image_path: Path,
        target_size: tuple[int, int],
    ) -> BytesIO:
        """向きを補正し、縦横比を維持してプリンター用サイズへ変換する。"""
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"Instaxの画像サイズが不正です: {target_size}")
        if not image_path.is_file():
            raise FileNotFoundError(f"印刷する画像がありません: {image_path}")

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            source_size = image.size

            source_is_landscape = image.width > image.height
            target_is_landscape = target_size[0] > target_size[1]
            if source_is_landscape != target_is_landscape:
                image = image.transpose(Image.Transpose.ROTATE_270)
                print(f"画像を90度回転しました: {source_size} -> {image.size}")

            image = ImageOps.fit(
                image,
                target_size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            ).convert("RGB")

            prepared = BytesIO()
            image.save(prepared, format="PNG")
            prepared.seek(0)
            print(f"Instax用画像を準備しました: {image_path} -> {target_size}")
            return prepared

    def run(self) -> None:
        try:
            while True:
                try:
                    job = self.print_queue.get(timeout=self.CONNECTION_CHECK_INTERVAL)
                except Empty:
                    # Bluetooth操作は印刷と同じスレッドで行い、競合を避ける。
                    try:
                        instax = self._ensure_connected(timeout=self.connect_timeout or 10)
                        # 印刷データ送信中や応答待ちには問い合わせを重ねない。
                        if not instax.packetsForPrinting and not instax.waitingForResponse:
                            instax.get_printer_status()
                    except Exception as error:
                        self.error = error
                        print(f"Instax接続の監視・再接続に失敗しました: {type(error).__name__}: {error}")
                        self._disconnect()
                    continue
                try:
                    if job is None:
                        return
                    self.print_image(job)
                except Exception as error:
                    self.error = error
                    print(f"Instax印刷に失敗しました: {type(error).__name__}: {error}")
                    self._disconnect()
                finally:
                    self.print_queue.task_done()
        finally:
            self._disconnect()

    def print_image(self, job: PrintJob) -> None:
        """画像をプリンターに合わせて変換し、実際の印刷命令を送る。"""
        print("Instax印刷を開始します")
        print(f"  image_path: {job.image_path}")
        print(f"  created_at: {job.created_at.isoformat(timespec='milliseconds')}")
        print(f"  person_ids: {job.person_ids}")

        started_at = monotonic()
        instax = self._ensure_connected()
        connected_at = monotonic()
        print(f"Instax接続確認: {connected_at - started_at:.2f}秒")
        prepared_image = self.prepare_image_for_instax(job.image_path, instax.imageSize)
        prepared_at = monotonic()
        print(f"Instax画像準備: {prepared_at - connected_at:.2f}秒")
        instax.print_image(prepared_image)
        print(f"Instax転送開始まで: {monotonic() - started_at:.2f}秒（転送は通知スレッドで継続）")
        # 固定待機は次のジョブを抑制するためのもの。現在の転送・印刷は並行して進む。
        instax.wait_five_seconds()
        print(f"Instax転送開始後の5秒待機が終了しました: {job.image_path}")

    def _ensure_connected(self, timeout: int | None = None) -> InstaxBLE:
        if self._is_connected():
            assert self.instax is not None
            self._connected.set()
            return self.instax

        self._disconnect()
        address = self.device_address or "自動検索"
        print(f"Instaxプリンターを検索します: {address}")
        try:
            self.instax = InstaxBLE(
                device_address=self.device_address,
                verbose=self.verbose,
            )
        except SystemExit as error:
            raise RuntimeError("Bluetoothアダプターが見つかりません。") from error

        self.instax.connect(timeout=self.connect_timeout if timeout is None else timeout)
        if not self._is_connected():
            raise ConnectionError("Instaxプリンターへ接続できませんでした。")

        if self.print_enabled:
            self.instax.enable_printing()
        else:
            print("Instaxの実印刷は無効です（ドライラン）")
        self._connected.set()
        print("Instaxプリンターへ接続しました")
        return self.instax

    def _is_connected(self) -> bool:
        return bool(
            self.instax is not None
            and self.instax.peripheral is not None
            and self.instax.peripheral.is_connected()
        )

    def _disconnect(self) -> None:
        self._connected.clear()
        if self.instax is None:
            return
        try:
            self.instax.disconnect()
        except Exception as error:
            print(f"Instaxの切断中にエラーが発生しました: {error}")
        finally:
            self.instax = None
