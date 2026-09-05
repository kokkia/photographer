import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# simplepybleがMiniconda配下の存在しないDBusソケットを参照する場合があるため、
# LinuxのシステムDBusソケットを明示する。
SYSTEM_DBUS_SOCKET = Path("/run/dbus/system_bus_socket")
if sys.platform.startswith("linux") and SYSTEM_DBUS_SOCKET.exists():
    os.environ["DBUS_SYSTEM_BUS_ADDRESS"] = f"unix:path={SYSTEM_DBUS_SOCKET}"

from InstaxBLE.InstaxBLE import InstaxBLE


def prepare_image_for_instax(
    image_path: Path,
    target_size: tuple[int, int],
) -> BytesIO:
    """Correct orientation and crop an image to the printer's aspect ratio."""
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"Invalid Instax image size: {target_size}")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        source_size = image.size

        source_is_landscape = image.width > image.height
        target_is_landscape = target_size[0] > target_size[1]
        if source_is_landscape != target_is_landscape:
            image = image.transpose(Image.Transpose.ROTATE_270)
            print(f"Rotated image 90 degrees: {source_size} -> {image.size}")

        image = ImageOps.fit(
            image,
            target_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        ).convert("RGB")

        # Pass lossless image data to InstaxBLE; it performs the final JPEG
        # compression needed by the printer.
        prepared = BytesIO()
        image.save(prepared, format="PNG")
        prepared.seek(0)
        print(f"Prepared image for Instax: {image_path} -> {target_size}")
        return prepared


def main() -> None:
    print("Starting Instax test...")
    instax = InstaxBLE(device_address="FA:AB:BC:8D:18:8B", verbose=True)
    print("Instax device initialized.")
    try:
        print("Searching for the Instax printer (timeout: 10 seconds)...")
        instax.connect(timeout=10)
        if instax.peripheral is None or not instax.peripheral.is_connected():
            print("Instax printer was not found or could not be connected.")
            return

        print("Instax printer connected.")
        instax.enable_printing()  # allow printing
        image_path = Path("test/captured_images/test_image.jpg")
        prepared_image = prepare_image_for_instax(image_path, instax.imageSize)
        instax.print_image(prepared_image)
        instax.wait_one_minute()  # hacky way of preventing disconnecting too soon
    except KeyboardInterrupt:
        print("\nInstax test cancelled.")
    finally:
        instax.disconnect()  # all done, disconnect


if __name__ == "__main__":
    main()
