import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# simplepybleがMiniconda配下の存在しないDBusソケットを参照する場合があるため、
# LinuxのシステムDBusソケットを明示する。
SYSTEM_DBUS_SOCKET = Path("/run/dbus/system_bus_socket")
if sys.platform.startswith("linux") and SYSTEM_DBUS_SOCKET.exists():
    os.environ["DBUS_SYSTEM_BUS_ADDRESS"] = f"unix:path={SYSTEM_DBUS_SOCKET}"

from InstaxBLE.InstaxBLE import InstaxBLE


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
        instax.print_image("test/captured_images/test_image.jpg")  # print image
        instax.wait_one_minute()  # hacky way of preventing disconnecting too soon
    except KeyboardInterrupt:
        print("\nInstax test cancelled.")
    finally:
        instax.disconnect()  # all done, disconnect


if __name__ == "__main__":
    main()
