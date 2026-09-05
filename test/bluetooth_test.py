import asyncio
from bleak import BleakScanner, BleakClient


TARGET_NAME = "INSTAX-70615591(BLE)"
TARGET_ADDRESS = "FA:AB:BC:8D:18:8B"


async def main():
    print("Scanning...")

    devices = await BleakScanner.discover(timeout=10.0)

    target = None

    for device in devices:
        print(f"name={device.name}, address={device.address}")

        if device.name == TARGET_NAME:
            target = device
            break

        if device.address == TARGET_ADDRESS:
            target = device
            break

    if target is None:
        print(f"\n{TARGET_NAME} was not found.")
        return

    print("\nINSTAX found!")
    print("name   :", target.name)
    print("address:", target.address)

    print("\nConnecting...")

    try:
        async with BleakClient(target) as client:
            print("connected:", client.is_connected)

            if client.is_connected:
                print("\nBLE connection successful.")

                print("\nServices:")
                for service in client.services:
                    print(f"[Service] {service.uuid}")

                    for char in service.characteristics:
                        print(
                            f"  [Characteristic] {char.uuid} "
                            f"properties={char.properties}"
                        )

    except Exception as e:
        print("\nConnection failed.")
        print(type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())