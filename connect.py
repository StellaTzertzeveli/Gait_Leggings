import asyncio
from bleak import BleakScanner, BleakClient
import struct

DEVICE_NAME = "NanoBLE DEMO R"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

def handle_notification(sender, data: bytearray):
    timestamp = struct.unpack_from('<I', data, 0)[0]
    print(f"Timestamp: {timestamp} ms")
    for i in range(8):
        offset = 4 + i * 3
        resistance_scaled = struct.unpack_from('<H', data, offset)[0]
        delta_us = data[offset + 2]
        resistance_ohm = resistance_scaled / 100.0
        print(f"  Sample {i}: {resistance_ohm:.2f} Ω  (Δt = {delta_us} µs)")

async def main():
    print("Scanning for device...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=20)
    if not device:
        print("Device not found. Is the board powered and advertising?")
        return

    async with BleakClient(device) as client:
        print(f"Connected to {device.name}")
        await client.start_notify(CHAR_UUID, handle_notification)
        print("Receiving data — press Ctrl+C to stop")
        await asyncio.sleep(30)
        await client.stop_notify(CHAR_UUID)

asyncio.run(main())