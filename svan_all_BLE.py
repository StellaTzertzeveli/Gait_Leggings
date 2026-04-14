import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for all BLE devices...")
    devices = await BleakScanner.discover(timeout=10)
    for d in devices:
        print(d.name, d.address)

asyncio.run(main())