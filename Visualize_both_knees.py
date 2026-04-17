import matplotlib
import time
matplotlib.use("TkAgg")
import asyncio, struct, collections, threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from bleak import BleakScanner, BleakClient

CHAR_UUID   = "19B10001-E8F2-537E-4F6C-D104768A1214"
WINDOW_S    = 30
MAX_POINTS  = WINDOW_S * 100

DEVICES = {
    "L": {
        "name":      "NanoBLE L1",
        "color":     "#185FA5",
        "times":     collections.deque(maxlen=MAX_POINTS),
        "values":    collections.deque(maxlen=MAX_POINTS),
        "t0":        None,
        "connected": threading.Event(),
    },
    "R": {
        "name":      "NanoBLE L3",
        "color":     "#993C1D",
        "times":     collections.deque(maxlen=MAX_POINTS),
        "values":    collections.deque(maxlen=MAX_POINTS),
        "t0":        None,
        "connected": threading.Event(),
    },
}

paused = threading.Event()

def make_handler(dev):
    def handle(sender, data: bytearray):
        if paused.is_set():
            return
        now = time.time()  # laptop wall clock — same for both devices
        for i in range(8):
            offset = 4 + i * 3
            raw    = struct.unpack_from("<H", data, offset)[0]
            ohms   = raw / 100.0
            t_sec = (now - t_start) + i * 0.01
            dev["times"].append(t_sec)
            dev["values"].append(ohms)
    return handle

async def connect_device(dev):
    print(f"Scanning for {dev['name']} ...")
    device = await BleakScanner.find_device_by_name(dev["name"], timeout=30)
    if not device:
        print(f"{dev['name']} not found.")
        return
    async with BleakClient(device) as client:
        dev["connected"].set()
        print(f"Connected to {dev['name']}")
        await client.start_notify(CHAR_UUID, make_handler(dev))
        while client.is_connected:
            await asyncio.sleep(0.1)
        dev["connected"].clear()
        print(f"Disconnected from {dev['name']}")

async def ble_main():
    await asyncio.gather(*(connect_device(d) for d in DEVICES.values()))

threading.Thread(target=lambda: asyncio.run(ble_main()), daemon=True).start()

t_start = time.time()
fig = plt.figure(figsize=(13, 5))
fig.patch.set_facecolor("#fafafa")
gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

ax_main = fig.add_subplot(gs[0])
ax_stat = fig.add_subplot(gs[1])
ax_main.set_facecolor("#f5f5f5")
ax_main.set_xlabel("Time (s)", fontsize=10)
ax_main.set_ylabel("Resistance (Ω)", fontsize=10)
ax_main.grid(True, alpha=0.4, linewidth=0.5)
ax_main.tick_params(labelsize=9)
title_txt = ax_main.set_title("Waiting for devices...", fontsize=10, loc="left")

lines = {}
fills = {}
for key, dev in DEVICES.items():
    lines[key], = ax_main.plot([], [], lw=1.2, color=dev["color"],
                                label=dev["name"])
    fills[key]  = ax_main.fill_between([], [], alpha=0.07, color=dev["color"])

ax_main.legend(loc="upper left", fontsize=9)

ax_stat.axis("off")
stat_rows  = ["Current", "Min", "Max", "Mean"]
stat_texts = {}
for col, (key, dev) in enumerate(DEVICES.items()):
    x = 0.05 + col * 0.48
    ax_stat.text(x, 0.97, dev["name"].split()[-1],
                 transform=ax_stat.transAxes,
                 fontsize=10, fontweight="bold", color=dev["color"])
    stat_texts[key] = []
    for row, lbl in enumerate(stat_rows):
        y = 0.85 - row * 0.2
        ax_stat.text(x, y, lbl, transform=ax_stat.transAxes,
                     fontsize=8, color="#888")
        t = ax_stat.text(x, y - 0.08, "—", transform=ax_stat.transAxes,
                         fontsize=12, fontweight="bold", color="#1a1a1a")
        stat_texts[key].append(t)

pause_ax  = fig.add_axes([0.91, 0.01, 0.08, 0.05])
pause_btn = plt.Button(pause_ax, "Pause", color="#e8e8e8", hovercolor="#d0d0d0")

def toggle_pause(event):
    if paused.is_set():
        paused.clear(); pause_btn.label.set_text("Pause")
    else:
        paused.set();   pause_btn.label.set_text("Resume")
    fig.canvas.draw_idle()

pause_btn.on_clicked(toggle_pause)

def update(frame):
    all_xs = []
    statuses = []
    for key, dev in DEVICES.items():
        xs = list(dev["times"])
        ys = list(dev["values"])
        if xs:
            all_xs.extend(xs)
        status = "LIVE" if dev["connected"].is_set() else "—"
        statuses.append(f"{dev['name'].split()[-1]}: {status}")
        lines[key].set_data(xs, ys)
        fills[key].remove()
        fills[key] = ax_main.fill_between(xs, ys, alpha=0.07, color=dev["color"])
        if ys:
            arr = np.array(ys)
            stat_texts[key][0].set_text(f"{ys[-1]:.1f} Ω")
            stat_texts[key][1].set_text(f"{arr.min():.1f} Ω")
            stat_texts[key][2].set_text(f"{arr.max():.1f} Ω")
            stat_texts[key][3].set_text(f"{arr.mean():.1f} Ω")

    pause_str = "  [PAUSED]" if paused.is_set() else ""
    title_txt.set_text("  ·  ".join(statuses) + pause_str)

    if all_xs:
        x_end = max(all_xs)
        now = time.time()
        ax_main.set_xlim(now - t_start - WINDOW_S, now - t_start + 0.5)
        all_ys = []
        for dev in DEVICES.values():
            all_ys.extend(dev["values"])
        if all_ys:
            lo, hi = min(all_ys), max(all_ys)
            pad = (hi - lo) * 0.2 or 2.0
            ax_main.set_ylim(lo - pad, hi + pad)

ani = animation.FuncAnimation(fig, update, interval=10,
                               blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()