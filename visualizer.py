import asyncio, struct, collections, threading, time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from bleak import BleakScanner, BleakClient
matplotlib.use("TkAgg")

DEVICE_NAME  = "NanoBLE DEMO R"
CHAR_UUID    = "19B10001-E8F2-537E-4F6C-D104768A1214"
WINDOW_S     = 30
SAMPLE_RATE  = 100
MAX_POINTS   = WINDOW_S * SAMPLE_RATE

times  = collections.deque(maxlen=MAX_POINTS)
values = collections.deque(maxlen=MAX_POINTS)
t0     = None
paused = threading.Event()
connected = threading.Event()

def handle_notification(sender, data: bytearray):
    global t0
    if paused.is_set():
        return
    timestamp_ms = struct.unpack_from("<I", data, 0)[0]
    if t0 is None:
        t0 = timestamp_ms
    for i in range(8):
        offset = 4 + i * 3
        raw   = struct.unpack_from("<H", data, offset)[0]
        ohms  = raw / 100.0
        t_sec = (timestamp_ms - t0) / 1000.0 + i * 0.01
        times.append(t_sec)
        values.append(ohms)

async def ble_loop():
    print("Scanning for NanoBLE DEMO R ...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=20)
    if not device:
        print("Device not found.")
        return
    async with BleakClient(device) as client:
        connected.set()
        print(f"Connected to {device.name}")
        await client.start_notify(CHAR_UUID, handle_notification)
        while client.is_connected:
            await asyncio.sleep(0.1)
        connected.clear()
        print("Disconnected.")

threading.Thread(target=lambda: asyncio.run(ble_loop()), daemon=True).start()

fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor("#fafafa")
gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

ax_main = fig.add_subplot(gs[0])
ax_stat = fig.add_subplot(gs[1])

ax_main.set_facecolor("#f5f5f5")
ax_main.set_xlabel("Time (s)", fontsize=10)
ax_main.set_ylabel("Resistance (Ω)", fontsize=10)
ax_main.grid(True, alpha=0.4, linewidth=0.5)
ax_main.tick_params(labelsize=9)

line,     = ax_main.plot([], [], lw=1.2, color="#185FA5")
fill_poly  = ax_main.fill_between([], [], alpha=0.08, color="#185FA5")

title_txt = ax_main.set_title("Waiting for device...", fontsize=10, loc="left")

ax_stat.axis("off")
stat_labels = ["Current", "Min", "Max", "Mean", "Samples"]
stat_values = ["—"] * 5
stat_texts  = []
for idx, (lbl, val) in enumerate(zip(stat_labels, stat_values)):
    y = 0.92 - idx * 0.18
    ax_stat.text(0.05, y,       lbl, transform=ax_stat.transAxes,
                 fontsize=9, color="#888")
    t = ax_stat.text(0.05, y - 0.07, val, transform=ax_stat.transAxes,
                     fontsize=13, fontweight="bold", color="#1a1a1a")
    stat_texts.append(t)

pause_ax  = fig.add_axes([0.91, 0.01, 0.08, 0.05])
pause_btn = plt.Button(pause_ax, "Pause", color="#e8e8e8", hovercolor="#d0d0d0")

def toggle_pause(event):
    if paused.is_set():
        paused.clear()
        pause_btn.label.set_text("Pause")
    else:
        paused.set()
        pause_btn.label.set_text("Resume")
    fig.canvas.draw_idle()

pause_btn.on_clicked(toggle_pause)

def update(frame):
    global fill_poly
    xs = list(times)
    ys = list(values)

    status = "LIVE" if connected.is_set() else "Disconnected"
    status = "PAUSED" if paused.is_set() else status
    title_txt.set_text(f"NanoBLE DEMO R  ·  {status}")

    if not xs:
        return line, *stat_texts

    line.set_data(xs, ys)

    fill_poly.remove()
    fill_poly = ax_main.fill_between(xs, ys, alpha=0.08, color="#185FA5")

    x_end = xs[-1]
    ax_main.set_xlim(x_end - WINDOW_S, x_end + 0.5)

    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.2 or 2.0
    ax_main.set_ylim(lo - pad, hi + pad)

    arr = np.array(ys)
    stat_texts[0].set_text(f"{ys[-1]:.1f} Ω")
    stat_texts[1].set_text(f"{arr.min():.1f} Ω")
    stat_texts[2].set_text(f"{arr.max():.1f} Ω")
    stat_texts[3].set_text(f"{arr.mean():.1f} Ω")
    stat_texts[4].set_text(f"{len(ys)}")

    return line, *stat_texts

ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()