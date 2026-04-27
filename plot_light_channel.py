import re
import time
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial


# Arduino sketch_apr27b settings
PORT = "COM7"  # автоподставлено по найденному устройству
BAUD = 9600

# Plot settings
MAX_POINTS = 300
UPDATE_MS = 100

CHANNELS = ["L", "F", "X"]  # поменяй на любые из: L, F, X, Y, Z
RES = {k: re.compile(rf"\b{k}:(-?\d+)\b") for k in ["L", "F", "X", "Y", "Z"]}


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Arduino часто перезагружается при открытии порта
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    t_buf = deque(maxlen=MAX_POINTS)
    y_buf = {k: deque(maxlen=MAX_POINTS) for k in CHANNELS}

    fig, ax = plt.subplots()
    lines = {}
    for k in CHANNELS:
        (ln,) = ax.plot([], [], label=k)
        lines[k] = ln
    ax.set_xlabel("t, s")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend()

    t0 = time.time()

    def update(_):
        # читаем весь накопившийся буфер, чтобы не отставать
        while ser.in_waiting:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue

            values = {}
            for k in CHANNELS:
                m = RES[k].search(raw)
                if m:
                    values[k] = int(m.group(1))

            if not values:
                continue

            t = time.time() - t0
            t_buf.append(t)
            for k in CHANNELS:
                if k in values:
                    y_buf[k].append(values[k])
                else:
                    # если в строке нет значения, повторяем последнее (чтобы линии были одной длины)
                    y_buf[k].append(y_buf[k][-1] if len(y_buf[k]) else 0)

        if len(t_buf) < 2:
            return tuple(lines.values())

        for k in CHANNELS:
            lines[k].set_data(t_buf, y_buf[k])

        ax.set_xlim(t_buf[0], t_buf[-1])
        ymin = min(min(y_buf[k]) for k in CHANNELS)
        ymax = max(max(y_buf[k]) for k in CHANNELS)
        pad = (ymax - ymin) * 0.1 if ymax > ymin else 10
        ax.set_ylim(ymin - pad, ymax + pad)

        return tuple(lines.values())

    anim = animation.FuncAnimation(fig, update, interval=UPDATE_MS, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()

