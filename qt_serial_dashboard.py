import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import csv
import math

import serial
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg


# Serial settings (sketch_apr27b.ino)
PORT = "COM7"
BAUD = 9600

# Plot settings
MAX_POINTS = 500
STALE_AFTER_SECONDS = 3.0

DATA_DIR = Path("data_logs")

# Light classification from sketch logic (kept consistent with Arduino)
LIGHT_DARK_LT = 300
LIGHT_NORM_LT = 700

# Fire alarm: if F above threshold for N seconds => alarm
FIRE_THRESHOLD = 250
FIRE_HOLD_SECONDS = 2.0

# Tamper / fall detection for accelerometer angles (degrees)
TILT_ABS_WARN_DEG = 25
TILT_ABS_ALARM_DEG = 45

# Guard mode (accelerometer): react to changes vs baseline
GUARD_DY_WARN_DEG = 8
GUARD_DY_ALARM_DEG = 15

# Visual markers
MAX_EVENT_MARKERS_PER_PLOT = 60

RE_L = re.compile(r"\bL:(\d+)\b")
RE_F = re.compile(r"\bF:(\d+)\b")
RE_X = re.compile(r"\bX:(-?\d+)\b")
RE_Y = re.compile(r"\bY:(-?\d+)\b")
RE_Z = re.compile(r"\bZ:(-?\d+)\b")

@dataclass(frozen=True)
class Sample:
    t: float
    L: int
    F: int
    X: int
    Y: int
    Z: int


def classify_light(L: int) -> str:
    if L < LIGHT_DARK_LT:
        return "DARK"
    if L < LIGHT_NORM_LT:
        return "NORM"
    return "BRIGHT"


def accel_status(_X: int, Y: int, Z: int) -> str:
    # X channel can be noisy on some setups; we intentionally ignore it.
    a = max(abs(Y), abs(Z))
    if a >= TILT_ABS_ALARM_DEG:
        return "ALARM"
    if a >= TILT_ABS_WARN_DEG:
        return "WARN"
    return "OK"


def css_pill(bg: str) -> str:
    return (
        "QLabel {"
        f"background: {bg};"
        "color: white;"
        "padding: 6px 10px;"
        "border-radius: 10px;"
        "font-weight: 600;"
        "}"
    )

def css_card() -> str:
    return (
        "QGroupBox {"
        "border: 1px solid #dee2e6;"
        "border-radius: 10px;"
        "margin-top: 8px;"
        "padding: 10px;"
        "}"
        "QGroupBox::title {"
        "subcontrol-origin: margin;"
        "left: 12px;"
        "padding: 0 6px;"
        "font-weight: 900;"
        "font-size: 22px;"
        "}"
    )


def css_big_value() -> str:
    # Numbers should be readable, but not dominate the screen.
    return "QLabel { font-size: 34px; font-weight: 900; }"


def css_big_sub() -> str:
    # Status text (Норма/Внимание/Тревога) is the main information.
    return "QLabel { font-size: 34px; font-weight: 900; }"


def css_card_title() -> str:
    return "QLabel { font-size: 22px; font-weight: 900; color: #212529; }"


def css_fire_alert() -> str:
    return (
        "QLabel {"
        "background: #dc3545;"
        "color: white;"
        "padding: 10px 12px;"
        "border-radius: 10px;"
        "font-size: 22px;"
        "font-weight: 900;"
        "}"
    )

def css_toggle() -> str:
    # A simple "switch" style for QCheckBox
    return """
QCheckBox {
  font-size: 18px;
  font-weight: 900;
}
QCheckBox::indicator {
  width: 56px;
  height: 28px;
  border-radius: 14px;
  background: #adb5bd;
}
QCheckBox::indicator:checked {
  background: #198754;
}
"""

def css_banner(bg: str) -> str:
    return (
        "QLabel {"
        f"background: {bg};"
        "color: white;"
        "padding: 10px 12px;"
        "border-radius: 10px;"
        "font-size: 22px;"
        "font-weight: 800;"
        "}"
    )


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def daily_log_path() -> Path:
    ensure_data_dir()
    day = time.strftime("%Y-%m-%d")
    return DATA_DIR / f"sensors_{day}.csv"


def append_csv_row(s: Sample) -> None:
    p = daily_log_path()
    new_file = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["unix_ts", "t_s", "L", "F", "X", "Y", "Z"])
        unix_ts = time.time()
        w.writerow([f"{unix_ts:.3f}", f"{s.t:.3f}", s.L, s.F, s.X, s.Y, s.Z])


def iter_recent_rows(hours: int) -> list[Sample]:
    """
    Load recent samples from daily CSV logs.
    This is intentionally simple and robust for barn deployment.
    """
    cutoff = time.time() - hours * 3600
    samples: list[Sample] = []
    ensure_data_dir()
    # Consider today + yesterday (covers 24h, mostly also fine for 7d via tab below)
    days = [0, 1, 2, 3, 4, 5, 6]
    for d in days:
        day = time.strftime("%Y-%m-%d", time.localtime(time.time() - d * 86400))
        p = DATA_DIR / f"sensors_{day}.csv"
        if not p.exists():
            continue
        try:
            with p.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        unix_ts = float(row["unix_ts"])
                        if unix_ts < cutoff:
                            continue
                        samples.append(
                            Sample(
                                t=float(row["t_s"]),
                                L=int(row["L"]),
                                F=int(row["F"]),
                                X=int(row["X"]),
                                Y=int(row["Y"]),
                                Z=int(row["Z"]),
                            )
                        )
                    except Exception:
                        continue
        except Exception:
            continue
    samples.sort(key=lambda s: s.t)
    return samples


class SerialReader(QtCore.QThread):
    sample = QtCore.pyqtSignal(float, int, int, int, int, int)  # t, L, F, X, Y, Z
    status = QtCore.pyqtSignal(str)

    def __init__(self, port: str, baud: int, parent=None):
        super().__init__(parent)
        self._port = port
        self._baud = baud
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        t0 = time.time()
        ser = None
        try:
            ser = serial.Serial(self._port, self._baud, timeout=1)
            time.sleep(2)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            self.status.emit(f"Connected: {self._port} @ {self._baud}")

            last = {"L": 0, "F": 0, "X": 0, "Y": 0, "Z": 0}

            while not self._stop:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                m = RE_L.search(raw)
                if m:
                    last["L"] = int(m.group(1))
                m = RE_F.search(raw)
                if m:
                    last["F"] = int(m.group(1))
                m = RE_X.search(raw)
                if m:
                    last["X"] = int(m.group(1))
                m = RE_Y.search(raw)
                if m:
                    last["Y"] = int(m.group(1))
                m = RE_Z.search(raw)
                if m:
                    last["Z"] = int(m.group(1))

                t = time.time() - t0
                self.sample.emit(t, last["L"], last["F"], last["X"], last["Y"], last["Z"])
        except Exception as e:
            self.status.emit(f"Serial error: {e}")
        finally:
            try:
                if ser is not None:
                    ser.close()
            except Exception:
                pass


class App(QtWidgets.QMainWindow):
    def __init__(self, port: str, baud: int):
        super().__init__()
        self.setWindowTitle("Робот в хлеву — панель мониторинга")

        self._t = deque(maxlen=MAX_POINTS)
        self._L = deque(maxlen=MAX_POINTS)
        self._F = deque(maxlen=MAX_POINTS)
        self._X = deque(maxlen=MAX_POINTS)
        self._Y = deque(maxlen=MAX_POINTS)
        self._Z = deque(maxlen=MAX_POINTS)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        self.status_label = QtWidgets.QLabel("Starting…")
        top.addWidget(self.status_label, 1)

        self.pill_light = QtWidgets.QLabel("LIGHT: ?")
        self.pill_light.setStyleSheet(css_pill("#6c757d"))
        top.addWidget(self.pill_light)

        self.pill_fire = QtWidgets.QLabel("FIRE: ?")
        self.pill_fire.setStyleSheet(css_pill("#6c757d"))
        top.addWidget(self.pill_fire)

        self.pill_accel = QtWidgets.QLabel("ACCEL: ?")
        self.pill_accel.setStyleSheet(css_pill("#6c757d"))
        top.addWidget(self.pill_accel)

        self.btn_ack = QtWidgets.QPushButton("Acknowledge alarm")
        self.btn_ack.clicked.connect(self.ack_alarm)
        top.addWidget(self.btn_ack)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        dashboard = QtWidgets.QWidget()
        dash_layout = QtWidgets.QVBoxLayout(dashboard)
        self.tabs.addTab(dashboard, "Сводка")

        self.banner = QtWidgets.QLabel("СТАТУС: НОРМА")
        self.banner.setStyleSheet(css_banner("#198754"))
        dash_layout.addWidget(self.banner)

        dash_layout.addStretch(1)

        cards_wrap = QtWidgets.QHBoxLayout()
        cards_wrap.addStretch(1)
        cards = QtWidgets.QHBoxLayout()
        cards_wrap.addLayout(cards)
        cards_wrap.addStretch(1)
        dash_layout.addLayout(cards_wrap)

        self.card_light = QtWidgets.QGroupBox("Освещённость (L)")
        self.card_light.setStyleSheet(css_card())
        self.card_light.setMinimumWidth(320)
        self.card_light.setMinimumHeight(230)
        cl = QtWidgets.QVBoxLayout(self.card_light)
        self.sub_L = QtWidgets.QLabel("—")
        self.sub_L.setStyleSheet(css_big_sub())
        self.sub_L.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self.sub_L)
        cl.addStretch(1)
        self.val_L = QtWidgets.QLabel("—")
        self.val_L.setStyleSheet(css_big_value())
        self.val_L.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self.val_L)
        cards.addWidget(self.card_light, 1)

        self.card_fire = QtWidgets.QGroupBox("Огонь/дым (F)")
        self.card_fire.setStyleSheet(css_card())
        self.card_fire.setMinimumWidth(320)
        self.card_fire.setMinimumHeight(230)
        cf = QtWidgets.QVBoxLayout(self.card_fire)
        self.fire_alert = QtWidgets.QLabel("ВНИМАНИЕ: ВОЗМОЖЕН ПОЖАР")
        self.fire_alert.setStyleSheet(css_fire_alert())
        self.fire_alert.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.fire_alert.setVisible(False)
        cf.addWidget(self.fire_alert)
        self.sub_F = QtWidgets.QLabel("—")
        self.sub_F.setStyleSheet(css_big_sub())
        self.sub_F.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cf.addWidget(self.sub_F)
        cf.addStretch(1)
        self.val_F = QtWidgets.QLabel("—")
        self.val_F.setStyleSheet(css_big_value())
        self.val_F.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cf.addWidget(self.val_F)
        cards.addWidget(self.card_fire, 1)

        self.card_accel = QtWidgets.QGroupBox("Наклон/встряска (Y/Z)")
        self.card_accel.setStyleSheet(css_card())
        self.card_accel.setMinimumWidth(320)
        self.card_accel.setMinimumHeight(230)
        ca = QtWidgets.QVBoxLayout(self.card_accel)
        self.sub_A = QtWidgets.QLabel("—")
        self.sub_A.setStyleSheet(css_big_sub())
        self.sub_A.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ca.addWidget(self.sub_A)
        ca.addStretch(1)
        self.val_A = QtWidgets.QLabel("—")
        self.val_A.setStyleSheet(css_big_value())
        self.val_A.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ca.addWidget(self.val_A)
        ca.addStretch(1)
        self.guard_toggle = QtWidgets.QCheckBox("ОХРАНА")
        self.guard_toggle.setStyleSheet(css_toggle())
        self.guard_toggle.setChecked(False)
        self.guard_toggle.stateChanged.connect(lambda s: self.set_guard(s == 2))
        self.guard_toggle.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.guard_toggle.setTristate(False)
        self.guard_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        ca.addWidget(self.guard_toggle, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        cards.addWidget(self.card_accel, 1)

        dash_layout.addStretch(1)

        info = QtWidgets.QHBoxLayout()
        dash_layout.addLayout(info)
        self.lbl_last_update = QtWidgets.QLabel("Последнее обновление: —")
        info.addWidget(self.lbl_last_update, 1)
        self.lbl_port = QtWidgets.QLabel(f"Связь: {port} @ {baud}")
        info.addWidget(self.lbl_port)

        self.card_advice = QtWidgets.QGroupBox("Что делать сейчас")
        self.card_advice.setStyleSheet(css_card())
        adv = QtWidgets.QVBoxLayout(self.card_advice)
        self.advice = QtWidgets.QLabel("—")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet("QLabel { font-size: 18px; font-weight: 800; }")
        adv.addWidget(self.advice)
        dash_layout.addWidget(self.card_advice, 0)

        self.event_log = QtWidgets.QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(400)
        dash_layout.addWidget(self.event_log, 1)

        graphs = QtWidgets.QWidget()
        graphs_layout = QtWidgets.QVBoxLayout(graphs)
        self.tabs.addTab(graphs, "Графики")

        history = QtWidgets.QWidget()
        hist_layout = QtWidgets.QVBoxLayout(history)
        self.tabs.addTab(history, "История")

        pg.setConfigOptions(antialias=True)

        self.plot_light = pg.PlotWidget(title="Освещённость (L)")
        self.plot_light.showGrid(x=True, y=True, alpha=0.3)
        self.curve_L = self.plot_light.plot(pen=pg.mkPen(width=2))
        graphs_layout.addWidget(self.plot_light, 1)

        self.plot_flame = pg.PlotWidget(title="Огонь/дым (F)")
        self.plot_flame.showGrid(x=True, y=True, alpha=0.3)
        self.curve_F = self.plot_flame.plot(pen=pg.mkPen("y", width=2))
        graphs_layout.addWidget(self.plot_flame, 1)

        self.plot_accel = pg.PlotWidget(title="Наклон (Y/Z)")
        self.plot_accel.showGrid(x=True, y=True, alpha=0.3)
        self.curve_Y = self.plot_accel.plot(pen=pg.mkPen("g", width=2), name="Y")
        self.curve_Z = self.plot_accel.plot(pen=pg.mkPen("c", width=2), name="Z")
        self.legend = self.plot_accel.addLegend()
        self.legend.addItem(self.curve_Y, "Y")
        self.legend.addItem(self.curve_Z, "Z")
        graphs_layout.addWidget(self.plot_accel, 1)

        # History UI
        hist_top = QtWidgets.QHBoxLayout()
        hist_layout.addLayout(hist_top)
        self.hist_range = QtWidgets.QComboBox()
        self.hist_range.addItems(["Last 24 hours", "Last 7 days"])
        hist_top.addWidget(self.hist_range)
        self.btn_load_hist = QtWidgets.QPushButton("Load")
        self.btn_load_hist.clicked.connect(self.load_history)
        hist_top.addWidget(self.btn_load_hist)
        hist_top.addStretch(1)

        self.hist_plot = pg.PlotWidget(title="История (L/F/Y/Z)")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hist_plot.addLegend()
        self.hist_L = self.hist_plot.plot(pen=pg.mkPen(width=2), name="L")
        self.hist_F = self.hist_plot.plot(pen=pg.mkPen("y", width=2), name="F")
        self.hist_Y = self.hist_plot.plot(pen=pg.mkPen("g", width=2), name="Y")
        self.hist_Z = self.hist_plot.plot(pen=pg.mkPen("c", width=2), name="Z")
        hist_layout.addWidget(self.hist_plot, 1)

        # Event markers (vertical lines)
        self._markers_light: list[pg.InfiniteLine] = []
        self._markers_flame: list[pg.InfiniteLine] = []
        self._markers_accel: list[pg.InfiniteLine] = []

        # Alarm state
        self._fire_above_since: float | None = None
        self._alarm_active = False
        self._alarm_ack = False
        self._last_beep = 0.0
        self._latest: Sample | None = None
        self._last_rx_wall = 0.0
        self._last_fire_popup = 0.0
        self._last_accel_popup = 0.0
        self._guard_enabled = False
        self._guard_baseline: tuple[int, int] | None = None  # (Y, Z)
        self._guard_last_event = 0.0

        self.reader = SerialReader(port, baud, self)
        self.reader.sample.connect(self.on_sample)
        self.reader.status.connect(self.status_label.setText)
        self.reader.start()

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self.tick_ui)
        self.ui_timer.start()

    @QtCore.pyqtSlot(float, int, int, int, int, int)
    def on_sample(self, t, L, F, X, Y, Z):
        self._t.append(t)
        self._L.append(L)
        self._F.append(F)
        self._X.append(X)
        self._Y.append(Y)
        self._Z.append(Z)

        s = Sample(t=t, L=L, F=F, X=X, Y=Y, Z=Z)
        self._latest = s
        self._last_rx_wall = time.time()
        try:
            append_csv_row(s)
        except Exception:
            pass

        self.update_status_and_events(s)

    def add_event_marker(self, plot: pg.PlotWidget, markers: list[pg.InfiniteLine], t: float, color: str):
        line = pg.InfiniteLine(pos=t, angle=90, movable=False, pen=pg.mkPen(color, width=2))
        plot.addItem(line)
        markers.append(line)
        while len(markers) > MAX_EVENT_MARKERS_PER_PLOT:
            old = markers.pop(0)
            try:
                plot.removeItem(old)
            except Exception:
                pass

    def log_event(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.event_log.appendPlainText(f"[{ts}] {msg}")

    def ack_alarm(self):
        self._alarm_ack = True
        if self._alarm_active:
            self.log_event("Тревога подтверждена оператором.")

    def set_guard(self, enabled: bool):
        self._guard_enabled = enabled
        if enabled:
            # set baseline immediately from latest sample (if available)
            if self._latest is not None:
                self._guard_baseline = (self._latest.Y, self._latest.Z)
            else:
                self._guard_baseline = (0, 0)
            self.log_event("Охрана по наклону: ВКЛ (зафиксирована норма).")
        else:
            self._guard_baseline = None
            self.log_event("Охрана по наклону: ВЫКЛ.")

    def check_guard(self, s: Sample):
        if not self._guard_enabled or self._guard_baseline is None:
            return
        by, bz = self._guard_baseline
        dy = abs(s.Y - by)
        dz = abs(s.Z - bz)
        d = max(dy, dz)
        now = time.time()

        if d >= GUARD_DY_ALARM_DEG:
            # rate-limit events
            if (now - self._guard_last_event) > 5.0:
                self._guard_last_event = now
                self.log_event(f"ОХРАНА: движение/наклон! (Δ={d}°, Y={s.Y}, Z={s.Z})")
                self.add_event_marker(self.plot_accel, self._markers_accel, s.t, "#dc3545")
                QtWidgets.QApplication.beep()
                if (now - self._last_accel_popup) > 20:
                    self._last_accel_popup = now
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Охрана: тревога",
                        "Обнаружено изменение наклона/движение.\n"
                        "Возможное перемещение робота или вмешательство.",
                    )
        elif d >= GUARD_DY_WARN_DEG:
            if (now - self._guard_last_event) > 10.0:
                self._guard_last_event = now
                self.log_event(f"ОХРАНА: изменение наклона (Δ={d}°).")

    def update_status_and_events(self, s: Sample):
        # Light pill (matches Arduino categories)
        light = classify_light(s.L)
        if light == "DARK":
            self.pill_light.setText("СВЕТ: ТЕМНО")
            self.pill_light.setStyleSheet(css_pill("#343a40"))
        elif light == "NORM":
            self.pill_light.setText("СВЕТ: НОРМА")
            self.pill_light.setStyleSheet(css_pill("#198754"))
        else:
            self.pill_light.setText("СВЕТ: ЯРКО")
            self.pill_light.setStyleSheet(css_pill("#0d6efd"))

        # Fire pill + alarm logic
        now = time.time()
        if s.F > FIRE_THRESHOLD:
            if self._fire_above_since is None:
                self._fire_above_since = now
                self.log_event(f"Риск пожара растёт (F={s.F}).")
                self.add_event_marker(self.plot_flame, self._markers_flame, s.t, "#ff7b00")
            if (now - self._fire_above_since) >= FIRE_HOLD_SECONDS:
                if not self._alarm_active:
                    self._alarm_active = True
                    self._alarm_ack = False
                    self.log_event(f"ПОЖАРНАЯ ТРЕВОГА (F={s.F})!")
                    self.add_event_marker(self.plot_flame, self._markers_flame, s.t, "#dc3545")
        else:
            if self._fire_above_since is not None:
                self.log_event(f"Риск пожара снизился (F={s.F}).")
                self.add_event_marker(self.plot_flame, self._markers_flame, s.t, "#198754")
            self._fire_above_since = None
            # auto-clear alarm when safe again (but keep acknowledged state)
            self._alarm_active = False

        if self._alarm_active:
            self.pill_fire.setText("ОГОНЬ: ТРЕВОГА")
            self.pill_fire.setStyleSheet(css_pill("#dc3545"))
            # periodic beep until acknowledged
            if not self._alarm_ack and (now - self._last_beep) > 1.0:
                QtWidgets.QApplication.beep()
                self._last_beep = now
        else:
            self.pill_fire.setText("ОГОНЬ: НОРМА" if s.F <= FIRE_THRESHOLD else "ОГОНЬ: ВНИМАНИЕ")
            self.pill_fire.setStyleSheet(css_pill("#198754" if s.F <= FIRE_THRESHOLD else "#ff7b00"))

        # Accel pill
        a_status = accel_status(s.X, s.Y, s.Z)
        if a_status == "OK":
            self.pill_accel.setText("НАКЛОН: НОРМА")
            self.pill_accel.setStyleSheet(css_pill("#198754"))
        elif a_status == "WARN":
            self.pill_accel.setText("НАКЛОН: ВНИМАНИЕ")
            self.pill_accel.setStyleSheet(css_pill("#ff7b00"))
        else:
            self.pill_accel.setText("НАКЛОН: ТРЕВОГА")
            self.pill_accel.setStyleSheet(css_pill("#dc3545"))
            self.log_event(f"Подозрение на падение/вскрытие (X={s.X}, Y={s.Y}, Z={s.Z}).")
            self.add_event_marker(self.plot_accel, self._markers_accel, s.t, "#dc3545")

        # Guard mode reacts to changes vs baseline
        self.check_guard(s)

    def tick_ui(self):
        # Connection / stale data indicator
        now = time.time()
        if self._last_rx_wall and (now - self._last_rx_wall) > STALE_AFTER_SECONDS:
            self.banner.setText("НЕТ ДАННЫХ — проверь питание/кабель/COM-порт")
            self.banner.setStyleSheet(css_banner("#6c757d"))
            self.advice.setText(
                "1) Проверь USB-кабель и питание платы.\n"
                "2) Убедись, что COM-порт не занят (Serial Monitor должен быть закрыт).\n"
                "3) Если не помогло — переподключи кабель."
            )
        elif self._alarm_active and not self._alarm_ack:
            self.banner.setText("ПОЖАРНАЯ ТРЕВОГА — действуй немедленно!")
            self.banner.setStyleSheet(css_banner("#dc3545"))
            self.advice.setText(
                "1) Проверь очаг (сено/проводка/обогреватель).\n"
                "2) Обеспечь вентиляцию/отключи питание опасных линий.\n"
                "3) Если есть дым/огонь — тушение/112."
            )
        else:
            self.banner.setText("СТАТУС: НОРМА")
            self.banner.setStyleSheet(css_banner("#198754"))
            self.advice.setText(
                "Система работает. Следи за «Огонь/дым» и «Наклон» — это самые критичные события."
            )

        # Update main info values
        s = self._latest
        if s is not None:
            self.val_L.setText(str(s.L))
            self.sub_L.setText(f"{'ТЕМНО' if classify_light(s.L)=='DARK' else 'НОРМА' if classify_light(s.L)=='NORM' else 'ЯРКО'}")

            self.val_F.setText(str(s.F))
            fire_text = "OK" if s.F <= FIRE_THRESHOLD else "WARN"
            if self._alarm_active:
                fire_text = "ALARM"
            fire_ru = {"OK": "НОРМА", "WARN": "ВНИМАНИЕ", "ALARM": "ТРЕВОГА"}[fire_text]
            self.sub_F.setText(fire_ru)
            self.fire_alert.setVisible(self._alarm_active and not self._alarm_ack)

            a = max(abs(s.Y), abs(s.Z))
            self.val_A.setText(f"{a}°")
            self.sub_A.setText(f"Y={s.Y}  Z={s.Z}")

            self.lbl_last_update.setText(f"Последнее обновление: {time.strftime('%H:%M:%S')}")

            # Popups for critical events (rate-limited)
            if self._alarm_active and not self._alarm_ack and (now - self._last_fire_popup) > 20:
                self._last_fire_popup = now
                QtWidgets.QMessageBox.critical(
                    self,
                    "Пожарная тревога",
                    "Датчик огня/дыма показывает опасное значение.\n"
                    "Проверь хлев и источник огня/дыма.",
                )

            if accel_status(s.X, s.Y, s.Z) == "ALARM" and (now - self._last_accel_popup) > 20:
                self._last_accel_popup = now
                QtWidgets.QMessageBox.warning(
                    self,
                    "Тревога по наклону",
                    "Резкий наклон/встряска.\n"
                    "Возможное падение робота или вскрытие корпуса.",
                )

        # Redraw graphs (in separate tab now)
        if len(self._t) >= 2:
            x = list(self._t)
            self.curve_L.setData(x, list(self._L))
            self.curve_F.setData(x, list(self._F))
            self.curve_Y.setData(x, list(self._Y))
            self.curve_Z.setData(x, list(self._Z))

    def load_history(self):
        text = self.hist_range.currentText()
        hours = 24 if "24" in text else 24 * 7
        rows = iter_recent_rows(hours=hours)
        if len(rows) < 2:
            self.status_label.setText("История: данных пока мало (подожди немного).")
            return

        # Build a monotonic time axis for history: use real time (unix) would require storing it,
        # but for practical farmer view, relative axis is sufficient.
        # We rebase to 0.
        t0 = rows[0].t
        xs = [r.t - t0 for r in rows]
        Ls = [r.L for r in rows]
        Fs = [r.F for r in rows]
        Xs = [r.X for r in rows]
        Ys = [r.Y for r in rows]
        Zs = [r.Z for r in rows]

        # Decimate for speed (aim ~5k points max)
        target = 5000
        step = max(1, len(xs) // target)
        if step > 1:
            xs = xs[::step]
            Ls = Ls[::step]
            Fs = Fs[::step]
            Ys = Ys[::step]
            Zs = Zs[::step]

        self.hist_L.setData(xs, Ls)
        self.hist_F.setData(xs, Fs)
        self.hist_Y.setData(xs, Ys)
        self.hist_Z.setData(xs, Zs)
        self.status_label.setText(f"История загружена: {len(xs)} точек ({text}).")

    def closeEvent(self, event):
        try:
            self.reader.stop()
            self.reader.wait(1500)
        finally:
            super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName("Робот в хлеву")
    win = App(PORT, BAUD)
    win.resize(1100, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

