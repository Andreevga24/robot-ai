import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import csv
import math

try:
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore
    _VISION_IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover
    cv2 = None
    YOLO = None
    _VISION_IMPORT_ERROR = repr(e)

import serial
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
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

# Vision (YOLO) settings
VISION_SOURCE = "0"  # "0" for default camera
VISION_IMGSZ = 640
VISION_DEVICE = "cpu"
VISION_MODEL_PATH = (Path(__file__).resolve().parent / "runs/detect/train_updated_115/weights/best.pt")
VISION_CONF_ALERT = 0.40
VISION_CONF_CHICKEN = 0.30
VISION_ALARM_HOLD_SECONDS = 0.40  # require persistence to reduce false positives

# Vision anti-false-positive filters
# - Class-specific confidence thresholds (override VISION_CONF_ALERT for counting/messages)
VISION_CONF_BY_CLASS: dict[str, float] = {
    "cow_lumpy": 0.65,
    "sheep_wool": 0.55,
    "corn": 0.55,
    "round hay": 0.55,
    "square hay": 0.55,
}
# - Minimum bbox area ratio (bbox_area / frame_area) per class for counting/messages
VISION_MIN_AREA_RATIO_BY_CLASS: dict[str, float] = {
    "cow_lumpy": 0.010,   # 1.0% of frame
    "sheep_wool": 0.008,  # 0.8%
    "corn": 0.004,        # 0.4%
    "round hay": 0.010,   # 1.0%
    "square hay": 0.010,  # 1.0%
}
# - Ignore detections too close to frame borders (often reflections / clutter)
VISION_ROI_BORDER_FRAC_X = 0.02  # ignore 2% left/right
VISION_ROI_BORDER_FRAC_Y = 0.02  # ignore 2% top/bottom

# Stable chicken counting (simple tracker)
VISION_CHICKEN_TRACK_MAX_DIST_PX = 80  # max centroid distance to match detections -> tracks
VISION_CHICKEN_TRACK_MAX_MISSES = 8  # frames to keep track without detection
VISION_CHICKEN_SMOOTH_FRAMES = 7  # extra smoothing of the final count (median)

ALERTS_DIR = Path("runs/alerts")

PREDATOR_CLASSES = {"fox", "marten", "wolf", "volf"}
FIRE_CLASS = "fire"
CHICKEN_CLASS = "chicken"
COW_LUMPY_CLASS = "cow_lumpy"
SHEEP_WOOL_CLASS = "sheep_wool"
CORN_CLASS = "corn"
HAY_CLASSES = {"round hay", "square hay"}
HAY_MIN_COUNT_PER_IMAGE = 3
CORN_MIN_COUNT_PER_IMAGE = 3

# Temporal stability for non-critical messages (anti-flicker)
VISION_STAB_WINDOW = 10  # frames
VISION_STAB_HAY_LOW_MIN_TRUE = 7
VISION_STAB_CORN_LOW_MIN_TRUE = 7
VISION_STAB_COW_LUMPY_WINDOW = 5
VISION_STAB_COW_LUMPY_MIN_TRUE = 3
VISION_STAB_SHEEP_WOOL_WINDOW = 4
VISION_STAB_SHEEP_WOOL_MIN_TRUE = 2

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

def css_vision_alarm(bg: str) -> str:
    return (
        "QLabel {"
        f"background: {bg};"
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


class VideoInferenceThread(QtCore.QThread):
    frame = QtCore.pyqtSignal(object)  # QImage
    frame_bgr = QtCore.pyqtSignal(object)  # dict: {ts: float, bgr: np.ndarray}
    summary = QtCore.pyqtSignal(object)  # dict with counts/alarms
    status = QtCore.pyqtSignal(str)

    def __init__(
        self,
        model_path: Path,
        source: str,
        conf_alert: float,
        conf_chicken: float,
        imgsz: int,
        device: str,
        parent=None,
    ):
        super().__init__(parent)
        self._model_path = model_path
        self._source = source
        self._conf_alert = conf_alert
        self._conf_chicken = conf_chicken
        self._imgsz = imgsz
        self._device = device
        self._stop = False
        self._trk_next_id = 1
        self._trk: dict[int, tuple[float, float, int]] = {}  # id -> (cx, cy, misses)

    def stop(self):
        self._stop = True

    @staticmethod
    def _parse_source(raw: str) -> int | str:
        try:
            return int(raw)
        except ValueError:
            return raw

    @staticmethod
    def _bgr_to_qimage(bgr) -> QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        # copy() to detach from numpy buffer lifecycle
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    def run(self):
        if cv2 is None or YOLO is None:
            self.status.emit("YOLO: dependencies missing (opencv-python / ultralytics).")
            return
        model_path = self._model_path.resolve()
        if not model_path.is_file():
            self.status.emit(f"YOLO: model not found: {model_path}")
            return

        try:
            model = YOLO(str(model_path))
        except Exception as e:
            self.status.emit(f"YOLO: failed to load model: {e}")
            return

        src = self._parse_source(self._source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status.emit(f"Camera: cannot open source: {self._source}")
            return

        self.status.emit(f"Camera: started ({self._source}), model: {model_path.name}")

        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                try:
                    r = model.predict(
                        source=frame,
                        conf=min(self._conf_alert, self._conf_chicken),
                        imgsz=self._imgsz,
                        device=self._device,
                        verbose=False,
                    )[0]
                except Exception as e:
                    self.status.emit(f"YOLO: inference error: {e}")
                    time.sleep(0.25)
                    continue

                names = getattr(model, "names", {}) or {}

                chicken_dets: list[tuple[float, float]] = []
                fire = False
                predator = False
                counts: dict[str, int] = {}
                frame_h, frame_w = frame.shape[:2]
                frame_area = float(frame_w * frame_h) if frame_w and frame_h else 1.0

                if r.boxes is not None and len(r.boxes) > 0:
                    try:
                        cls_ids = r.boxes.cls.tolist()
                        confs = r.boxes.conf.tolist()
                        xys = r.boxes.xyxy.tolist()
                    except Exception:
                        cls_ids = []
                        confs = []
                        xys = []

                    for (cid, c, xyxy) in zip(cls_ids, confs, xys):
                        label = names.get(int(cid), str(int(cid)))
                        if label == CHICKEN_CLASS and c >= self._conf_chicken:
                            x1, y1, x2, y2 = xyxy
                            cx = (float(x1) + float(x2)) / 2.0
                            cy = (float(y1) + float(y2)) / 2.0
                            chicken_dets.append((cx, cy))
                        if c >= self._conf_alert:
                            if label == FIRE_CLASS:
                                fire = True
                            if label in PREDATOR_CLASSES:
                                predator = True

                        # Counting/message stream uses stricter filters (per-class conf, bbox size, ROI)
                        need_conf = float(VISION_CONF_BY_CLASS.get(label, self._conf_alert))
                        if c < need_conf:
                            continue
                        try:
                            x1, y1, x2, y2 = xyxy
                            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
                        except Exception:
                            continue
                        # Clamp
                        x1f = max(0.0, min(x1f, float(frame_w)))
                        x2f = max(0.0, min(x2f, float(frame_w)))
                        y1f = max(0.0, min(y1f, float(frame_h)))
                        y2f = max(0.0, min(y2f, float(frame_h)))
                        bw = max(0.0, x2f - x1f)
                        bh = max(0.0, y2f - y1f)
                        if bw <= 1.0 or bh <= 1.0:
                            continue
                        # ROI border ignore
                        cx = (x1f + x2f) / 2.0
                        cy = (y1f + y2f) / 2.0
                        if (
                            cx < (VISION_ROI_BORDER_FRAC_X * frame_w)
                            or cx > ((1.0 - VISION_ROI_BORDER_FRAC_X) * frame_w)
                            or cy < (VISION_ROI_BORDER_FRAC_Y * frame_h)
                            or cy > ((1.0 - VISION_ROI_BORDER_FRAC_Y) * frame_h)
                        ):
                            continue
                        # Min area ratio per class
                        min_ratio = float(VISION_MIN_AREA_RATIO_BY_CLASS.get(label, 0.0))
                        if ((bw * bh) / frame_area) < min_ratio:
                            continue

                        counts[label] = counts.get(label, 0) + 1

                # Update chicken tracker to stabilize counting
                chicken = self._update_chicken_tracks(chicken_dets)

                annotated = r.plot()
                self.frame_bgr.emit({"ts": time.time(), "bgr": annotated})
                qimg = self._bgr_to_qimage(annotated)
                self.frame.emit(qimg)
                self.summary.emit(
                    {
                        "ts": time.time(),
                        "chicken": chicken,
                        "fire": fire,
                        "predator": predator,
                        "counts": counts,
                    }
                )
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self.status.emit("Camera: stopped.")

    def _update_chicken_tracks(self, dets: list[tuple[float, float]]) -> int:
        # Greedy nearest-neighbor assignment (fast, good enough for single-camera barn sweep)
        # Track state: (cx, cy, misses)
        if not self._trk:
            for (cx, cy) in dets:
                tid = self._trk_next_id
                self._trk_next_id += 1
                self._trk[tid] = (cx, cy, 0)
            return len(self._trk)

        # Prepare matching
        unmatched_tracks = set(self._trk.keys())
        unmatched_dets = set(range(len(dets)))
        matches: list[tuple[int, int]] = []  # (track_id, det_idx)

        # Build all distances
        dist_pairs: list[tuple[float, int, int]] = []
        for tid, (tx, ty, _miss) in self._trk.items():
            for di, (cx, cy) in enumerate(dets):
                dx = tx - cx
                dy = ty - cy
                dist = math.hypot(dx, dy)
                if dist <= VISION_CHICKEN_TRACK_MAX_DIST_PX:
                    dist_pairs.append((dist, tid, di))

        dist_pairs.sort(key=lambda t: t[0])
        for _dist, tid, di in dist_pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)
            matches.append((tid, di))

        # Update matched tracks
        for tid, di in matches:
            cx, cy = dets[di]
            self._trk[tid] = (cx, cy, 0)

        # Missed tracks
        to_delete: list[int] = []
        for tid in list(unmatched_tracks):
            tx, ty, miss = self._trk[tid]
            miss += 1
            if miss > VISION_CHICKEN_TRACK_MAX_MISSES:
                to_delete.append(tid)
            else:
                self._trk[tid] = (tx, ty, miss)
        for tid in to_delete:
            self._trk.pop(tid, None)

        # New tracks for unmatched detections
        for di in list(unmatched_dets):
            cx, cy = dets[di]
            tid = self._trk_next_id
            self._trk_next_id += 1
            self._trk[tid] = (cx, cy, 0)

        return len(self._trk)


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

        # Dashboard tab with scroll (camera + cards may not fit on small screens)
        dashboard = QtWidgets.QWidget()
        dash_outer = QtWidgets.QVBoxLayout(dashboard)
        dash_outer.setContentsMargins(0, 0, 0, 0)
        dash_outer.setSpacing(0)

        dash_scroll = QtWidgets.QScrollArea()
        dash_scroll.setWidgetResizable(True)
        dash_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        dash_outer.addWidget(dash_scroll, 1)

        dash_content = QtWidgets.QWidget()
        dash_scroll.setWidget(dash_content)
        dash_layout = QtWidgets.QVBoxLayout(dash_content)
        self.tabs.addTab(dashboard, "Сводка")

        self.banner = QtWidgets.QLabel("СТАТУС: НОРМА")
        self.banner.setStyleSheet(css_banner("#198754"))
        dash_layout.addWidget(self.banner)

        # Camera / YOLO block (operator needs it on summary)
        self.card_camera = QtWidgets.QGroupBox("Камера (YOLO)")
        self.card_camera.setStyleSheet(css_card())
        cam_layout = QtWidgets.QVBoxLayout(self.card_camera)

        self.cam_status = QtWidgets.QLabel("Камера: —")
        self.cam_status.setStyleSheet("QLabel { font-size: 16px; font-weight: 700; color: #495057; }")
        cam_layout.addWidget(self.cam_status)

        self.cam_alarm = QtWidgets.QLabel("VISION: НОРМА")
        self.cam_alarm.setStyleSheet(css_vision_alarm("#198754"))
        cam_layout.addWidget(self.cam_alarm)

        stats = QtWidgets.QHBoxLayout()
        cam_layout.addLayout(stats)
        self.cam_chicken = QtWidgets.QLabel("CHICKEN: —")
        self.cam_chicken.setStyleSheet("QLabel { font-size: 28px; font-weight: 900; }")
        stats.addWidget(self.cam_chicken)
        stats.addStretch(1)

        self.btn_cam_toggle = QtWidgets.QPushButton("Старт камеры")
        self.btn_cam_toggle.clicked.connect(self.toggle_camera)
        stats.addWidget(self.btn_cam_toggle)

        cam_controls = QtWidgets.QHBoxLayout()
        cam_layout.addLayout(cam_controls)

        self.chk_cam_autostart = QtWidgets.QCheckBox("Автостарт")
        self.chk_cam_autostart.setChecked(True)
        cam_controls.addWidget(self.chk_cam_autostart)

        cam_controls.addSpacing(12)

        cam_controls.addWidget(QtWidgets.QLabel("conf тревога"))
        self.spin_conf_alert = QtWidgets.QDoubleSpinBox()
        self.spin_conf_alert.setRange(0.01, 0.99)
        self.spin_conf_alert.setSingleStep(0.05)
        self.spin_conf_alert.setDecimals(2)
        self.spin_conf_alert.setValue(VISION_CONF_ALERT)
        cam_controls.addWidget(self.spin_conf_alert)

        cam_controls.addWidget(QtWidgets.QLabel("conf chicken"))
        self.spin_conf_chicken = QtWidgets.QDoubleSpinBox()
        self.spin_conf_chicken.setRange(0.01, 0.99)
        self.spin_conf_chicken.setSingleStep(0.05)
        self.spin_conf_chicken.setDecimals(2)
        self.spin_conf_chicken.setValue(VISION_CONF_CHICKEN)
        cam_controls.addWidget(self.spin_conf_chicken)

        cam_controls.addWidget(QtWidgets.QLabel("hold, сек"))
        self.spin_hold = QtWidgets.QDoubleSpinBox()
        self.spin_hold.setRange(0.0, 3.0)
        self.spin_hold.setSingleStep(0.1)
        self.spin_hold.setDecimals(1)
        self.spin_hold.setValue(VISION_ALARM_HOLD_SECONDS)
        cam_controls.addWidget(self.spin_hold)

        self.btn_cam_apply = QtWidgets.QPushButton("Применить")
        self.btn_cam_apply.clicked.connect(self.apply_vision_settings)
        cam_controls.addWidget(self.btn_cam_apply)

        cam_controls.addStretch(1)

        self.btn_snapshot = QtWidgets.QPushButton("Скрин")
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        cam_controls.addWidget(self.btn_snapshot)

        self.btn_clip = QtWidgets.QPushButton("Клип 5 сек")
        self.btn_clip.clicked.connect(self.save_clip_5s)
        cam_controls.addWidget(self.btn_clip)

        self.video = QtWidgets.QLabel()
        self.video.setMinimumHeight(360)
        self.video.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet(
            "QLabel { background: #111; border-radius: 10px; padding: 6px; color: #adb5bd; }"
        )
        self.video.setText("Видео не запущено.")
        cam_layout.addWidget(self.video, 1)

        dash_layout.addWidget(self.card_camera, 1)

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

        # Vision state
        self._vision_fire = False
        self._vision_predator = False
        self._vision_alarm_active = False
        self._vision_alarm_ack = False
        self._vision_above_since: float | None = None
        self._vision_last_beep = 0.0
        self._vision_last_popup = 0.0
        self._vision_chicken_recent = deque(maxlen=VISION_CHICKEN_SMOOTH_FRAMES)
        self._video_thread: VideoInferenceThread | None = None
        self._vision_frame_buf = deque(maxlen=400)  # (ts, bgr)
        self._last_qimg: QImage | None = None
        self._vision_counts: dict[str, int] = {}
        self._vision_last_msg: dict[str, float] = {}
        self._vision_reco_lines: list[str] = []
        self._vision_counts_hist = deque(maxlen=VISION_STAB_WINDOW)

        self.reader = SerialReader(port, baud, self)
        self.reader.sample.connect(self.on_sample)
        self.reader.status.connect(self.status_label.setText)
        self.reader.start()

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self.tick_ui)
        self.ui_timer.start()

        # Camera is started by the operator (button) to avoid "silent no-op" at boot.
        self.refresh_camera_status()
        if self.chk_cam_autostart.isChecked():
            QtCore.QTimer.singleShot(200, lambda: self.toggle_camera(force_on=True))

    def refresh_camera_status(self):
        if cv2 is None or YOLO is None:
            msg = "Камера: YOLO недоступен."
            if "_VISION_IMPORT_ERROR" in globals() and _VISION_IMPORT_ERROR:
                msg += f" Ошибка импорта: {_VISION_IMPORT_ERROR}"
            else:
                msg += " Нужны `opencv-python` и `ultralytics`."
            self.cam_status.setText(msg)
            return
        if not VISION_MODEL_PATH.is_file():
            self.cam_status.setText(f"YOLO: не найден файл модели: {VISION_MODEL_PATH}")
            return
        self.cam_status.setText(f"Камера готова. Модель: {VISION_MODEL_PATH.name}, source={VISION_SOURCE}")

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
        self._vision_alarm_ack = True
        if self._vision_alarm_active:
            self.log_event("VISION-тревога подтверждена оператором.")

    def toggle_camera(self, force_on: bool = False):
        want_on = force_on or self._video_thread is None
        if want_on:
            self.log_event("Камера: запуск…")
            if cv2 is None or YOLO is None:
                self.refresh_camera_status()
                extra = ""
                if "_VISION_IMPORT_ERROR" in globals() and _VISION_IMPORT_ERROR:
                    extra = f"\n\nТекст ошибки импорта:\n{_VISION_IMPORT_ERROR}"
                QtWidgets.QMessageBox.information(
                    self,
                    "Камера",
                    "Не установлены зависимости для YOLO.\n\n"
                    "Выполни:\n"
                    "py -m pip install -r requirements.txt"
                    f"{extra}",
                )
                return
            if not VISION_MODEL_PATH.is_file():
                self.cam_status.setText(f"YOLO: не найден файл модели: {VISION_MODEL_PATH}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "YOLO модель не найдена",
                    "Не найден файл весов модели.\n\n"
                    f"Проверь путь:\n{VISION_MODEL_PATH}",
                )
                return
            self.cam_status.setText("Камера: запускаю поток…")
            self._video_thread = VideoInferenceThread(
                model_path=VISION_MODEL_PATH,
                source=VISION_SOURCE,
                conf_alert=float(self.spin_conf_alert.value()),
                conf_chicken=float(self.spin_conf_chicken.value()),
                imgsz=VISION_IMGSZ,
                device=VISION_DEVICE,
                parent=self,
            )
            self._video_thread.frame.connect(self.on_video_frame)
            self._video_thread.frame_bgr.connect(self.on_video_frame_bgr)
            self._video_thread.summary.connect(self.on_vision_summary)
            self._video_thread.status.connect(self.cam_status.setText)
            self._video_thread.start()
            self.btn_cam_toggle.setText("Стоп камера")
            self.video.setText("Запуск…")
        else:
            try:
                if self._video_thread is not None:
                    self.log_event("Камера: остановка…")
                    self._video_thread.stop()
                    self._video_thread.wait(1500)
            finally:
                self._video_thread = None
                self.btn_cam_toggle.setText("Старт камеры")
                self.video.setText("Видео остановлено.")
                self.refresh_camera_status()

    @QtCore.pyqtSlot(object)
    def on_video_frame(self, qimg: QImage):
        if qimg is None:
            return
        self._last_qimg = qimg
        pix = QPixmap.fromImage(qimg)
        # Fit to label size while keeping aspect ratio
        scaled = pix.scaled(
            self.video.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.video.setPixmap(scaled)

    @QtCore.pyqtSlot(object)
    def on_video_frame_bgr(self, payload: dict):
        try:
            ts = float(payload.get("ts", time.time()))
            bgr = payload.get("bgr", None)
            if bgr is None:
                return
            self._vision_frame_buf.append((ts, bgr))
        except Exception:
            return

    @QtCore.pyqtSlot(object)
    def on_vision_summary(self, s: dict):
        self._vision_chicken_recent.append(int(s.get("chicken", 0)))
        self._vision_fire = bool(s.get("fire", False))
        self._vision_predator = bool(s.get("predator", False))
        self._vision_counts = dict(s.get("counts", {}) or {})
        self._vision_counts_hist.append(self._vision_counts)

        now = time.time()
        above = self._vision_fire or self._vision_predator
        if above:
            if self._vision_above_since is None:
                self._vision_above_since = now
            if (now - self._vision_above_since) >= float(self.spin_hold.value()):
                if not self._vision_alarm_active:
                    self._vision_alarm_active = True
                    self._vision_alarm_ack = False
                    if self._vision_fire:
                        self.log_event("VISION: ПОЖАР обнаружен моделью!")
                    elif self._vision_predator:
                        self.log_event("VISION: ХИЩНИК в хлеву (модель)!")
        else:
            self._vision_above_since = None
            self._vision_alarm_active = False

        # Non-critical vision messages (rate-limited)
        self._emit_vision_messages(now)
        self._vision_reco_lines = self._compute_vision_recommendations()

    def _emit_vision_messages(self, now: float) -> None:
        def cooldown_ok(key: str, seconds: float) -> bool:
            last = float(self._vision_last_msg.get(key, 0.0))
            return (now - last) >= seconds

        cow_lumpy_present = self._stable_presence(COW_LUMPY_CLASS, VISION_STAB_COW_LUMPY_WINDOW, VISION_STAB_COW_LUMPY_MIN_TRUE)
        if cow_lumpy_present and cooldown_ok("cow_lumpy", 20.0):
            self._vision_last_msg["cow_lumpy"] = now
            self.log_event('VISION: Возможно зарожение (cow_lumpy).')

        sheep_present = self._stable_presence(SHEEP_WOOL_CLASS, VISION_STAB_SHEEP_WOOL_WINDOW, VISION_STAB_SHEEP_WOOL_MIN_TRUE)
        if sheep_present and cooldown_ok("sheep_wool", 25.0):
            self._vision_last_msg["sheep_wool"] = now
            self.log_event('VISION: Время стрижки овец (sheep_wool).')

        hay_low = self._stable_hay_low()
        if hay_low and cooldown_ok("hay_low", 30.0):
            self._vision_last_msg["hay_low"] = now
            self.log_event("VISION: Пополните запасы сена.")

        corn_low = self._stable_corn_low()
        if corn_low and cooldown_ok("corn_low", 30.0):
            self._vision_last_msg["corn_low"] = now
            self.log_event("VISION: пополните запасы пшена.")

    def _stable_presence(self, cls: str, window: int, min_true: int) -> bool:
        if window <= 0 or min_true <= 0:
            return False
        hist = list(self._vision_counts_hist)[-window:]
        if len(hist) < window:
            return False
        trues = sum(1 for d in hist if int(d.get(cls, 0)) > 0)
        return trues >= min_true

    def _stable_hay_low(self) -> bool:
        hist = list(self._vision_counts_hist)
        if len(hist) < VISION_STAB_WINDOW:
            return False
        def is_low(d: dict[str, int]) -> bool:
            total = sum(int(d.get(c, 0)) for c in HAY_CLASSES)
            return 0 < total < HAY_MIN_COUNT_PER_IMAGE
        trues = sum(1 for d in hist if is_low(d))
        return trues >= VISION_STAB_HAY_LOW_MIN_TRUE

    def _stable_corn_low(self) -> bool:
        hist = list(self._vision_counts_hist)
        if len(hist) < VISION_STAB_WINDOW:
            return False
        def is_low(d: dict[str, int]) -> bool:
            c = int(d.get(CORN_CLASS, 0))
            return 0 < c < CORN_MIN_COUNT_PER_IMAGE
        trues = sum(1 for d in hist if is_low(d))
        return trues >= VISION_STAB_CORN_LOW_MIN_TRUE

    def _compute_vision_recommendations(self) -> list[str]:
        lines: list[str] = []
        if self._stable_presence(COW_LUMPY_CLASS, VISION_STAB_COW_LUMPY_WINDOW, VISION_STAB_COW_LUMPY_MIN_TRUE):
            lines.append('Возможно зарожение')

        if self._stable_presence(SHEEP_WOOL_CLASS, VISION_STAB_SHEEP_WOOL_WINDOW, VISION_STAB_SHEEP_WOOL_MIN_TRUE):
            lines.append('Время стрижки овец')

        if self._stable_hay_low():
            lines.append("Пополните запасы сена")

        if self._stable_corn_low():
            lines.append("пополните запасы пшена")

        return lines

    def apply_vision_settings(self):
        self.log_event(
            f"VISION: применены настройки: conf_alert={self.spin_conf_alert.value():.2f}, "
            f"conf_chicken={self.spin_conf_chicken.value():.2f}, hold={self.spin_hold.value():.1f}s"
        )
        # Settings are applied on next camera start. Restart if currently running.
        if self._video_thread is not None:
            self.toggle_camera(force_on=False)
            QtCore.QTimer.singleShot(250, lambda: self.toggle_camera(force_on=True))

    def _ensure_alerts_dir(self) -> Path:
        p = (Path(__file__).resolve().parent / ALERTS_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_snapshot(self):
        if self._last_qimg is None:
            self.log_event("Скрин: нет кадра (камера не запущена).")
            return
        out_dir = self._ensure_alerts_dir()
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "fire" if self._vision_fire else "predator" if self._vision_predator else "frame"
        path = out_dir / f"{ts}_{tag}.png"
        ok = QPixmap.fromImage(self._last_qimg).save(str(path))
        if ok:
            self.log_event(f"Скрин сохранён: {path}")
        else:
            self.log_event("Скрин: не удалось сохранить файл.")

    def save_clip_5s(self):
        if cv2 is None:
            self.log_event("Клип: cv2 недоступен.")
            return
        now = time.time()
        window_s = 5.0
        frames = [(ts, bgr) for (ts, bgr) in list(self._vision_frame_buf) if (now - ts) <= window_s]
        if len(frames) < 5:
            self.log_event("Клип: мало кадров (запусти камеру и подожди пару секунд).")
            return

        out_dir = self._ensure_alerts_dir()
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "fire" if self._vision_fire else "predator" if self._vision_predator else "vision"
        path = out_dir / f"{ts}_{tag}_5s.mp4"

        first = frames[0][1]
        h, w = first.shape[:2]
        fps = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wri = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        try:
            for _ts, bgr in frames:
                if bgr.shape[0] != h or bgr.shape[1] != w:
                    bgr = cv2.resize(bgr, (w, h))
                wri.write(bgr)
        finally:
            wri.release()
        self.log_event(f"Клип сохранён: {path}")


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
        # Compute smoothed chicken count for the camera view
        if self._vision_chicken_recent:
            sorted_counts = sorted(self._vision_chicken_recent)
            median_count = sorted_counts[len(sorted_counts) // 2]
            self.cam_chicken.setText(f"CHICKEN: {median_count}")
        else:
            self.cam_chicken.setText("CHICKEN: —")

        # Vision alarm banner on camera tab
        if self._vision_alarm_active and not self._vision_alarm_ack:
            if self._vision_fire:
                self.cam_alarm.setText("VISION: ПОЖАР")
                self.cam_alarm.setStyleSheet(css_vision_alarm("#dc3545"))
            else:
                self.cam_alarm.setText("VISION: ХИЩНИК В ХЛЕВУ")
                self.cam_alarm.setStyleSheet(css_vision_alarm("#fd7e14"))
        else:
            self.cam_alarm.setText("VISION: НОРМА")
            self.cam_alarm.setStyleSheet(css_vision_alarm("#198754"))

        # Priority: serial stale -> vision fire -> vision predator -> sensor fire -> normal
        if self._last_rx_wall and (now - self._last_rx_wall) > STALE_AFTER_SECONDS:
            self.banner.setText("НЕТ ДАННЫХ — проверь питание/кабель/COM-порт")
            self.banner.setStyleSheet(css_banner("#6c757d"))
            self.advice.setText(
                "1) Проверь USB-кабель и питание платы.\n"
                "2) Убедись, что COM-порт не занят (Serial Monitor должен быть закрыт).\n"
                "3) Если не помогло — переподключи кабель."
            )
        elif self._vision_alarm_active and not self._vision_alarm_ack and self._vision_fire:
            self.banner.setText("ПОЖАР (VISION) — действуй немедленно!")
            self.banner.setStyleSheet(css_banner("#dc3545"))
            self.advice.setText(
                "1) Останови робота и осмотри загон.\n"
                "2) Проверь сено/проводку/обогреватели.\n"
                "3) Если есть дым/огонь — тушение/112."
            )
        elif self._vision_alarm_active and not self._vision_alarm_ack and self._vision_predator:
            self.banner.setText("ХИЩНИК В ХЛЕВУ (VISION) — проверь загон!")
            self.banner.setStyleSheet(css_banner("#fd7e14"))
            self.advice.setText(
                "1) Останови робота и проверь загон.\n"
                "2) Убедись, что животные в безопасности.\n"
                "3) При необходимости — закрой проходы/подними тревогу."
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
            base = "Система работает. Следи за «Огонь/дым» и «Наклон» — это самые критичные события."
            if self._vision_reco_lines:
                extra = "\n\nVISION:\n" + "\n".join(f"- {t}" for t in self._vision_reco_lines)
                self.advice.setText(base + extra)
            else:
                self.advice.setText(base)

        # Vision beep + popup until acknowledged
        if self._vision_alarm_active and not self._vision_alarm_ack and (now - self._vision_last_beep) > 1.0:
            QtWidgets.QApplication.beep()
            self._vision_last_beep = now

        if self._vision_alarm_active and not self._vision_alarm_ack and (now - self._vision_last_popup) > 25.0:
            self._vision_last_popup = now
            if self._vision_fire:
                QtWidgets.QMessageBox.critical(
                    self,
                    "VISION: пожар",
                    "Модель обнаружила пожар/огонь.\n"
                    "Проверь загон и останови робота при необходимости.",
                )
            elif self._vision_predator:
                QtWidgets.QMessageBox.warning(
                    self,
                    "VISION: хищник в хлеву",
                    "Модель обнаружила хищника (fox/marten/wolf).\n"
                    "Проверь загон и животных.",
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
            try:
                if self._video_thread is not None:
                    self._video_thread.stop()
                    self._video_thread.wait(1500)
            except Exception:
                pass
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

