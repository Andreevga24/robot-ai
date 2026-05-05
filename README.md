# Robot AI: мониторинг Arduino + YOLO датасет/детекция

Репозиторий объединяет два направления:
- **Serial dashboard**: мониторинг датчиков с Arduino (PyQt6, тревоги, история).
- **YOLO pipeline**: подготовка датасета из LabelMe JSON, обучение и инференс (Ultralytics).

## Структура проекта
- `qt_serial_dashboard.py` — основной UI для мониторинга по USB-Serial.
- `sketch_apr27b/sketch_apr27b.ino` — скетч Arduino (источник данных).
- `plot_light_channel.py` — простой legacy-график на matplotlib.
- `yolo_prepare_dataset.py` — конвертация LabelMe JSON -> YOLO bbox.
- `yolo_predict_detected_only.py` — инференс YOLO с сохранением только кадров с детекциями.
- `yolo_webcam_detect.py` — realtime-детекция с веб-камеры/потока.
- `yolo_visualize_labels.py` — визуализация bbox-разметки YOLO.
- `requirements.txt` — зависимости dashboard.
- `requirements_yolo.txt` — зависимости YOLO.
- `dataset/` — исходные LabelMe JSON и изображения.
- `yolo_dataset/` — подготовленный датасет YOLO (`images/`, `labels/`, `data.yaml`).

## Быстрый старт

### 1) Dashboard (Arduino -> ПК)
Установка:
```bash
py -m pip install -r requirements.txt
```

Запуск:
1. Порт USB-Serial: на **Windows** по умолчанию `COM7` (см. `PORT` / `SERIAL_PORT` ниже). На **Linux/Pi** обычно подхватывается `/dev/ttyUSB0` или `/dev/ttyACM0`.
2. Запусти:
```bash
py qt_serial_dashboard.py
```

**Переменные окружения (Serial / сеть):**
- `SERIAL_PORT` — явно задать порт (пример: `COM3`, `/dev/ttyACM0`).
- `SERIAL_TCP=host:port` — читать данные Arduino, проброшенные с Pi по TCP (см. раздел ниже про `socat`).
- `SERIAL_STALE_SECONDS` — сколько секунд без пакетов считать линию «мёртвой» (для TCP по Wi‑Fi полезно увеличить; по умолчанию для TCP ~35с, для USB ~3с).
- Дополнительно для TCP-клиента: `SERIAL_TCP_RECONNECT_DELAY`, `SERIAL_TCP_RECV_TIMEOUT`, `SERIAL_TCP_CLOSE_GRACE_MS`.

**Переменные окружения (камера / YOLO в панели):**
- `VISION_ENABLED` — `0` чтобы отключить камеру/YOLO (полезно на слабом железе).
- `VISION_SOURCE` — стартовый индекс камеры (`0`, `1`, …) или URL потока (если поддерживается OpenCV на вашей системе). В интерфейсе индекс можно сменить спинбоксом «камера №…»; после изменения нажмите **Применить** (камера перезапустится).
- `VISION_LOOP_TAIL_SLEEP` — пауза после кадра на Pi (например `0.04`), чтобы разгрузить CPU/USB под Serial.

### Raspberry Pi 4 (Dashboard без YOLO)
На Raspberry Pi часто не хватает места/памяти для установки `opencv-python` и `ultralytics`. Если вам нужен только мониторинг Arduino (Serial dashboard), используйте облегчённые зависимости:

```bash
cd ~/robot-ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_pi_dashboard.txt
```

Порт можно задать переменной окружения (пример для Arduino):

```bash
export SERIAL_PORT=/dev/ttyACM0
python3 qt_serial_dashboard.py
```

### Arduino на Raspberry Pi, dashboard на ПК (Serial по сети)
На **Pi** пробросьте UART/USB-Arduino в TCP (пример порт `5001`; замените `/dev/ttyACM0` на свой):

```bash
sudo apt install -y socat
socat TCP-LISTEN:5001,reuseaddr,fork FILE:/dev/ttyACM0,b9600,raw
```

На **Windows** в том же репозитории запустите панель с адресом Pi:

```powershell
set SERIAL_TCP=192.168.x.x:5001
py qt_serial_dashboard.py
```

Камера, подключённая к Pi, на ПК по умолчанию **не видна** как «камера 0». Для YOLO на ПК нужен **сетевой поток** с Pi (RTSP/MJPEG) и тогда в коде задаётся источник как URL, либо обработка видео остаётся на Pi.

YOLO-детекция в панели (вкладка **Камера**):
- По умолчанию используется `VISION_SOURCE="0"` (первая камера; можно переопределить через окружение).
- Путь к весам задаётся в `VISION_MODEL_PATH` (в репозитории по умолчанию это `runs/detect/train_updated_115/weights/best.pt`; при обучении у себя подставьте свой `runs/detect/.../weights/best.pt`).
- **Тревоги (VISION)** — это отдельный канал от датчиков Serial:
  - `fire` при уверенности ≥ `conf` тревоги и удержании `VISION_ALARM_HOLD_SECONDS` → **ПОЖАР (VISION)** (баннер/звук/подтверждение).
  - `fox`, `marten`, `wolf`, `volf` при тех же условиях → **ХИЩНИК В ХЛЕВУ (VISION)**.
  - `chicken` → счётчик **CHICKEN: N** (со сглаживанием треков на кадре).

**Операторские подсказки (не тревога):** при устойчивых детекциях панель может выводить текст в журнал и в блок «Что делать сейчас» (при общем статусе «норма»):
- `cow_lumpy` → «Возможно зарожение»
- `sheep_wool` → «Время стрижки овец»
- сумма `round hay` + `square hay` меньше 3 на кадре (при условии, что сено вообще видно) → «Пополните запасы сена»
- `corn` меньше 3 на кадре (при условии, что кукуруза вообще видна) → «пополните запасы пшена»

**Снижение ложных срабатываний (подсказки):** для подсчёта/сообщений используются дополнительные фильтры в коде — см. `VISION_CONF_BY_CLASS`, `VISION_MIN_AREA_RATIO_BY_CLASS`, ROI у края кадра, а также кратковременная «устойчивость по кадрам» (`VISION_STAB_*`). Настройте пороги под свою камеру и типичный ракурс.

Важно: перед запуском закрой `Serial Monitor` / `Serial Plotter` в Arduino IDE, иначе COM-порт будет занят.

### 2) YOLO (подготовка + обучение + детекция)
Установка:
```bash
py -m pip install -r requirements_yolo.txt
```

Подготовка датасета (из `dataset/`, включая вложенные папки с LabelMe JSON):
```bash
py yolo_prepare_dataset.py --src dataset --out yolo_dataset --val 0.2 --seed 42 --copy-images
```

Датасет для дообучения YOLO11 (актуальный пример)

После обновления исходного `dataset/` удобно собирать отдельный выходной датасет, чтобы не смешивать версии:

```bash
py yolo_prepare_dataset.py --src dataset --out yolo_dataset_yolov1_updated --val 0.2 --seed 42 --copy-images --darknet
```

Файлы/папки, которые появятся:
- `yolo_dataset_yolov1_updated/images/{train,val}` — изображения
- `yolo_dataset_yolov1_updated/labels/{train,val}` — YOLO разметка (`.txt`)
- `yolo_dataset_yolov1_updated/data.yaml` — конфиг для Ultralytics
- (опционально, `--darknet`) `train.txt`, `val.txt`, `obj.names`, `obj.data` — файлы для YOLOv1/Darknet

Список классов берётся из `data.yaml` или `obj.names`. Пример актуального набора (может меняться при обновлении датасета):
`chicken, corn, cow_lumpy, cow_wound, fire, fox, marten, round hay, sheared sheep, sheep_wool, square hay, volf, wolf`

Обучение (Ultralytics):
```bash
yolo detect train data=yolo_dataset_yolov1_updated/data.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=8 device=0
```

Если нет GPU:
```bash
yolo detect train data=yolo_dataset_yolov1_updated/data.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=4 device=cpu
```

Дообучение (продолжить с чекпоинта `last.pt`):
```bash
yolo detect train data=yolo_dataset_yolov1_updated/data.yaml model=runs/detect/train/weights/last.pt imgsz=640 epochs=10 batch=4 device=cpu
```

Детекция:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images conf=0.25
```

Пример теста на папке `test1`:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=test1 conf=0.25
```

Детекция с веб-камеры (realtime):
```bash
py yolo_webcam_detect.py --model runs/detect/train/weights/best.pt --source 0 --show
```

Сохранить видео с разметкой:
```bash
py yolo_webcam_detect.py --model runs/detect/train/weights/best.pt --source 0 --show --save
```

## Dashboard: формат Serial данных
Ожидаемая строка от Arduino:
`L:<число> <статус> | F:<число> <статус> | X:<число> Y:<число> Z:<число>`

Пример:
`L:512 NORM | F:120 SAFE | X:2 Y:-1 Z:0`

В интерфейсе используется наклон по `Y/Z` (ось `X` игнорируется).

## Dashboard: логика тревог
- По огню/дыму: `F > 250` на протяжении >= 2 сек -> пожарная тревога.
- По наклону: предупреждение при `|Y|` или `|Z| >= 25`, тревога при `>= 45`.
- В режиме `ОХРАНА` отслеживается отклонение от baseline:
  - предупреждение при `delta >= 8`
  - тревога при `delta >= 15`
- **VISION (YOLO):** классы `fire` и хищники (`fox`, `marten`, `wolf`, `volf`) при превышении порога уверенности и удержании срабатывания (`VISION_ALARM_HOLD_SECONDS`) — отдельные визуальные/звуковые тревоги в панели.

## Логи и артефакты
- Dashboard-логи: `data_logs/sensors_YYYY-MM-DD.csv`
- YOLO-артефакты обучения: `runs/detect/...`
- После подготовки датасета:
  - `<out>/prepare_summary.txt`
  - `<out>/missing_images.txt` (если не все изображения найдены)

## Диагностика проблем
- `NO DATA / НЕТ ДАННЫХ` в dashboard: проверь питание Arduino, USB-кабель (должен передавать данные), порт и что IDE не заняла COM.
- `PermissionError / Access is denied`: порт занят другим приложением.
- В YOLO-конвертации есть `missing_images > 0`: положи изображения рядом с `.json` в `dataset/` (или `dataset/images/`) и запусти подготовку снова.

