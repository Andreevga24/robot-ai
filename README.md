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
1. В `qt_serial_dashboard.py` установи корректный порт, например `PORT = "COM7"`.
2. Запусти:
```bash
py qt_serial_dashboard.py
```

YOLO-детекция в панели (вкладка **Камера**):
- По умолчанию используется `VISION_SOURCE="0"` (одна камера).
- Модель берётся из `runs/detect/train/weights/best.pt` (см. `VISION_MODEL_PATH`).
- Логика тревог:
  - `fire` -> тревога **ПОЖАР (VISION)** (экран + звук).
  - `fox|marten|volf|wolf` -> тревога **ХИЩНИК В ХЛЕВУ (VISION)** (экран + звук).
  - `chicken` -> счётчик **CHICKEN: N** на текущем кадре (со сглаживанием).

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

