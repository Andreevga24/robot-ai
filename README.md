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

Важно: перед запуском закрой `Serial Monitor` / `Serial Plotter` в Arduino IDE, иначе COM-порт будет занят.

### 2) YOLO (подготовка + обучение + детекция)
Установка:
```bash
py -m pip install -r requirements_yolo.txt
```

Подготовка датасета:
```bash
py yolo_prepare_dataset.py --src dataset --out yolo_dataset --val 0.2 --seed 42 --copy-images
```

Обучение:
```bash
yolo detect train data=yolo_dataset/data.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=8 device=0
```

Если нет GPU:
```bash
yolo detect train data=yolo_dataset/data.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=4 device=cpu
```

Детекция:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images conf=0.25
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
  - `yolo_dataset/prepare_summary.txt`
  - `yolo_dataset/missing_images.txt` (если не все изображения найдены)

## Диагностика проблем
- `NO DATA / НЕТ ДАННЫХ` в dashboard: проверь питание Arduino, USB-кабель (должен передавать данные), порт и что IDE не заняла COM.
- `PermissionError / Access is denied`: порт занят другим приложением.
- В YOLO-конвертации есть `missing_images > 0`: положи изображения рядом с `.json` в `dataset/` (или `dataset/images/`) и запусти подготовку снова.

