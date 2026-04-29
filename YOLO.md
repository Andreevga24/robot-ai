## YOLO: дообучение на своих данных (Ultralytics YOLOv8/YOLO11)

В папке `dataset/` у тебя разметка в формате **LabelMe JSON** (фигуры `linestrip`). Для обучения детектора YOLO нужен формат **bbox** + картинки.

### 1) Установить зависимости

Из корня проекта:

```bash
py -m pip install -r requirements_yolo.txt
```

### 2) Подготовить датасет YOLO (конвертация JSON → bbox)

Скрипт:

```bash
py yolo_prepare_dataset.py --src dataset --out yolo_dataset --val 0.2 --seed 42 --copy-images
```

Результат:
- `yolo_dataset/images/train|val/` — картинки
- `yolo_dataset/labels/train|val/` — `.txt` разметка YOLO
- `yolo_dataset/data.yaml` — конфиг для обучения
- `yolo_dataset/missing_images.txt` — если каких-то картинок не найдено

Важно: сейчас в репозитории могут быть только `.json`. Если `missing_images > 0`, положи изображения рядом с JSON (в `dataset/`) или в `dataset/images/` и запусти ещё раз.

### 3) Дообучение

Вариант через CLI:

```bash
yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=8 device=0
```

Если нет GPU, используй `device=cpu` (будет медленно):

```bash
yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=4 device=cpu
```

### 4) Детекция на своих изображениях

После обучения веса будут в `runs/detect/train/weights/best.pt`.

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images conf=0.25
```

