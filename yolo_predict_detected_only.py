"""
Рекурсивный инференс YOLO: сохраняются только изображения, на которых есть детекции.
Прогресс пишется в --progress; при повторном запуске обрабатываются только незавершённые кадры.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO: рекурсивно обойти папку и сохранить только кадры с детекциями."
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("datesrt"),
        help="Корневая папка (включая вложенные). По умолчанию: datesrt",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/train5/weights/best.pt"),
        help="Веса .pt",
    )
    ap.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Каталог проекта Ultralytics",
    )
    ap.add_argument(
        "--name",
        type=str,
        default="predict_detected_only",
        help="Имя подпапки для сохранения (внутри --project)",
    )
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="JSON с обработанными путями (по умолчанию: <out>/.yolo_predict_progress.json)",
    )
    ap.add_argument(
        "--reset-progress",
        action="store_true",
        help="Игнорировать файл прогресса и начать с нуля",
    )
    args = ap.parse_args()

    src = args.source.resolve()
    if not src.is_dir():
        print(f"Папка не найдена: {src}", file=sys.stderr)
        return 1

    weights = args.model.resolve()
    if not weights.is_file():
        print(f"Нет файла весов: {weights}", file=sys.stderr)
        return 1

    paths = collect_images(src)
    if not paths:
        print(f"В {src} не найдено изображений ({', '.join(sorted(IMAGE_EXTS))}).", file=sys.stderr)
        return 1

    out_root = (args.project / args.name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    progress_path = args.progress
    if progress_path is None:
        progress_path = out_root / ".yolo_predict_progress.json"
    else:
        progress_path = progress_path.resolve()

    done: dict[str, str] = {}
    if not args.reset_progress and progress_path.is_file():
        try:
            done = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            done = {}

    pending = [p for p in paths if str(p.relative_to(src)) not in done]
    skipped = len(paths) - len(pending)

    model = YOLO(str(weights))
    saved = 0
    empty = 0

    def flush_progress() -> None:
        progress_path.write_text(json.dumps(done, ensure_ascii=False, indent=0), encoding="utf-8")

    # stream=True сохраняет порядок соответствия путям
    if pending:
        gen = model.predict(
            source=[str(p) for p in pending],
            stream=True,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        for path, r in zip(pending, gen):
            rel = str(path.relative_to(src))
            if r.boxes is None or len(r.boxes) == 0:
                empty += 1
                done[rel] = "empty"
                flush_progress()
                continue
            dest = out_root / path.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            bgr = r.plot()
            cv2.imwrite(str(dest), bgr)
            saved += 1
            done[rel] = "det"
            flush_progress()

    # Итоги по всему датасету (включая восстановленные из прогресса)
    total_det = sum(1 for v in done.values() if v == "det")
    total_empty = sum(1 for v in done.values() if v == "empty")

    print(f"Всего изображений в папке: {len(paths)}")
    if skipped:
        print(f"Пропущено (уже в прогрессе): {skipped}")
    print(f"Обработано в этом запуске: {len(pending)}")
    print(f"С детекциями (всего по прогрессу): {total_det}")
    print(f"Без детекций (всего по прогрессу): {total_empty}")
    print(f"Результаты: {out_root}")
    print(f"Прогресс: {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
