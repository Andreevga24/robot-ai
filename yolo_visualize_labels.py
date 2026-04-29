from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def clamp_box(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def main() -> int:
    ap = argparse.ArgumentParser(description="Render YOLO labels as bbox overlays for quick inspection.")
    ap.add_argument("--data", default="yolo_dataset", help="YOLO dataset root containing images/ and labels/.")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to preview.")
    ap.add_argument("--n", type=int, default=16, help="How many images to preview.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--out", default="", help="Output folder (default: <data>/preview/<split>).")
    args = ap.parse_args()

    data = Path(args.data)
    images_dir = data / "images" / args.split
    labels_dir = data / "labels" / args.split
    out_dir = Path(args.out) if args.out else (data / "preview" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
    if not images:
        raise SystemExit(f"No images found in {images_dir.resolve()}")

    random.seed(args.seed)
    sample = images if len(images) <= args.n else random.sample(images, args.n)

    rendered = 0
    missing_labels = 0
    total_boxes = 0

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        boxes = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = parts[:5]
            try:
                xc_f = float(xc)
                yc_f = float(yc)
                bw_f = float(bw)
                bh_f = float(bh)
            except ValueError:
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(xc_f, yc_f, bw_f, bh_f, w, h)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            boxes.append((x1, y1, x2, y2))

        total_boxes += len(boxes)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
            img,
            f"boxes: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        rendered += 1

    print(f"rendered: {rendered}")
    print(f"missing_labels: {missing_labels}")
    print(f"total_boxes_in_preview: {total_boxes}")
    print(f"out_dir: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

