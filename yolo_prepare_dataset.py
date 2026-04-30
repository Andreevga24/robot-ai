from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloBox:
    xc: float
    yc: float
    w: float
    h: float

    def to_line(self, cls_idx: int) -> str:
        return f"{cls_idx} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def points_to_bbox(points: list[list[float]], img_w: int, img_h: int) -> YoloBox | None:
    if not points:
        return None
    xs = [p[0] for p in points if isinstance(p, list) and len(p) >= 2]
    ys = [p[1] for p in points if isinstance(p, list) and len(p) >= 2]
    if not xs or not ys:
        return None

    x1 = clamp(min(xs), 0.0, float(img_w))
    y1 = clamp(min(ys), 0.0, float(img_h))
    x2 = clamp(max(xs), 0.0, float(img_w))
    y2 = clamp(max(ys), 0.0, float(img_h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None

    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h

    # Safety clamp in case of weird annotations at borders
    return YoloBox(
        xc=clamp(xc, 0.0, 1.0),
        yc=clamp(yc, 0.0, 1.0),
        w=clamp(w, 0.0, 1.0),
        h=clamp(h, 0.0, 1.0),
    )


def find_image(json_path: Path, image_path_value: str) -> Path | None:
    """
    Tries common locations. Many LabelMe exports keep `imagePath` as a basename.
    """
    if not image_path_value:
        return None

    p = Path(image_path_value)
    candidates: list[Path] = []

    # If it's an absolute path, try it directly first.
    if p.is_absolute():
        candidates.append(p)

    # Relative to the JSON file (same dir)
    candidates.append(json_path.parent / p.name)
    # Common subfolders
    candidates.append(json_path.parent / "images" / p.name)
    candidates.append(json_path.parent / "Images" / p.name)
    candidates.append(json_path.parent / "JPEGImages" / p.name)

    for c in candidates:
        if c.exists():
            return c
    return None


def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"names: {class_names!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_darknet_files(out_dir: Path, class_names: list[str]) -> None:
    """
    Emit the classic YOLOv1/Darknet training files:
    - obj.names: class names (one per line)
    - obj.data: metadata pointing to train/val lists and names
    - train.txt / val.txt: absolute image paths (one per line)
    """
    (out_dir / "obj.names").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    train_images_dir = (out_dir / "images" / "train").resolve()
    val_images_dir = (out_dir / "images" / "val").resolve()

    def list_images(d: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

    train_list = [p.resolve().as_posix() for p in list_images(train_images_dir)]
    val_list = [p.resolve().as_posix() for p in list_images(val_images_dir)]

    (out_dir / "train.txt").write_text("\n".join(train_list) + ("\n" if train_list else ""), encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_list) + ("\n" if val_list else ""), encoding="utf-8")

    (out_dir / "obj.data").write_text(
        "\n".join(
            [
                f"classes = {len(class_names)}",
                f"train = {(out_dir / 'train.txt').resolve().as_posix()}",
                f"valid = {(out_dir / 'val.txt').resolve().as_posix()}",
                f"names = {(out_dir / 'obj.names').resolve().as_posix()}",
                f"backup = {(out_dir / 'backup').resolve().as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def discover_classes(json_files: list[Path]) -> list[str]:
    labels: set[str] = set()
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for s in data.get("shapes") or []:
            label = s.get("label")
            if isinstance(label, str) and label.strip():
                labels.add(label.strip())
    # Stable ordering for reproducibility
    return sorted(labels)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert LabelMe JSON polygons/linestrips to YOLO bbox dataset (Ultralytics)."
    )
    ap.add_argument("--src", default="dataset", help="Folder with LabelMe .json files (and images).")
    ap.add_argument("--out", default="yolo_dataset", help="Output folder.")
    ap.add_argument("--val", type=float, default=0.2, help="Validation split ratio.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output folder. If not set, images are not copied.",
    )
    ap.add_argument(
        "--darknet",
        action="store_true",
        help="Also write YOLOv1/Darknet files: train.txt/val.txt/obj.names/obj.data",
    )
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"

    # LabelMe exports are often nested by class/folder; include all JSONs recursively.
    json_files = sorted(src_dir.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json files found in {src_dir.resolve()}")

    random.seed(args.seed)
    random.shuffle(json_files)
    n_val = max(1, int(round(len(json_files) * args.val)))
    val_set = set(json_files[:n_val])

    class_names = discover_classes(json_files)
    if not class_names:
        raise SystemExit("No class labels discovered in JSON shapes[].label")
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    missing_images: list[str] = []
    kept = 0
    skipped_empty = 0
    total_shapes = 0

    for jp in json_files:
        data = json.loads(jp.read_text(encoding="utf-8"))
        img_w = int(data.get("imageWidth") or 0)
        img_h = int(data.get("imageHeight") or 0)
        image_path_value = str(data.get("imagePath") or "")
        img_path = find_image(jp, image_path_value)

        if not img_path or not img_path.exists():
            missing_images.append(f"{jp.name} -> {image_path_value}")
            continue

        split = "val" if jp in val_set else "train"
        out_labels_dir = out_labels_val if split == "val" else out_labels_train
        out_images_dir = out_images_val if split == "val" else out_images_train

        shapes = data.get("shapes") or []
        lines: list[str] = []
        for s in shapes:
            total_shapes += 1
            label = s.get("label")
            if label not in class_to_idx:
                continue
            points = s.get("points") or []
            box = points_to_bbox(points, img_w=img_w, img_h=img_h)
            if box is None:
                continue
            lines.append(box.to_line(class_to_idx[label]))

        if not lines:
            skipped_empty += 1
            continue

        out_label_path = out_labels_dir / f"{img_path.stem}.txt"
        out_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if args.copy_images:
            shutil.copy2(img_path, out_images_dir / img_path.name)

        kept += 1

    write_data_yaml(out_dir, class_names)
    if args.darknet:
        (out_dir / "backup").mkdir(parents=True, exist_ok=True)
        write_darknet_files(out_dir, class_names)

    summary = [
        f"json_files: {len(json_files)}",
        f"kept: {kept}",
        f"skipped_empty_or_invalid: {skipped_empty}",
        f"missing_images: {len(missing_images)}",
        f"total_shapes_seen: {total_shapes}",
        f"output: {out_dir.resolve()}",
    ]
    (out_dir / "prepare_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if missing_images:
        (out_dir / "missing_images.txt").write_text("\n".join(missing_images) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print("\nclasses:")
    print("\n".join([f"- {n}" for n in class_names]))
    if missing_images:
        print(f"\nMissing images list written to: {(out_dir / 'missing_images.txt').resolve()}")
        print(
            "Put the corresponding .jpg/.png files next to the .json (or in dataset/images/), "
            "then re-run with --copy-images."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

