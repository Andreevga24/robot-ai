"""
Графики и схемы в стиле ML-дашборда, но на данных этого репозитория.

Источники данных:
  - Ultralytics YOLO: runs/detect/<run>/results.csv
  - Классы и датасет: yolo_dataset_yolov1_updated/data.yaml

Примеры:
  python project_visualizations.py
  python project_visualizations.py --results runs/detect/train_updated_115/results.csv --out-dir project_plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import yaml


def _load_results_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Пустой файл: {path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)

    return {
        "epoch": col("epoch"),
        "train_box": col("train/box_loss"),
        "train_cls": col("train/cls_loss"),
        "train_dfl": col("train/dfl_loss"),
        "val_box": col("val/box_loss"),
        "val_cls": col("val/cls_loss"),
        "val_dfl": col("val/dfl_loss"),
        "precision": col("metrics/precision(B)"),
        "recall": col("metrics/recall(B)"),
    }


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_training_dashboard(
    results_csv: Path,
    out_path: Path,
    dpi: int = 120,
) -> None:
    d = _load_results_csv(results_csv)
    epoch = d["epoch"]
    train_loss = d["train_box"] + d["train_cls"] + d["train_dfl"]
    val_loss = d["val_box"] + d["val_cls"] + d["val_dfl"]
    prec = d["precision"]
    rec = d["recall"]

    bg = "#1e1e1e"
    grid = "#3a3a3a"
    train_color = "#4da3ff"
    val_color = "#6ecf68"

    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "axes.edgecolor": grid,
            "axes.labelcolor": "#e0e0e0",
            "text.color": "#e0e0e0",
            "xtick.color": "#c8c8c8",
            "ytick.color": "#c8c8c8",
            "grid.color": grid,
            "grid.alpha": 0.55,
            "legend.facecolor": "#2a2a2a",
            "legend.edgecolor": grid,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=dpi)
    fig.patch.set_facecolor(bg)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, linestyle="-", linewidth=0.6)

    # Loss: сумма box + cls + dfl (как единая «функция потерь» по эпохам)
    axes[0].plot(epoch, train_loss, color=train_color, linewidth=2, label="Тренировочная сумма потерь")
    axes[0].plot(epoch, val_loss, color=val_color, linewidth=2, label="Проверочная сумма потерь")
    axes[0].set_title("Функция потерь (box + cls + dfl)", fontsize=11)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Сумма потерь")
    axes[0].legend(loc="upper right", fontsize=8)
    tl, vl = float(train_loss[-1]), float(val_loss[-1])
    axes[0].text(
        0.02,
        -0.22,
        f"Тренировочная сумма потерь = {tl:.4f}\nПроверочная сумма потерь = {vl:.4f}",
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        color="#cfcfcf",
    )

    # Метрики: precision и recall (обе считаются на валидации в Ultralytics)
    axes[1].plot(epoch, prec, color=train_color, linewidth=2, label="Precision (B)")
    axes[1].plot(epoch, rec, color=val_color, linewidth=2, label="Recall (B)")
    axes[1].set_title("Метрики детекции (валидация)", fontsize=11)
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение (0–1)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="lower right", fontsize=8)
    tp, vp = float(prec[-1]), float(rec[-1])
    axes[1].text(
        0.02,
        -0.22,
        f"Precision (B) = {tp:.4f}\nRecall (B) = {vp:.4f}",
        transform=axes[1].transAxes,
        fontsize=9,
        va="top",
        color="#cfcfcf",
    )

    run_name = results_csv.parent.name
    fig.suptitle(
        f"YOLO · {run_name} · {results_csv.name}",
        fontsize=12,
        color="#f0f0f0",
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def plot_pipeline_schematic(
    data_yaml: Path,
    out_path: Path,
    dpi: int = 120,
    model_label: str = "YOLO (Ultralytics)",
) -> None:
    cfg = _read_yaml(data_yaml)
    names = cfg.get("names") or []
    n_cls = len(names) if isinstance(names, list) else int(cfg.get("nc", 0))

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Сетка как на скриншотах
    for g in np.linspace(0, 10, 21):
        ax.axhline(g, color="#e8e8e8", linewidth=0.6, zorder=0)
        ax.axvline(g, color="#e8e8e8", linewidth=0.6, zorder=0)

    nodes = [
        ("Датасет\nrobot-ai", "train / val\nYOLO format"),
        ("Загрузка\n640×640", "батч, аугментации"),
        ("Backbone\nCSPDarknet", "извлечение признаков"),
        ("Neck\nPAN-FPN", "мультимасштаб"),
        ("Head\nDetect", f"{n_cls} классов"),
        ("Постобработка", "NMS, порог conf"),
        ("Выход", "боксы + классы"),
    ]

    x0, y_top = 1.1, 8.6
    w, h = 1.55, 0.95
    dy = 1.15

    def draw_node(title: str, subtitle: str, y: float) -> tuple[float, float, float, float]:
        x = x0
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#9a9a9a",
            facecolor="#ffffff",
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#111111",
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=7,
            color="#444444",
            zorder=3,
        )
        # порты
        port_r = 0.06
        ax.add_patch(
            plt.Circle((x, y + h / 2), port_r, facecolor="#ffffff", edgecolor="#888888", linewidth=0.8, zorder=4)
        )
        ax.add_patch(
            plt.Circle((x + w, y + h / 2), port_r, facecolor="#ffffff", edgecolor="#888888", linewidth=0.8, zorder=4)
        )
        return x, y, w, h

    boxes: list[tuple[float, float, float, float]] = []
    y_cur = y_top
    for title, sub in nodes:
        boxes.append(draw_node(title, sub, y_cur))
        y_cur -= dy

    # Отдельный блок «модель» справа в стиле первого скрина
    mx, my, mw, mh = 6.2, 6.5, 3.2, 1.4
    mbox = FancyBboxPatch(
        (mx, my),
        mw,
        mh,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        linewidth=1.2,
        edgecolor="#9a9a9a",
        facecolor="#ffffff",
        zorder=2,
    )
    ax.add_patch(mbox)
    ax.text(mx + mw / 2, my + mh * 0.72, model_label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(
        mx + mw / 2,
        my + mh * 0.38,
        f"Классов: {n_cls}",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )
    ax.text(
        mx + mw / 2,
        my + mh * 0.14,
        data_yaml.name,
        ha="center",
        va="center",
        fontsize=7,
        color="#666666",
    )
    ax.add_patch(plt.Circle((mx, my + mh / 2), 0.07, fc="#fff", ec="#888", lw=0.8, zorder=4))
    ax.add_patch(plt.Circle((mx + mw, my + mh / 2), 0.07, fc="#fff", ec="#888", lw=0.8, zorder=4))

    # Стрелки вниз между узлами слева
    for i in range(len(boxes) - 1):
        x, y, w, h = boxes[i]
        x2, y2, w2, h2 = boxes[i + 1]
        ax.annotate(
            "",
            xy=(x + w / 2, y2 + h2 + 0.02),
            xytext=(x + w / 2, y + h - 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.2, mutation_scale=12),
        )

    # Стрелка от backbone к блоку модели (условная связь)
    bx, by, bw, bh = boxes[2]
    ax.annotate(
        "",
        xy=(mx + 0.05, my + mh * 0.5),
        xytext=(bx + bw + 0.08, by + bh * 0.5),
        arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=1.0, linestyle=(0, (4, 3)), mutation_scale=10),
    )

    short = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
    ax.text(
        0.5,
        0.35,
        "Классы: " + (short if short else str(names)),
        ha="center",
        fontsize=8,
        color="#333333",
        transform=ax.transAxes,
    )

    ax.set_title("Схема пайплайна детекции (данные проекта robot-ai)", fontsize=12, pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Графики обучения и схема пайплайна по данным проекта.")
    root = Path(__file__).resolve().parent
    ap.add_argument(
        "--results",
        type=Path,
        default=root / "runs/detect/train_updated_115/results.csv",
        help="Путь к results.csv от Ultralytics.",
    )
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=root / "yolo_dataset_yolov1_updated/data.yaml",
        help="data.yaml датасета.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "project_plots",
        help="Каталог для PNG.",
    )
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    results = args.results.resolve()
    data_yaml = args.data_yaml.resolve()
    out_dir = args.out_dir.resolve()

    if not results.is_file():
        print(f"Не найден results.csv: {results}")
        return 1
    if not data_yaml.is_file():
        print(f"Не найден data.yaml: {data_yaml}")
        return 1

    plot_training_dashboard(results, out_dir / "training_dashboard.png", dpi=args.dpi)
    print(f"Записано: {out_dir / 'training_dashboard.png'}")

    plot_pipeline_schematic(data_yaml, out_dir / "pipeline_schematic.png", dpi=args.dpi)
    print(f"Записано: {out_dir / 'pipeline_schematic.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
