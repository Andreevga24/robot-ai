#!/usr/bin/env python3
"""
Разбор CSV логов датчиков (формат qt_serial_dashboard: unix_ts,t_s,L,F,X,Y,Z).

Примеры:
  py analyze_sensor_log.py data_logs/sensors_2026-05-03.csv
  py analyze_sensor_log.py data_logs/sensors_2026-05-03.csv --gap 20 --export-clean data_logs/clean_sessions
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Анализ разрывов и сессий в sensors_*.csv")
    ap.add_argument("csv_file", type=Path, help="Путь к CSV (например data_logs/sensors_2026-05-03.csv)")
    ap.add_argument(
        "--gap",
        type=float,
        default=15.0,
        metavar="SEC",
        help="Пауза между строками больше SEC считается разрывом сессии (по умолчанию 15)",
    )
    ap.add_argument(
        "--export-clean",
        type=Path,
        default=None,
        metavar="DIR",
        help="Каталог: сохранить отдельный CSV на каждую непрерывную сессию",
    )
    args = ap.parse_args()

    path: Path = args.csv_file
    if not path.is_file():
        raise SystemExit(f"Файл не найден: {path}")

    gap_s = float(args.gap)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Файл пустой.")
        return

    fields = rows[0].keys()
    expected = {"unix_ts", "t_s", "L", "F", "X", "Y", "Z"}
    if not expected.issubset(set(fields)):
        print("Предупреждение: необычные колонки:", list(fields))

    ts = [float(r["unix_ts"]) for r in rows]

    sessions: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > gap_s:
            sessions.append((start, i - 1))
            start = i
    sessions.append((start, len(rows) - 1))

    total_span = ts[-1] - ts[0]
    print(f"Файл: {path}")
    print(f"Строк: {len(rows)}  |  unix_ts: {ts[0]:.3f} … {ts[-1]:.3f}  |  охват ~{total_span/3600:.2f} ч")
    print(f"Порог разрыва: {gap_s:g} с -> сессий: {len(sessions)}")
    print()

    for si, (a, b) in enumerate(sessions):
        dur = ts[b] - ts[a]
        n = b - a + 1
        rate = n / dur if dur > 0 else 0.0
        gap_before = None
        if a > 0:
            gap_before = ts[a] - ts[a - 1]
        head = f"  #{si + 1}: строки {a + 1}–{b + 1}"
        if gap_before is not None:
            head += f"  (пауза перед сессией {gap_before:.1f} с)"
        print(f"{head}")
        print(f"       длительность {dur:.1f} с, точек {n}, ~{rate:.2f} строк/с")

    if args.export_clean:
        out_dir: Path = args.export_clean
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        for si, (a, b) in enumerate(sessions):
            chunk = rows[a : b + 1]
            outp = out_dir / f"{stem}_session{si + 1}.csv"
            with outp.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(chunk)
            print()
            print(f"Записано: {outp}  ({len(chunk)} строк)")


if __name__ == "__main__":
    main()
