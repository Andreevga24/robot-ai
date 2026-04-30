from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(raw: str) -> int | str:
    # "0" -> webcam index 0, otherwise treat as URL/device string.
    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Realtime YOLO detection from webcam/stream."
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/train/weights/best.pt"),
        help="Path to YOLO weights (.pt).",
    )
    ap.add_argument(
        "--source",
        type=str,
        default="0",
        help='Camera index or stream URL. Example: "0", "1", "rtsp://..."',
    )
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    ap.add_argument("--device", type=str, default="cpu", help='Device: "cpu", "0", etc.')
    ap.add_argument("--show", action="store_true", help="Show preview window.")
    ap.add_argument("--save", action="store_true", help="Save annotated video.")
    ap.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Output root directory when --save is enabled.",
    )
    ap.add_argument(
        "--name",
        type=str,
        default="webcam_detect",
        help="Output subdirectory name when --save is enabled.",
    )
    args = ap.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"Model file not found: {model_path}")
        return 1

    src = parse_source(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {args.source}")
        return 1

    out_writer: cv2.VideoWriter | None = None
    out_path: Path | None = None

    if args.save:
        out_dir = (args.project / args.name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"webcam_{ts}.mp4"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))

    model = YOLO(str(model_path))
    frame_count = 0

    print("Press 'q' in the preview window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        annotated = result.plot()
        frame_count += 1

        if out_writer is not None:
            out_writer.write(annotated)

        if args.show:
            cv2.imshow("YOLO Webcam Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Frames processed: {frame_count}")
    if out_path is not None:
        print(f"Saved video: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
