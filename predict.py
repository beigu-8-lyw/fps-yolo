"""Detect enemies in one screenshot and print the result as JSON."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for path in (args.weights, args.image):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be between 0 and 1.")

    result = YOLO(str(args.weights)).predict(
        source=str(args.image), conf=args.confidence, imgsz=args.imgsz,
        device=args.device, verbose=False,
    )[0]
    boxes = result.boxes.cpu()
    detections = []
    for class_id, confidence, xywh in zip(
        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()
    ):
        x_center, y_center, width, height = xywh
        detections.append({
            "class": result.names[int(class_id)],
            "confidence": confidence,
            "bbox": {"cx": x_center, "cy": y_center, "w": width, "h": height},
            "horizontal_offset": x_center - 0.5,
        })
    print(json.dumps({"image": args.image.name, "detections": detections}, indent=2))


if __name__ == "__main__":
    main()
