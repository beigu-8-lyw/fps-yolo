"""Fine-tune YOLO and evaluate the best checkpoint on held-out episodes."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/doom/dataset.yaml"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="doom", help="Run directory name under runs/")
    args = parser.parse_args()
    dataset = args.data.resolve()
    if not dataset.is_file():
        parser.error(f"Dataset does not exist: {dataset}. Run collect.py first.")

    model = YOLO(args.model)
    model.train(
        data=str(dataset), device=args.device, epochs=args.epochs,
        batch=args.batch, imgsz=args.imgsz, seed=args.seed,
        patience=10, workers=0, project=str(Path("runs").resolve()), name=args.name,
    )
    best_path = Path(model.trainer.best)
    print(f"Best checkpoint: {best_path}")
    metrics = YOLO(str(best_path)).val(
        data=str(dataset), split="test", device=args.device,
        batch=args.batch, imgsz=args.imgsz, workers=0,
        project=str(best_path.parent.parent), name="test",
    )
    print(f"Test mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
