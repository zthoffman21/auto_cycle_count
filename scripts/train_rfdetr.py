from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR segmentation for totes.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rfdetr import RFDETRSegSmall

    _validate_categories(args.dataset)
    model = RFDETRSegSmall()
    model.train(
        dataset_dir=str(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=str(args.output),
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.005,
        eval_interval=2,
        compute_val_loss=False,
        fp16_eval=True,
    )

    checkpoint = args.output / "checkpoint_best_total.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"training did not produce {checkpoint}")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, args.model_output)
    print(f"saved={args.model_output}")


def _validate_categories(dataset: Path) -> None:
    annotation_path = dataset / "train" / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO training annotations not found: {annotation_path}")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(annotation["categories"], key=lambda category: category["id"])
    names = [category["name"].lower() for category in categories]
    if names != ["tote", "cell"]:
        raise ValueError(
            "RF-DETR categories must be ordered as tote, cell so runtime class IDs are 0, 1; "
            f"found {names}"
        )


if __name__ == "__main__":
    main()
