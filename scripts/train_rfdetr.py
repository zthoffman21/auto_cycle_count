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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="RF-DETR dataloader workers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        help="PyTorch Lightning training log interval.",
    )
    parser.add_argument(
        "--progress-bar",
        choices=("tqdm", "rich", "none"),
        default="tqdm",
        help="PyTorch Lightning progress bar style.",
    )
    parser.add_argument(
        "--sanity-val-steps",
        type=int,
        default=0,
        help="Lightning validation sanity-check batches before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rfdetr import RFDETRSegSmall
    import rfdetr.training as rfdetr_training

    _validate_categories(args.dataset)
    _set_trainer_defaults(
        rfdetr_training,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.sanity_val_steps,
    )
    model = RFDETRSegSmall()
    progress_bar = None if args.progress_bar == "none" else args.progress_bar
    interrupted = False
    try:
        model.train(
            dataset_dir=str(args.dataset),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            num_workers=args.num_workers,
            progress_bar=progress_bar,
            output_dir=str(args.output),
            early_stopping=True,
            early_stopping_patience=5,
            early_stopping_min_delta=0.005,
            eval_interval=2,
            compute_val_loss=False,
            fp16_eval=True,
        )
    except KeyboardInterrupt:
        interrupted = True

    checkpoint = _find_best_checkpoint(args.output)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, args.model_output)
    print(f"saved={args.model_output}")
    if interrupted:
        print("training interrupted; copied the latest best checkpoint before exiting")


def _set_trainer_defaults(
    rfdetr_training: object,
    *,
    log_every_n_steps: int,
    num_sanity_val_steps: int,
) -> None:
    if log_every_n_steps < 1:
        raise ValueError("--log-every-n-steps must be at least 1")
    if num_sanity_val_steps < 0:
        raise ValueError("--sanity-val-steps must be at least 0")

    original_build_trainer = rfdetr_training.build_trainer

    def build_trainer_with_defaults(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("log_every_n_steps", log_every_n_steps)
        kwargs.setdefault("num_sanity_val_steps", num_sanity_val_steps)
        return original_build_trainer(*args, **kwargs)

    rfdetr_training.build_trainer = build_trainer_with_defaults


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


def _find_best_checkpoint(output: Path) -> Path:
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
    ):
        checkpoint = output / name
        if checkpoint.is_file():
            return checkpoint
    candidates = sorted(output.glob("checkpoint_best*.pth"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(
        "training did not produce an RF-DETR best checkpoint under "
        f"{output}; expected checkpoint_best_total.pth, checkpoint_best_ema.pth, "
        "or checkpoint_best_regular.pth"
    )


if __name__ == "__main__":
    main()
