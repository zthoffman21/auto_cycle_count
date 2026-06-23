from __future__ import annotations

import argparse
import random
from pathlib import Path

LABELS = ("EMPTY", "NON_EMPTY")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DINOv2 empty-cell linear head.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor, AutoModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()

    samples = _collect_samples(args.dataset)
    train_samples, validation_samples = _stratified_split(samples, args.validation_fraction)
    train_features, train_labels = _extract_features(
        train_samples, processor, backbone, device, args.batch_size, Image, torch
    )
    validation_features, validation_labels = _extract_features(
        validation_samples, processor, backbone, device, args.batch_size, Image, torch
    )

    head = torch.nn.Linear(train_features.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    counts = torch.bincount(train_labels, minlength=len(LABELS)).float()
    class_weights = counts.sum() / counts.clamp_min(1)
    class_weights = (class_weights / class_weights.mean()).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_features), train_labels)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 25 == 0:
            accuracy = _accuracy(
                head,
                validation_features.to(device),
                validation_labels.to(device),
                torch,
            )
            print(
                f"epoch={epoch + 1} loss={loss.item():.4f} "
                f"validation_accuracy={accuracy:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "weight": head.weight.detach().cpu().contiguous(),
            "bias": head.bias.detach().cpu().contiguous(),
        },
        args.output,
        metadata={"labels": ",".join(LABELS), "backbone": args.model.name},
    )
    final_accuracy = _accuracy(
        head,
        validation_features.to(device),
        validation_labels.to(device),
        torch,
    )
    print(f"saved={args.output} validation_accuracy={final_accuracy:.4f}")


def _collect_samples(dataset: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label_index, label in enumerate(LABELS):
        directory = dataset / label
        paths = sorted(
            path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(paths) < 2:
            raise ValueError(f"{directory} requires at least two images")
        samples.extend((path, label_index) for path in paths)
    return samples


def _stratified_split(
    samples: list[tuple[Path, int]], validation_fraction: float
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    train: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    for label_index in range(len(LABELS)):
        group = [sample for sample in samples if sample[1] == label_index]
        random.shuffle(group)
        validation_count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    random.shuffle(train)
    random.shuffle(validation)
    return train, validation


def _extract_features(samples, processor, backbone, device, batch_size, image_type, torch):
    features = []
    labels = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = []
            for path, _ in batch:
                with image_type.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            output = backbone(**inputs).last_hidden_state[:, 0]
            features.append(output.cpu())
            labels.extend(label for _, label in batch)
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def _accuracy(head, features, labels, torch) -> float:
    head.eval()
    with torch.inference_mode():
        predictions = head(features).argmax(dim=1)
    return float((predictions == labels).float().mean().item())


if __name__ == "__main__":
    main()
