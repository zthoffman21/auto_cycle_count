"""Train a patch-level one-class anomaly detector for empty cell classification.

Unlike the linear probe (which is a binary classifier), this learns only what
an empty cell looks like. At inference, any patch embedding that is far from
the empty centroid raises the anomaly score — catching novel objects without
needing them in the training set.

Key design: per-position centroids.  DINOv2 produces N patch tokens, each
corresponding to a fixed spatial location.  Instead of one global centroid
(which causes all distances to converge in high dimensions), we compute one
centroid per patch position.  A foreign object is compared to what an empty
tote normally looks like *at that exact location*, giving a much tighter
reference and much better separation.

NON_EMPTY samples are never used for training — only EMPTY cells are needed.
NON_EMPTY images are loaded at the end solely to report how well the trained
detector separates the two classes before you deploy.

Dataset layout expected:
    <dataset>/
        EMPTY/       *.png  (polygon-masked cell crops)
        NON_EMPTY/   *.png  (polygon-masked cell crops — validation only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DINOv2 patch anomaly detector.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=95.0,
        help="percentile of per-image max anomaly score used as the scale (default 95)",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()

    empty_paths = _collect_paths(args.dataset / "EMPTY")
    nonempty_paths = _collect_paths(args.dataset / "NON_EMPTY")

    print(f"EMPTY samples:     {len(empty_paths)}  (used for training)")
    print(f"NON_EMPTY samples: {len(nonempty_paths)}  (validation only — not used for training)")

    print("\nExtracting patch embeddings from EMPTY samples...")
    # Shape: (N_images, N_patches, embed_dim) — spatial structure preserved
    empty_stack = _extract_patches(
        empty_paths, processor, backbone, device, args.batch_size, Image, torch
    )

    # One centroid per patch position: (N_patches, embed_dim)
    centroids = empty_stack.mean(dim=0)
    n_patches, embed_dim = centroids.shape
    print(f"Per-position centroids: {n_patches} positions x {embed_dim}-dim")

    # Per-image max anomaly distance for scale estimation
    empty_max_dists = _max_anomaly_distances(empty_stack, centroids, torch)
    scale = float(torch.quantile(empty_max_dists, args.scale_percentile / 100.0).item())
    pct = args.scale_percentile
    print(
        f"\nAnomaly scale (p{pct:.0f} of empty max-position-distances): {scale:.4f}"
        f"\n  min={empty_max_dists.min():.4f}"
        f"  median={empty_max_dists.median():.4f}"
        f"  max={empty_max_dists.max():.4f}"
    )

    print("\n--- Separation report (score = max_position_distance / scale) ---")
    _report("EMPTY (training data)", empty_max_dists, scale)

    if nonempty_paths:
        print("\nExtracting patch embeddings from NON_EMPTY samples (validation only)...")
        nonempty_stack = _extract_patches(
            nonempty_paths, processor, backbone, device, args.batch_size, Image, torch
        )
        nonempty_max_dists = _max_anomaly_distances(nonempty_stack, centroids, torch)
        _report("NON_EMPTY (validation only)", nonempty_max_dists, scale)

    # Normalized median empty score — used to calibrate the probability sigmoid
    # so the curve is anchored to where the actual empty distribution lives.
    empty_p50_norm = float(empty_max_dists.median().item()) / scale
    print(f"empty_p50_norm (median empty score / scale): {empty_p50_norm:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "centroids": centroids.cpu().contiguous(),  # (N_patches, embed_dim)
            "scale": torch.tensor(scale, dtype=torch.float32),
            "empty_p50_norm": torch.tensor(empty_p50_norm, dtype=torch.float32),
        },
        args.output,
        metadata={
            "backbone": args.model.name,
            "scale_percentile": str(args.scale_percentile),
            "empty_samples": str(len(empty_paths)),
            "n_patches": str(n_patches),
        },
    )
    print(f"\nSaved: {args.output}")


def _collect_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _extract_patches(paths, processor, backbone, device, batch_size, image_type, torch):
    """Return patch embeddings with spatial structure — shape (N_images, N_patches, embed_dim)."""
    all_patches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with image_type.open(path) as src:
                    images.append(src.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = backbone(**inputs)
            # Skip CLS token (index 0); keep per-position patch tokens
            patches = output.last_hidden_state[:, 1:, :]  # (B, N_patches, embed_dim)
            all_patches.append(patches.cpu())
            if (start // batch_size + 1) % 5 == 0 or start + batch_size >= len(paths):
                print(f"  {min(start + batch_size, len(paths))} / {len(paths)}")
    return torch.cat(all_patches, dim=0)  # (N_images, N_patches, embed_dim)


def _max_anomaly_distances(stack, centroids, torch):
    """Per-image max distance from each patch to its position-specific centroid.

    Returns shape (N_images,).  Each value is the worst-case patch anomaly for
    that image — the signal a foreign object at any single location would produce.
    """
    # stack:    (N_images, N_patches, embed_dim)
    # centroids: (N_patches, embed_dim)
    dists = torch.norm(stack - centroids.unsqueeze(0), dim=-1)  # (N_images, N_patches)
    return dists.max(dim=-1).values  # (N_images,)


def _report(label: str, max_dists, scale: float) -> None:
    scores = max_dists / scale
    flagged = int((scores >= 1.0).sum().item())
    pct = 100 * flagged / len(scores)
    print(
        f"  {label}: {len(scores)} samples"
        f"  score  min={scores.min():.3f}  median={scores.median():.3f}"
        f"  max={scores.max():.3f}"
        f"  flagged (score≥1): {flagged}/{len(scores)} ({pct:.1f}%)"
    )


if __name__ == "__main__":
    main()
