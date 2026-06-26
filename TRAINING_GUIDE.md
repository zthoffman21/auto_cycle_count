# Initial Training and Live-Test Guide

This guide produces the two project-specific model artifacts required by the real vision runtime:

```text
models/rf-detr-seg-small-totes.pth
models/dinov2-vits14/
models/empty-cell-head.safetensors
```

`EMPTY` and `NON_EMPTY` are the only trained cell classes.

- `UNCERTAIN` is returned when neither binary class reaches its configured confidence threshold.
- `BAD_CAPTURE` is returned when inference cannot extract a usable image or cell crop.

Runtime classification defaults to `0.70` for both EMPTY and NON_EMPTY. Predictions below both
thresholds remain `UNCERTAIN`.

## 1. Start the annotation dashboard

Run these commands from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -BuildDependencyImage
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

The first build creates `cycle-count-vision-dependencies:latest`, which contains the slow CUDA,
PyTorch, and RF-DETR dependency layer. Later app-only rebuilds automatically reuse that image:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

Open the training page:

```powershell
Start-Process http://localhost:8000/train
```

## 2. Upload and annotate images

For every uploaded image:

1. Review the automatically assigned dataset split. New uploads default to roughly 80% `train`
   and 20% `valid`; change the split only when you need to rebalance the dataset manually.
2. Select the tote layout.
3. Draw exactly one tote polygon or box. For an open tote, this automatically creates cell `A`
   with the same geometry.
4. For a two-cell layout, draw one line through the center of the divider.
5. For a four-cell layout, draw both divider lines. Each line must cross the full tote area.
6. The dashboard generates and identifies the cell polygons automatically.
7. Label every generated cell `EMPTY` or `NON_EMPTY`.
8. Save once the form reports that the annotation is ready to export.

Ready annotations automatically generate:

```text
data/training/exports/rfdetr/
data/training/exports/cells/
```

## 3. Stop the container

```powershell
docker stop cycle-count-vision
```

## 4. Verify the GPU training image

Verify that the GPU is visible:

```powershell
docker run --rm --gpus all cycle-count-vision:latest python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The expected result includes `True` and the NVIDIA GPU name.

## 5. Download the DINOv2 backbone

Create the model directory:

```powershell
New-Item -ItemType Directory -Force models
```

Download DINOv2-small into the project:

```powershell
docker run --rm --mount "type=bind,source=$((Get-Location).Path),target=/workspace" --workdir /workspace --env HF_HOME=/tmp/huggingface --env XDG_CACHE_HOME=/tmp cycle-count-vision:latest hf download facebook/dinov2-small --local-dir models/dinov2-vits14
```

## 6. Train RF-DETR tote and cell segmentation

```powershell
docker run --name rfdetr-train --rm --gpus all --shm-size=8g --mount "type=bind,source=$((Get-Location).Path),target=/workspace" --workdir /workspace cycle-count-vision:latest python scripts/train_rfdetr.py --dataset data/training/exports/rfdetr --output output/rfdetr --model-output models/rf-detr-seg-small-totes.pth --log-every-n-steps 1 --num-workers 3 --progress-bar tqdm --sanity-val-steps 0
```

This trains class ID `0` as `tote` and class ID `1` as `cell`. During inference, detected cell
masks must be at least 90% contained by the tote mask, are clipped to that mask, and are
deduplicated before the layout is inferred from the resulting count. Remaining cells must also
have no substantial pairwise overlap. Inference reports the observed layout without accepting an
expected layout; WES can compare the observation with its own tote information.

Inference uses separate confidence thresholds because open-tote images contain overlapping tote
and cell masks. The raw RF-DETR threshold is `0.05`, the tote threshold is `0.10`, and the cell
threshold remains `0.50`. These can be overridden with `CYCLE_COUNT_RFDETR_CONFIDENCE_THRESHOLD`,
`CYCLE_COUNT_RFDETR_TOTE_CONFIDENCE_THRESHOLD`, and
`CYCLE_COUNT_RFDETR_CELL_CONFIDENCE_THRESHOLD`.

## 7. Train the binary DINOv2 cell classifier

```powershell
docker run --rm --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/workspace" --workdir /workspace cycle-count-vision:latest python scripts/train_empty_cell_head.py --dataset data/training/exports/cells --model models/dinov2-vits14 --output models/empty-cell-head.safetensors
```

## 8. Train the patch anomaly detector (optional — enables comparison mode)

```powershell
docker run --rm --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/workspace" --workdir /workspace cycle-count-vision:latest python scripts/train_patch_anomaly_detector.py --dataset data/training/exports/cells --model models/dinov2-vits14 --output models/patch-anomaly.safetensors
```

This trains a one-class anomaly detector that learns only what an empty cell looks like. Unlike
the binary classifier, it detects novel objects that were never in the training set by measuring
how far each image patch deviates from the empty-cell distribution. When
`CYCLE_COUNT_PATCH_ANOMALY_MODEL_PATH` is set, the dashboard shows a method toggle so you can
compare both classifiers on the same image.

## 9. Download the GroundingDINO model (optional — enables comparison mode)

```powershell
docker run --rm --mount "type=bind,source=$((Get-Location).Path),target=/workspace" --workdir /workspace --env HF_HOME=/tmp/huggingface --env XDG_CACHE_HOME=/tmp cycle-count-vision:latest hf download IDEA-Research/grounding-dino-base --local-dir models/grounding-dino-base
```

This downloads the zero-shot object detection model (~700 MB). No training is required — GroundingDINO is used as-is with the text prompt `"object."`. When `CYCLE_COUNT_GROUNDING_DINO_MODEL_PATH` is set, the dashboard shows a method toggle that includes `grounding_dino` alongside the other classifiers.

Tune `CYCLE_COUNT_GROUNDING_DINO_BOX_THRESHOLD` (default `0.25`) if you get too many false positives (raise it) or miss real items (lower it).

## 10. Verify model artifacts

```powershell
Get-ChildItem -Recurse models
```

Confirm that all three required artifacts exist:

```text
models/rf-detr-seg-small-totes.pth
models/dinov2-vits14/config.json
models/dinov2-vits14/model.safetensors
models/dinov2-vits14/preprocessor_config.json
models/empty-cell-head.safetensors
```

## 11. Start

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

Open the inspection dashboard:

```powershell
Start-Process http://localhost:8000/
```
