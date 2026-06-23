# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Phase 1 empty-tote inspection service for warehouse cycle counting. Given an image, the system detects whether totes are empty using computer vision, returning a PASS/REVIEW/FAIL/ERROR decision.

## Commands

```powershell
# Install (dev only — no GPU required)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Install with vision dependencies (GPU + CUDA required)
pip install -e ".[dev,vision]"

# Build frontend
npm --prefix dashboard install && npm --prefix dashboard run build

# Run API server (dev adapters — no model checkpoints needed)
uvicorn tote_vision.main:app --reload

# Lint
ruff check .

# Tests
pytest
pytest tests/test_api.py::test_inspection_endpoint_uses_wire_contract   # single test

# Docker (GPU + model checkpoints required)
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

API docs at `http://localhost:8000/docs`, dashboard at `http://localhost:8000/`.

## Architecture

Clean architecture with four layers inside `tote_vision/`:

| Layer | Path | Responsibility |
|---|---|---|
| API | `api/` | FastAPI routes, Pydantic schemas (camelCase wire contracts) |
| Application | `application/` | `InspectEmptyTote` use case — orchestrates the inspection pipeline |
| Core | `core/` | Business logic (decision engine, geometry validator), Protocol interfaces (`ports.py`), enums |
| Adapters | `adapters/` | Model implementations (RF-DETR, DINOv2) and infrastructure (storage, image resolution) |

### Inspection Pipeline

`POST /inspect` → `InspectEmptyTote.execute()`:
1. **ToteDetector** — detects tote presence and polygon bounding box
2. **LayoutDetector** — determines layout (OPEN / TWO_CELL / FOUR_CELL) and crops each cell
3. **GeometryValidator** — validates cell counts and bounds; returns INVALID_GEOMETRY on failure
4. **CellClassifier** — classifies each cell as EMPTY / NON_EMPTY / UNCERTAIN / BAD_CAPTURE (runs async/parallel)
5. **DecisionEngine** — applies policy thresholds to produce final decision + reason code

`bootstrap.py` is the only place that knows which concrete adapter is used; the use case and core are adapter-agnostic. RF-DETR handles tote and layout detection; DINOv2 handles cell classification. Both require GPU and trained model checkpoints (paths set via env vars).

### Key Enumerations

```python
ToteLayout:           OPEN, TWO_CELL, FOUR_CELL, UNKNOWN
CellClassification:   EMPTY, NON_EMPTY, UNCERTAIN, BAD_CAPTURE
InspectionDecision:   PASS, REVIEW, FAIL, ERROR
ReasonCode:           ALL_CELLS_EMPTY, NON_EMPTY_CELL_DETECTED, UNCERTAIN_CELL,
                      BAD_CAPTURE, TOTE_NOT_DETECTED, INVALID_GEOMETRY, INFERENCE_ERROR
CoordinateSpace:      IMAGE_PIXELS, NORMALIZED_100
```

### Frontend

React/TypeScript SPA in `dashboard/` built with Vite. Provides an inspection dashboard and a training annotation UI (`TrainingPage.tsx`). Built output is served as static files by FastAPI when `CYCLE_COUNT_DASHBOARD_ENABLED=true`.

### Training Scripts

Standalone training scripts in `scripts/`:
- `train_rfdetr.py` — RF-DETR tote and cell segmentation
- `train_empty_cell_head.py` — binary DINOv2 cell classifier
- `train_patch_anomaly_detector.py` — one-class anomaly detector (optional)

Training annotations stored in `data/training/`, inspection artifacts in `data/artifacts/`.

## Configuration

All settings via environment variables, loaded by `config.py` using Pydantic `BaseSettings`. See `.env.example` for the full list. Key vars:

- `CYCLE_COUNT_EMPTY_THRESHOLD` / `CYCLE_COUNT_NON_EMPTY_THRESHOLD` — classifier confidence cutoffs
- `CYCLE_COUNT_ARTIFACT_DIRECTORY` / `CYCLE_COUNT_TRAINING_DIRECTORY` — data paths
- `CYCLE_COUNT_RFDETR_CHECKPOINT_PATH` / `CYCLE_COUNT_DINOV2_MODEL_PATH` / `CYCLE_COUNT_DINOV2_CLASSIFIER_PATH` — required model checkpoint paths
- `CYCLE_COUNT_DASHBOARD_DIST_PATH` — path to built frontend dist (default: `dashboard/dist`)

## Code Style

- **Ruff** for linting/formatting; line length 100, Python 3.12+, rules: E, F, I, B, UP, SIM, RUF.
- API schemas use **camelCase** for JSON wire contracts (Pydantic aliases); internal models use snake_case.
- Core layer (`core/`) must not import from `adapters/` or `api/` — dependencies point inward only.
- Ports (`core/ports.py`) define `Protocol` interfaces; adapters implement them structurally (no inheritance).
