from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tote_vision.adapters.training_dataset import TrainingDatasetStore
from tote_vision.api.training_routes import router as training_router
from tote_vision.application.predict_training_geometry import (
    TrainingGeometryPrediction,
    TrainingGeometryPredictionError,
    _four_point_polygon,
    _prediction_regions,
)
from tote_vision.core.models import CellClassification, CellGeometry, ToteLayout
from tote_vision.core.training_models import (
    DatasetSplit,
    RegionClass,
    TrainingImage,
    TrainingRegion,
)


class FakeTrainingGeometryPredictor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def predict(self, image: TrainingImage, image_path: Path) -> TrainingGeometryPrediction:
        self.calls.append(image.image_id)
        if self.fail:
            raise TrainingGeometryPredictionError("no geometry")
        regions = (
            TrainingRegion(
                region_id="tote-1",
                region_class=RegionClass.TOTE,
                polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
            ),
            TrainingRegion(
                region_id="cell-a",
                region_class=RegionClass.CELL,
                polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
                cell_id="A",
                cell_state=None,
            ),
        )
        return TrainingGeometryPrediction(ToteLayout.OPEN, 0.91, regions)


def _png(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), "gray").save(path, format="PNG")
    return path.read_bytes()


def _client(
    tmp_path: Path,
    predictor: FakeTrainingGeometryPredictor,
) -> tuple[TestClient, TrainingDatasetStore]:
    app = FastAPI()
    app.state.training_store = TrainingDatasetStore(tmp_path / "training")
    app.state.training_geometry_predictor = predictor
    app.include_router(training_router)
    return TestClient(app), app.state.training_store


def test_tote_prediction_is_normalized_to_four_ordered_points() -> None:
    polygon = ((10, 10), (90, 8), (95, 70), (55, 78), (12, 65))

    normalized = _four_point_polygon(polygon)

    assert len(normalized) == 4
    assert normalized[0][0] <= normalized[1][0]
    assert normalized[0][1] <= normalized[3][1]


def test_two_cell_prediction_builds_divider_and_unclassified_cells() -> None:
    tote = ((0, 0), (100, 0), (100, 80), (0, 80))
    cells = (
        CellGeometry(cell_id="A", polygon=((0, 0), (48, 0), (48, 80), (0, 80))),
        CellGeometry(cell_id="B", polygon=((52, 0), (100, 0), (100, 80), (52, 80))),
    )

    regions = _prediction_regions(tote, ToteLayout.TWO_CELL, cells)

    assert [region.region_class for region in regions] == [
        RegionClass.TOTE,
        RegionClass.DIVIDER,
        RegionClass.CELL,
        RegionClass.CELL,
    ]
    assert [region.cell_id for region in regions if region.region_class is RegionClass.CELL] == [
        "A",
        "B",
    ]
    assert all(region.cell_state is None for region in regions)


def test_four_cell_prediction_builds_two_dividers_and_four_unclassified_cells() -> None:
    tote = ((0, 0), (100, 0), (100, 80), (0, 80))
    cells = (
        CellGeometry(cell_id="A", polygon=((0, 0), (48, 0), (48, 38), (0, 38))),
        CellGeometry(cell_id="B", polygon=((52, 0), (100, 0), (100, 38), (52, 38))),
        CellGeometry(cell_id="C", polygon=((0, 42), (48, 42), (48, 80), (0, 80))),
        CellGeometry(cell_id="D", polygon=((52, 42), (100, 42), (100, 80), (52, 80))),
    )

    regions = _prediction_regions(tote, ToteLayout.FOUR_CELL, cells)

    assert sum(region.region_class is RegionClass.DIVIDER for region in regions) == 2
    assert [region.cell_id for region in regions if region.region_class is RegionClass.CELL] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert all(region.cell_state is None for region in regions)


def test_bulk_prediction_updates_only_drafts_and_leaves_labels_unclassified(tmp_path: Path) -> None:
    predictor = FakeTrainingGeometryPredictor()
    client, store = _client(tmp_path, predictor)
    image_path = tmp_path / "source.png"
    draft = store.create_image("draft.png", _png(image_path))
    ready = store.create_image("ready.png", _png(image_path))
    store.update_image(
        ready.image_id,
        split=DatasetSplit.TRAIN,
        layout=ToteLayout.OPEN,
        regions=(
            TrainingRegion(
                region_id="tote-ready",
                region_class=RegionClass.TOTE,
                polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
            ),
            TrainingRegion(
                region_id="cell-ready",
                region_class=RegionClass.CELL,
                polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
                cell_id="A",
                cell_state=CellClassification.EMPTY,
            ),
        ),
        ready=True,
    )

    response = client.post(
        "/training/images/predict-drafts",
        json={"imageIds": [draft.image_id, ready.image_id], "overwrite": False},
    )

    assert response.status_code == 200
    assert response.json()["predicted"] == 1
    assert response.json()["skipped"] == 1
    updated = store.get_image(draft.image_id)
    assert not updated.ready
    assert updated.layout is ToteLayout.OPEN
    assert [
        cell.cell_state
        for cell in updated.regions
        if cell.region_class is RegionClass.CELL
    ] == [None]
    assert predictor.calls == [draft.image_id]


def test_failed_prediction_does_not_mutate_draft(tmp_path: Path) -> None:
    predictor = FakeTrainingGeometryPredictor(fail=True)
    client, store = _client(tmp_path, predictor)
    image = store.create_image("draft.png", _png(tmp_path / "source.png"))

    response = client.post(
        "/training/images/predict-drafts",
        json={"imageIds": [image.image_id], "overwrite": False},
    )

    assert response.status_code == 200
    assert response.json()["failed"] == 1
    assert store.get_image(image.image_id).regions == ()


def test_prediction_status_and_endpoint_are_unavailable_without_predictor(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    app.state.training_store = TrainingDatasetStore(tmp_path / "training")
    app.state.training_geometry_predictor = None
    app.state.max_upload_bytes = 1024
    app.include_router(training_router)
    client = TestClient(app)

    status_response = client.get("/training/status")
    prediction_response = client.post("/training/images/predict-drafts", json={})

    assert status_response.status_code == 200
    assert status_response.json()["predictionAvailable"] is False
    assert prediction_response.status_code == 503
