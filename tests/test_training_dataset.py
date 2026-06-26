import json
import time
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tote_vision.adapters.training_dataset import (
    TrainingDataError,
    TrainingDatasetStore,
)
from tote_vision.core.models import CellClassification, ToteLayout
from tote_vision.core.training_models import (
    DatasetSplit,
    RegionClass,
    TrainingRegion,
)


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (100, 80), "gray").save(output, format="PNG")
    return output.getvalue()


def _regions() -> tuple[TrainingRegion, ...]:
    return (
        TrainingRegion(
            region_id="tote-1",
            region_class=RegionClass.TOTE,
            polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
        ),
        TrainingRegion(
            region_id="cell-1",
            region_class=RegionClass.CELL,
            polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
            cell_id="A",
            cell_state=CellClassification.EMPTY,
        ),
    )


def _wait_for_export(store: TrainingDatasetStore) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        in_progress, requested, error = store.export_state()
        if error:
            raise AssertionError(error)
        if not in_progress and not requested:
            return
        time.sleep(0.01)
    raise AssertionError("training export did not complete")


def test_ready_annotation_exports_coco_and_cell_crop(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())

    updated = store.update_image(
        image.image_id,
        split=DatasetSplit.TRAIN,
        layout=ToteLayout.OPEN,
        regions=_regions(),
        ready=True,
    )
    _wait_for_export(store)

    assert updated.ready
    annotation_path = tmp_path / "exports/rfdetr/train/_annotations.coco.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert [category["name"] for category in coco["categories"]] == ["tote", "cell"]
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert len(list((tmp_path / "exports/cells/EMPTY").glob("*.png"))) == 1


def test_incomplete_ready_annotation_is_rejected(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())

    with pytest.raises(TrainingDataError, match="exactly one tote"):
        store.update_image(
            image.image_id,
            split=DatasetSplit.TRAIN,
            layout=ToteLayout.OPEN,
            regions=(),
            ready=True,
        )


def test_incomplete_draft_annotation_is_saved(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())

    updated = store.update_image(
        image.image_id,
        split=DatasetSplit.VALID,
        layout=ToteLayout.UNKNOWN,
        regions=(),
        ready=False,
    )

    assert not updated.ready
    assert updated.split is DatasetSplit.VALID


def test_uploaded_images_default_to_rough_train_valid_split(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)

    splits = [store.create_image(f"source-{index}.png", _png()).split for index in range(10)]

    assert splits == [
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.VALID,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.VALID,
    ]


def test_manual_split_override_is_respected_by_future_defaults(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    first = store.create_image("source-1.png", _png())
    store.update_image(
        first.image_id,
        split=DatasetSplit.VALID,
        layout=ToteLayout.UNKNOWN,
        regions=(),
        ready=False,
    )

    splits = [store.create_image(f"source-{index}.png", _png()).split for index in range(2, 6)]

    assert splits == [
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
    ]


def test_runtime_fail_states_are_rejected_as_training_labels(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())
    uncertain_cell = TrainingRegion(
        region_id="cell-1",
        region_class=RegionClass.CELL,
        polygon=((5, 5), (95, 5), (95, 75), (5, 75)),
        cell_id="A",
        cell_state=CellClassification.UNCERTAIN,
    )

    with pytest.raises(TrainingDataError, match="EMPTY or NON_EMPTY"):
        store.update_image(
            image.image_id,
            split=DatasetSplit.TRAIN,
            layout=ToteLayout.OPEN,
            regions=(uncertain_cell,),
            ready=False,
        )


def test_delete_removes_source_annotation_and_exports(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())
    store.update_image(
        image.image_id,
        split=DatasetSplit.TRAIN,
        layout=ToteLayout.OPEN,
        regions=_regions(),
        ready=True,
    )

    store.delete_image(image.image_id)
    _wait_for_export(store)

    assert store.list_images() == ()
    assert not (tmp_path / "images" / image.storage_filename).exists()
    coco = json.loads(
        (tmp_path / "exports/rfdetr/train/_annotations.coco.json").read_text(
            encoding="utf-8"
        )
    )
    assert coco["images"] == []
    assert not list((tmp_path / "exports/cells/EMPTY").glob("*.png"))


def test_delete_all_returns_deleted_count(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    store.create_image("one.png", _png())
    store.create_image("two.png", _png())

    deleted = store.delete_all_images()

    assert deleted == 2
    assert store.list_images() == ()


def test_open_tote_requires_shared_tote_and_cell_polygon(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())
    regions = (
        TrainingRegion(
            region_id="tote-1",
            region_class=RegionClass.TOTE,
            polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
        ),
        TrainingRegion(
            region_id="cell-1",
            region_class=RegionClass.CELL,
            polygon=((5, 5), (95, 5), (95, 75), (5, 75)),
            cell_id="A",
            cell_state=CellClassification.EMPTY,
        ),
    )

    with pytest.raises(TrainingDataError, match="same polygon"):
        store.update_image(
            image.image_id,
            split=DatasetSplit.TRAIN,
            layout=ToteLayout.OPEN,
            regions=regions,
            ready=True,
        )


def test_two_cell_layout_requires_divider_but_does_not_export_it(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())
    regions = (
        TrainingRegion(
            region_id="tote-1",
            region_class=RegionClass.TOTE,
            polygon=((0, 0), (100, 0), (100, 80), (0, 80)),
        ),
        TrainingRegion(
            region_id="divider-1",
            region_class=RegionClass.DIVIDER,
            polygon=((50, 0), (50, 80)),
        ),
        TrainingRegion(
            region_id="cell-a",
            region_class=RegionClass.CELL,
            polygon=((0, 0), (50, 0), (50, 80), (0, 80)),
            cell_id="A",
            cell_state=CellClassification.EMPTY,
        ),
        TrainingRegion(
            region_id="cell-b",
            region_class=RegionClass.CELL,
            polygon=((50, 0), (100, 0), (100, 80), (50, 80)),
            cell_id="B",
            cell_state=CellClassification.NON_EMPTY,
        ),
    )

    store.update_image(
        image.image_id,
        split=DatasetSplit.TRAIN,
        layout=ToteLayout.TWO_CELL,
        regions=regions,
        ready=True,
    )
    _wait_for_export(store)

    coco = json.loads(
        (tmp_path / "exports/rfdetr/train/_annotations.coco.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(coco["annotations"]) == 3


def test_two_cell_layout_without_divider_is_rejected(tmp_path: Path) -> None:
    store = TrainingDatasetStore(tmp_path)
    image = store.create_image("source.png", _png())
    regions = tuple(region for region in _regions() if region.region_class is not RegionClass.CELL)
    cells = (
        TrainingRegion(
            region_id="cell-a",
            region_class=RegionClass.CELL,
            polygon=((0, 0), (50, 0), (50, 80), (0, 80)),
            cell_id="A",
            cell_state=CellClassification.EMPTY,
        ),
        TrainingRegion(
            region_id="cell-b",
            region_class=RegionClass.CELL,
            polygon=((50, 0), (100, 0), (100, 80), (50, 80)),
            cell_id="B",
            cell_state=CellClassification.EMPTY,
        ),
    )

    with pytest.raises(TrainingDataError, match="1 divider"):
        store.update_image(
            image.image_id,
            split=DatasetSplit.TRAIN,
            layout=ToteLayout.TWO_CELL,
            regions=regions + cells,
            ready=True,
        )
