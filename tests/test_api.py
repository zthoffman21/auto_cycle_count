import time
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from tote_vision.config import Settings
from tote_vision.main import create_app


def test_inspection_endpoint_reports_unavailable_without_models() -> None:
    client = TestClient(create_app(Settings()))

    response = client.post(
        "/vision/inspect-empty-tote",
        json={
            "imageUri": "file:///tmp/test.png",
        },
    )

    assert response.status_code == 503
    assert "vision inference is not available" in response.json()["detail"]


def test_dashboard_upload_reports_unavailable_without_models(tmp_path: Path) -> None:
    settings = Settings(artifact_directory=tmp_path)
    client = TestClient(create_app(settings))
    png = b"\x89PNG\r\n\x1a\n" + b"test-image-payload"

    response = client.post(
        "/dashboard/inspect",
        data={
            "tote_id": "TOTE-9",
            "station_id": "DEV-STATION",
            "camera_id": "UPLOAD",
        },
        files={"image": ("tote.png", png, "image/png")},
    )

    assert response.status_code == 503
    assert "vision inference is not available" in response.json()["detail"]
    assert client.get("/").status_code == 200


def test_dashboard_rejects_disguised_non_image(tmp_path: Path) -> None:
    app = create_app(Settings(artifact_directory=tmp_path))
    app.state.inspector = object()
    client = TestClient(app)

    response = client.post(
        "/dashboard/inspect",
        data={"tote_id": "TOTE-9"},
        files={"image": ("fake.png", b"not an image", "image/png")},
    )

    assert response.status_code == 415


def test_training_page_upload_and_annotation_flow(tmp_path: Path) -> None:
    output = BytesIO()
    Image.new("RGB", (100, 80), "gray").save(output, format="PNG")
    settings = Settings(
        artifact_directory=tmp_path / "artifacts",
        training_directory=tmp_path / "training",
    )
    client = TestClient(create_app(settings))

    upload_response = client.post(
        "/training/images",
        files={"images": ("tote.png", output.getvalue(), "image/png")},
    )

    assert upload_response.status_code == 201
    image = upload_response.json()[0]
    assert client.get("/train").status_code == 200

    update_response = client.put(
        f"/training/images/{image['imageId']}",
        json={
            "split": "train",
            "layout": "OPEN",
            "ready": True,
            "regions": [
                {
                    "regionId": "tote-1",
                    "regionClass": "tote",
                    "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]],
                },
                {
                    "regionId": "cell-1",
                    "regionClass": "cell",
                    "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]],
                    "cellId": "A",
                    "cellState": "EMPTY",
                },
            ],
        },
    )

    assert update_response.status_code == 200
    deadline = time.monotonic() + 2
    while True:
        status_payload = client.get("/training/status").json()
        if not status_payload["exportInProgress"] and not status_payload["exportPending"]:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("training export did not complete")
        time.sleep(0.01)
    assert status_payload["readyImages"] == 1
    assert status_payload["exportedCellCrops"] == 1

    delete_response = client.delete(f"/training/images/{image['imageId']}")
    assert delete_response.status_code == 204
    assert client.get("/training/status").json()["totalImages"] == 0


def test_training_bulk_delete(tmp_path: Path) -> None:
    output = BytesIO()
    Image.new("RGB", (20, 20), "gray").save(output, format="PNG")
    client = TestClient(create_app(Settings(training_directory=tmp_path / "training")))
    for filename in ("one.png", "two.png"):
        response = client.post(
            "/training/images",
            files={"images": (filename, output.getvalue(), "image/png")},
        )
        assert response.status_code == 201

    response = client.delete("/training/images")

    assert response.status_code == 200
    assert response.json() == {"deletedImages": 2}
    assert client.get("/training/images").json() == []
