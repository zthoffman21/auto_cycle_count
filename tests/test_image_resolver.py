from pathlib import Path

import pytest

from tote_vision.adapters.image_resolver import (
    LocalImageResolver,
    UnsupportedImageUriError,
)


def test_resolves_dashboard_artifact_inside_root(tmp_path: Path) -> None:
    image = tmp_path / "tote.png"
    image.write_bytes(b"image")

    resolved = LocalImageResolver(tmp_path).resolve("/artifacts/tote.png")

    assert resolved == image.resolve()


def test_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"image")

    with pytest.raises(UnsupportedImageUriError):
        LocalImageResolver(tmp_path).resolve("/artifacts/../outside.png")

