from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


class UnsupportedImageUriError(ValueError):
    pass


class LocalImageResolver:
    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root.resolve()

    def resolve(self, image_uri: str) -> Path:
        if image_uri.startswith("/artifacts/"):
            candidate = self._artifact_root / image_uri.removeprefix("/artifacts/")
        elif image_uri.startswith("file://"):
            parsed = urlparse(image_uri)
            candidate = Path(unquote(parsed.path))
        else:
            candidate = Path(image_uri)

        resolved = candidate.resolve()
        if image_uri.startswith("/artifacts/") and not resolved.is_relative_to(self._artifact_root):
            raise UnsupportedImageUriError("artifact image URI escapes configured storage root")
        if not resolved.is_file():
            raise FileNotFoundError(f"inspection image does not exist: {resolved}")
        return resolved

