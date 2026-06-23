from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


class UnsupportedImageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    path: Path
    public_uri: str


class LocalArtifactStore:
    _IMAGE_SIGNATURES: ClassVar[dict[str, tuple[bytes, str]]] = {
        "image/jpeg": (b"\xff\xd8\xff", ".jpg"),
        "image/png": (b"\x89PNG\r\n\x1a\n", ".png"),
        "image/webp": (b"RIFF", ".webp"),
    }

    def __init__(self, root: Path, public_prefix: str = "/artifacts") -> None:
        self._root = root.resolve()
        self._public_prefix = public_prefix.rstrip("/")
        self._root.mkdir(parents=True, exist_ok=True)

    def save_image(self, artifact_id: str, content_type: str, content: bytes) -> StoredArtifact:
        signature = self._IMAGE_SIGNATURES.get(content_type)
        if signature is None or not _matches_signature(content_type, content, signature[0]):
            raise UnsupportedImageError("upload must be a valid JPEG, PNG, or WebP image")

        suffix = signature[1]
        path = self._root / f"{artifact_id}{suffix}"
        path.write_bytes(content)
        return StoredArtifact(path=path, public_uri=f"{self._public_prefix}/{path.name}")


def _matches_signature(content_type: str, content: bytes, signature: bytes) -> bool:
    if not content.startswith(signature):
        return False
    if content_type == "image/webp":
        return len(content) >= 12 and content[8:12] == b"WEBP"
    return True
