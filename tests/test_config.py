import pytest
from pydantic import ValidationError

from tote_vision.config import Settings


def test_production_rejects_unauthenticated_dashboard() -> None:
    with pytest.raises(ValidationError):
        Settings(
            environment="production",
            dashboard_enabled=True,
            rfdetr_checkpoint_path="/models/rf-detr.pth",
            dinov2_model_path="/models/dinov2",
            dinov2_classifier_path="/models/head.safetensors",
        )
