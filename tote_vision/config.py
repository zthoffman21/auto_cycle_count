from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CYCLE_COUNT_",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    empty_threshold: float = Field(default=0.70, ge=0, le=1)
    non_empty_threshold: float = Field(default=0.70, ge=0, le=1)
    decision_version: str = "empty-tote-v1"
    artifact_directory: Path = Path("data/artifacts")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    dashboard_enabled: bool = True
    dashboard_dist_path: Path = Path("dashboard/dist")
    rfdetr_checkpoint_path: Path | None = None
    rfdetr_confidence_threshold: float = Field(default=0.05, ge=0, le=1)
    rfdetr_tote_confidence_threshold: float = Field(default=0.10, ge=0, le=1)
    rfdetr_cell_confidence_threshold: float = Field(default=0.50, ge=0, le=1)
    rfdetr_tote_class_id: int = Field(default=0, ge=0)
    rfdetr_cell_class_id: int = Field(default=1, ge=0)
    rfdetr_cell_min_containment: float = Field(default=0.9, ge=0, le=1)
    rfdetr_cell_duplicate_iou_threshold: float = Field(default=0.7, ge=0, le=1)
    rfdetr_cell_max_iou: float = Field(default=0.15, ge=0, le=1)
    dinov2_model_path: Path | None = None
    dinov2_classifier_path: Path | None = None
    patch_anomaly_model_path: Path | None = None
    grounding_dino_model_path: Path | None = None
    grounding_dino_box_threshold: float = Field(default=0.25, ge=0, le=1)
    vision_device: str = "cuda"
    training_directory: Path = Path("data/training")

    def model_post_init(self, __context: object) -> None:
        if self.environment.lower() == "production" and self.dashboard_enabled:
            raise ValueError("dashboard must be disabled in production until authentication exists")
        required_paths = {
            "rfdetr_checkpoint_path": self.rfdetr_checkpoint_path,
            "dinov2_model_path": self.dinov2_model_path,
            "dinov2_classifier_path": self.dinov2_classifier_path,
        }
        missing = [name for name, path in required_paths.items() if path is None]
        if missing:
            raise ValueError(f"vision inference requires: {', '.join(missing)}")
