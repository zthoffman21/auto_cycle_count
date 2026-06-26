from __future__ import annotations

from tote_vision.adapters.dinov2 import Dinov2CellClassifier
from tote_vision.adapters.grounding_dino import GroundingDinoCellClassifier
from tote_vision.adapters.image_resolver import LocalImageResolver
from tote_vision.adapters.patch_anomaly import PatchAnomalyCellClassifier
from tote_vision.adapters.rfdetr import (
    RfdetrInferenceSession,
    RfdetrLayoutDetector,
    RfdetrToteDetector,
)
from tote_vision.application.inspect_empty_tote import InspectEmptyTote
from tote_vision.config import Settings
from tote_vision.core.decision import DecisionEngine, DecisionPolicy
from tote_vision.core.geometry import GeometryValidator


def build_inspector(settings: Settings) -> InspectEmptyTote:
    assert settings.rfdetr_checkpoint_path is not None
    assert settings.dinov2_model_path is not None
    assert settings.dinov2_classifier_path is not None

    image_resolver = LocalImageResolver(settings.artifact_directory)
    session = RfdetrInferenceSession(
        checkpoint_path=settings.rfdetr_checkpoint_path,
        image_resolver=image_resolver,
        confidence_threshold=settings.rfdetr_confidence_threshold,
    )
    tote_detector = RfdetrToteDetector(
        session,
        settings.rfdetr_tote_class_id,
        confidence_threshold=settings.rfdetr_tote_confidence_threshold,
    )
    layout_detector = RfdetrLayoutDetector(
        session,
        settings.rfdetr_cell_class_id,
        tote_class_id=settings.rfdetr_tote_class_id,
        confidence_threshold=settings.rfdetr_cell_confidence_threshold,
        open_tote_fallback_confidence_threshold=(
            settings.rfdetr_open_tote_fallback_confidence_threshold
        ),
        min_containment=settings.rfdetr_cell_min_containment,
        duplicate_iou_threshold=settings.rfdetr_cell_duplicate_iou_threshold,
        max_cell_iou=settings.rfdetr_cell_max_iou,
    )
    cell_classifiers = {
        "linear_probe": Dinov2CellClassifier(
            model_path=settings.dinov2_model_path,
            classifier_path=settings.dinov2_classifier_path,
            image_resolver=image_resolver,
            device=settings.vision_device,
            empty_threshold=settings.empty_threshold,
            non_empty_threshold=settings.non_empty_threshold,
            classifier_name="linear_probe",
        ),
    }
    if settings.patch_anomaly_model_path is not None:
        cell_classifiers["patch_anomaly"] = PatchAnomalyCellClassifier(
            model_path=settings.dinov2_model_path,
            anomaly_model_path=settings.patch_anomaly_model_path,
            image_resolver=image_resolver,
            device=settings.vision_device,
            empty_threshold=settings.empty_threshold,
            non_empty_threshold=settings.non_empty_threshold,
        )
    if settings.grounding_dino_model_path is not None:
        cell_classifiers["grounding_dino"] = GroundingDinoCellClassifier(
            model_path=settings.grounding_dino_model_path,
            image_resolver=image_resolver,
            device=settings.vision_device,
            empty_threshold=settings.empty_threshold,
            non_empty_threshold=settings.non_empty_threshold,
            box_threshold=settings.grounding_dino_box_threshold,
        )

    return InspectEmptyTote(
        tote_detector=tote_detector,
        layout_detector=layout_detector,
        cell_classifiers=cell_classifiers,
        geometry_validator=GeometryValidator(),
        decision_engine=DecisionEngine(DecisionPolicy()),
        decision_version=settings.decision_version,
    )
