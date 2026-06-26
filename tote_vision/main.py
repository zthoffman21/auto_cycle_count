import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tote_vision.adapters.local_artifacts import LocalArtifactStore
from tote_vision.adapters.training_dataset import TrainingDatasetStore
from tote_vision.api.dashboard_routes import router as dashboard_router
from tote_vision.api.routes import router
from tote_vision.api.training_routes import router as training_router
from tote_vision.application.predict_training_geometry import TrainingGeometryPredictor
from tote_vision.bootstrap import build_inspector, build_training_geometry_predictor
from tote_vision.config import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s [%(inspection_id)s] %(message)s",
            defaults={"inspection_id": "-"},
        )
    )
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
    app = FastAPI(
        title="Cycle Count Vision Service",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
    )
    app.state.inspector = _try_build_inspector(settings)
    app.state.vision_startup_error = (
        None
        if app.state.inspector is not None
        else "vision models are not configured or could not be loaded"
    )
    app.include_router(router)
    if settings.dashboard_enabled:
        app.state.artifact_store = LocalArtifactStore(settings.artifact_directory)
        app.state.max_upload_bytes = settings.max_upload_bytes
        app.state.training_store = TrainingDatasetStore(settings.training_directory)
        app.state.training_geometry_predictor = _try_build_training_geometry_predictor(
            settings,
            app.state.inspector,
        )
        app.include_router(dashboard_router)
        app.include_router(training_router)
        app.mount(
            "/artifacts",
            StaticFiles(directory=settings.artifact_directory),
            name="artifacts",
        )

        dashboard_dist_directory = settings.dashboard_dist_path
        dashboard_assets_directory = dashboard_dist_directory / "assets"
        dashboard_index = dashboard_dist_directory / "index.html"
        app.mount(
            "/assets",
            StaticFiles(directory=dashboard_assets_directory),
            name="dashboard",
        )
        app.mount(
            "/training-data/images",
            StaticFiles(directory=app.state.training_store.image_directory),
            name="training-images",
        )

        @app.get("/", include_in_schema=False)
        async def dashboard() -> FileResponse:
            return FileResponse(dashboard_index)

        @app.get("/train", include_in_schema=False)
        async def training_dashboard() -> FileResponse:
            return FileResponse(dashboard_index)

    return app


def _try_build_inspector(settings: Settings) -> Any | None:
    if (
        settings.rfdetr_checkpoint_path is None
        or settings.dinov2_model_path is None
        or settings.dinov2_classifier_path is None
    ):
        return None
    try:
        return build_inspector(settings)
    except (RuntimeError, OSError) as exc:
        logger.warning("vision inspector is unavailable: %s", exc)
        return None


def _try_build_training_geometry_predictor(
    settings: Settings,
    inspector: Any | None,
) -> TrainingGeometryPredictor | None:
    if inspector is not None:
        return TrainingGeometryPredictor(
            inspector._tote_detector,
            inspector._layout_detector,
        )
    if settings.rfdetr_checkpoint_path is None:
        return None
    try:
        return build_training_geometry_predictor(settings)
    except (RuntimeError, OSError) as exc:
        logger.warning("training geometry prediction is unavailable: %s", exc)
        return None


app = create_app()
