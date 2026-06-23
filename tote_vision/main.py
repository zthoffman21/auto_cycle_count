import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tote_vision.adapters.local_artifacts import LocalArtifactStore
from tote_vision.adapters.training_dataset import TrainingDatasetStore
from tote_vision.api.dashboard_routes import router as dashboard_router
from tote_vision.api.routes import router
from tote_vision.api.training_routes import router as training_router
from tote_vision.bootstrap import build_inspector
from tote_vision.config import Settings


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
    app.state.inspector = build_inspector(settings)
    app.include_router(router)
    if settings.dashboard_enabled:
        app.state.artifact_store = LocalArtifactStore(settings.artifact_directory)
        app.state.max_upload_bytes = settings.max_upload_bytes
        app.state.training_store = TrainingDatasetStore(settings.training_directory)
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


app = create_app()
