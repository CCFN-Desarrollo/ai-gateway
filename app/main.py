import logging
import sys
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings


def _configure_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    _configure_logging()
    settings.validate_runtime()
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting %s v%s [%s]",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
    )
    yield
    logger.info("Shutting down %s", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "AI Orchestration Gateway — receives document images from external systems "
        "(WhatsApp bots, CRMs, etc.), routes them through OCR and Vision AI pipelines, "
        "applies confidence scoring, and returns structured validation results."
    ),
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)

app.include_router(api_router, prefix="/api/v1")
