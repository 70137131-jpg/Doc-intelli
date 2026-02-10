from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.exceptions import DocIntelliError
from app.core.logging import setup_logging, get_logger
from app.core.middleware import CorrelationIDMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware
from app.api.v1.router import api_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Document Intelligence Platform")

    # Initialize MinIO buckets
    try:
        from app.services.storage_service import StorageService

        storage = StorageService()
        storage.ensure_buckets()
        logger.info("MinIO buckets initialized")
    except Exception as e:
        logger.warning(f"MinIO initialization skipped: {e}")

    yield

    logger.info("Shutting down Document Intelligence Platform")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-Powered Document Intelligence Platform with RAG, Classification, and Agentic Workflows",
    lifespan=lifespan,
)

# Middleware stack (order matters — outermost first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CorrelationIDMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


# Global exception handler
@app.exception_handler(DocIntelliError)
async def doc_intelli_exception_handler(request: Request, exc: DocIntelliError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500},
    )


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for load balancers and container orchestration."""
    return {"status": "healthy", "version": settings.app_version}
