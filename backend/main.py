import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api import health, model_info, spike, transaction_risk
from backend.core.auth import require_api_key
from backend.core.config import settings
from backend.core.logging import configure_logging, get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("Starting up - pre-loading risk engine artifacts (model, spike table, feature reference stats)...")
    startup_health = health.run_health_check()
    app.state.startup_health = startup_health
    if startup_health["status"] != "healthy":
        failed = [name for name, c in startup_health["components"].items() if not c["loaded"]]
        logger.warning("Startup health check is DEGRADED - failed component(s): %s", failed)
    else:
        logger.info("Startup health check: healthy. All artifacts loaded and cached for the life of this process.")
    if settings.api_key == "local-dev-key-CHANGE-ME":
        logger.warning(
            "RISK_API_API_KEY is unset and using the committed default dev key. This is fine for "
            "local development but MUST be overridden before any shared/hosted deployment - the "
            "default value is public (it's in version control)."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description=(
        "Thin FastAPI service layer over src/risk_engine/ (Phase 4). Prototype trained and "
        "evaluated on the synthetic PaySim dataset only - see GET /api/v1/model-info for the "
        "data-provenance caveat. Single shared-secret API key auth (Phase 7) on all endpoints "
        "except /api/v1/health - see reports/phase7_summary.md for exactly what that does and "
        "does not protect against. No rate-limiting."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,  # env-driven (RISK_API_CORS_ALLOW_ORIGINS) - see module docstring
    allow_credentials=False,  # an API key header is not a cookie - nothing to allow credentials for
    allow_methods=["GET", "POST"],  # the only methods this API actually exposes
    allow_headers=["Content-Type", settings.api_key_header_name],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    extra = {
        "endpoint": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
    }
    domain_extra = getattr(request.state, "log_extra", None)
    if domain_extra:
        extra.update({k: v for k, v in domain_extra.items() if v is not None})

    logger.info("request completed", extra=extra)
    return response


# /health is intentionally NOT behind require_api_key - see module docstring.
app.include_router(health.router, prefix="/api/v1", tags=["health"])

app.include_router(
    model_info.router, prefix="/api/v1", tags=["model-info"], dependencies=[Depends(require_api_key)]
)
app.include_router(
    transaction_risk.router,
    prefix="/api/v1/transactions",
    tags=["transactions"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    spike.router, prefix="/api/v1/spikes", tags=["spikes"], dependencies=[Depends(require_api_key)]
)
