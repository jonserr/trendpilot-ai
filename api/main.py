import hmac
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import health, history, metadata, predict
from api.services.model_service import ModelService
from src.utils.secrets import SecretConfigurationError, get_api_key

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_api_key()  # Reject missing required keys and broken file configuration before serving.
    # Load TensorFlow + the model in the background so /health responds
    # immediately after boot instead of blocking for several seconds.
    ModelService.warm_up_async()
    yield


app = FastAPI(
    title="Trendpilot AI - Forecasting API",
    description=(
        "Inference service for the Trendpilot forecasting platform. "
        "The feature schema is dataset-driven — see /dataset-schema."
    ),
    version=version("trendpilot-ai"),
    lifespan=lifespan,
)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Authenticate protected endpoints; invalid secret configuration fails closed."""
    try:
        expected = get_api_key()
    except SecretConfigurationError:
        logger.error("API credential configuration is invalid")
        return JSONResponse(status_code=503, content={"detail": "API authentication unavailable"})
    open_paths = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
    if expected and request.url.path not in open_paths:
        supplied = request.headers.get("X-API-Key", "").encode("utf-8")
        if not hmac.compare_digest(supplied, expected.encode("ascii")):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


# Include routes
app.include_router(health.router)
app.include_router(metadata.router)
app.include_router(predict.router)
app.include_router(history.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
