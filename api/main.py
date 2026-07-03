import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import health, history, metadata, predict
from api.services.model_service import ModelService


@asynccontextmanager
async def lifespan(_: FastAPI):
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
    version="3.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Optional API-key auth: set the API_KEY environment variable to require
    an X-API-Key header on all endpoints except /health and the docs."""
    expected = os.environ.get("API_KEY")
    open_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    if expected and request.url.path not in open_paths:
        if request.headers.get("X-API-Key") != expected:
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
