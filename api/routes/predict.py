import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from api.services.model_service import ModelService
from src.config import SEASONAL_COLS, get_config
from src.data.features import seasonality_from_month
from src.db.prediction_logger import log_prediction

router = APIRouter()


def _required_client_cols(model_service: ModelService) -> list:
    """Feature columns a client must send: the trained model's inputs minus
    the seasonal terms derived server-side."""
    metadata = model_service.get_metadata() or {}
    cols = metadata.get("feature_columns") or get_config().model_feature_cols
    return [c for c in cols if c not in SEASONAL_COLS]


def _prepare_features(request: PredictRequest, model_service: ModelService, warnings: list) -> dict:
    """Validate the feature payload against the active schema, apply scenario
    adjustments, and derive server-side seasonality terms."""
    cfg = get_config()
    required = _required_client_cols(model_service)
    features_dict = dict(request.features)

    missing = [c for c in required if c not in features_dict]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features",
                "missing_features": missing,
                "hint": "GET /model-metadata lists the feature columns the model expects.",
            },
        )

    unknown = [c for c in features_dict if c not in required]
    for key in unknown:
        warnings.append(f"Ignored unknown feature '{key}'")
        features_dict.pop(key)

    if request.scenario_adjustments:
        for key, value in request.scenario_adjustments.items():
            if key in features_dict:
                features_dict[key] = value
                warnings.append(f"Applied scenario adjustment for {key}: {value}")
            else:
                warnings.append(f"Ignored unknown scenario adjustment '{key}'")

    # Type checks: everything that is not categorical must be numeric.
    categorical = set(cfg.categorical_features)
    for key, value in features_dict.items():
        if key in categorical:
            features_dict[key] = str(value)
            continue
        try:
            features_dict[key] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{key}' must be numeric, got {value!r}.",
            )

    if "month" in features_dict:
        month = int(features_dict["month"])
        if not 1 <= month <= 12:
            raise HTTPException(status_code=422, detail="Feature 'month' must be in 1-12.")
        month_sin, month_cos = seasonality_from_month(month)
        features_dict["month_sin"] = float(month_sin)
        features_dict["month_cos"] = float(month_cos)
    return features_dict


def _entity_fields(request: PredictRequest) -> tuple[str, dict]:
    cfg = get_config()
    entity = {
        col: str(request.features[col]) for col in cfg.entity_hierarchy if col in request.features
    }
    entity_id = entity.get(cfg.entity_col, "unknown")
    return entity_id, entity


def _log_response(request: PredictRequest, response: PredictResponse, endpoint, duration_ms):
    log_prediction(
        endpoint=endpoint,
        model_name=response.model_version,
        model_version=response.model_version,
        entity_id=response.entity_id,
        entity_context=response.entity,
        horizon=response.horizon,
        request_payload=dict(request.features),
        scenario_adjustments=request.scenario_adjustments,
        prediction={"predicted_value": response.predicted_value},
        feature_columns=response.feature_columns_used,
        warnings=response.warnings,
        duration_ms=duration_ms,
    )


def _build_response(
    request: PredictRequest, prediction: float, metadata: dict, warnings: list
) -> PredictResponse:
    entity_id, entity = _entity_fields(request)
    return PredictResponse(
        entity_id=entity_id,
        entity=entity,
        predicted_value=prediction,
        target_column=metadata.get("target_column", get_config().target_col),
        horizon=request.horizon,
        model_version=metadata.get("model_version", metadata.get("model_name", "unknown")),
        feature_columns_used=metadata.get("feature_columns", []),
        warnings=warnings,
    )


@router.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(
    request: PredictRequest, model_service: ModelService = Depends(ModelService.get_instance)
):
    started = time.perf_counter()
    warnings: list = []
    # Validate before the model check so schema errors are reported (422)
    # even when no model has been trained yet.
    features_dict = _prepare_features(request, model_service, warnings)

    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not currently loaded or trained.")

    try:
        prediction = model_service.predict(pd.DataFrame([features_dict]))[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}") from e

    duration_ms = (time.perf_counter() - started) * 1000
    metadata = model_service.get_metadata() or {}

    response = _build_response(request, prediction, metadata, warnings)
    _log_response(request, response, "/predict", duration_ms)
    return response


@router.post("/batch-predict", response_model=BatchPredictResponse, tags=["Inference"])
def batch_predict(
    request: BatchPredictRequest, model_service: ModelService = Depends(ModelService.get_instance)
):
    started = time.perf_counter()
    per_request_warnings: list[list] = []
    rows = []
    for req in request.requests:
        warnings: list = []
        rows.append(_prepare_features(req, model_service, warnings))
        per_request_warnings.append(warnings)

    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not currently loaded or trained.")

    try:
        # Single DataFrame -> one model.predict call for the whole batch
        batch_results = model_service.predict(pd.DataFrame(rows))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}") from e

    duration_ms = (time.perf_counter() - started) * 1000
    metadata = model_service.get_metadata() or {}

    responses = []
    for i, req in enumerate(request.requests):
        response = _build_response(req, batch_results[i], metadata, per_request_warnings[i])
        _log_response(req, response, "/batch-predict", duration_ms / len(request.requests))
        responses.append(response)

    return BatchPredictResponse(predictions=responses)
