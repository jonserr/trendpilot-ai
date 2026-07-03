from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

FeatureValue = Union[float, int, str]


class PredictRequest(BaseModel):
    """Dataset-agnostic prediction request.

    ``features`` must contain every column the trained model expects —
    reported by ``/model-metadata`` as ``feature_columns``, except derived
    seasonal terms (``month_sin``/``month_cos``), which are computed
    server-side. Requests are validated against the active dataset config
    and trained model at runtime.
    """

    features: Dict[str, FeatureValue] = Field(
        ..., description="Feature name -> value mapping matching the active dataset schema"
    )
    horizon: int = Field(1, ge=1, le=24, description="Forecast horizon in dataset periods")
    scenario_adjustments: Optional[dict] = Field(
        None, description="Feature overrides applied before inference (what-if scenarios)"
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    entity_id: str
    entity: Dict[str, str] = Field(
        default_factory=dict, description="Entity-hierarchy values from the request"
    )
    predicted_value: float
    target_column: str
    horizon: int = 1
    model_version: str
    feature_columns_used: List[str]
    warnings: List[str] = []


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
