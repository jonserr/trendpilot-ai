from fastapi import APIRouter, Depends, HTTPException

from api.services.model_service import ModelService
from src.config import get_config
from src.data.loaders import load_data

router = APIRouter()


@router.get("/model-metadata", tags=["Model"])
def get_metadata(model_service: ModelService = Depends(ModelService.get_instance)):
    metadata = model_service.get_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return metadata


@router.get("/dataset-schema", tags=["Model"])
def get_dataset_schema():
    """The active dataset configuration: required columns and feature layout."""
    cfg = get_config()
    return {
        "name": cfg.name,
        "date_column": cfg.date_col,
        "target_column": cfg.target_col,
        "entity_column": cfg.entity_col,
        "entity_hierarchy": list(cfg.entity_hierarchy),
        "categorical_features": list(cfg.categorical_features),
        "numeric_features": list(cfg.numeric_features),
        "lags": list(cfg.lags),
        "rolling_windows": list(cfg.rolling_windows),
        "required_raw_columns": cfg.raw_required_cols,
        "client_feature_columns": cfg.client_feature_cols,
    }


@router.get("/entities", tags=["Model"])
def get_entities():
    """Available entities from the active dataset, grouped by the configured
    entity hierarchy (e.g. region -> city -> zipcode)."""
    cfg = get_config()
    if not cfg.data_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Run 'make fetch-data' or provide your own dataset.",
        )

    df = load_data(cfg=cfg)
    hierarchy = [c for c in cfg.entity_hierarchy if c in df.columns]
    if len(hierarchy) == 1:
        return {
            "entity_column": cfg.entity_col,
            "entities": sorted(df[cfg.entity_col].unique().tolist()),
        }

    parents, leaf = hierarchy[:-1], hierarchy[-1]
    entities: dict = {}
    for keys, group in df.groupby(parents):
        keys = keys if isinstance(keys, tuple) else (keys,)
        node = entities
        for key in keys[:-1]:
            node = node.setdefault(str(key), {})
        node[str(keys[-1])] = sorted(group[leaf].unique().tolist())
    return {"entity_column": cfg.entity_col, "hierarchy": hierarchy, "entities": entities}
