import pytest
from pydantic import ValidationError

from api.schemas import PredictRequest

VALID_FEATURES = {
    "region": "WA",
    "location": "Seattle, WA",
    "inventory": 1500,
    "mortgage_rate": 6.5,
    "days_on_market": 30,
    "sale_to_list_ratio": 1.0,
    "new_listings": 300,
    "month": 6,
    "quarter": 2,
    "year": 2026,
    "median_sale_price_lag_1": 800000.0,
    "median_sale_price_lag_2": 795000.0,
    "median_sale_price_lag_3": 790000.0,
    "median_sale_price_rolling_mean_3": 795000.0,
    "median_sale_price_rolling_mean_6": 780000.0,
}


def test_predict_request_valid():
    request = PredictRequest(features=VALID_FEATURES)
    assert request.features["location"] == "Seattle, WA"
    assert request.features["month"] == 6


def test_predict_request_defaults():
    request = PredictRequest(features=VALID_FEATURES)
    assert request.horizon == 1
    assert request.scenario_adjustments is None


def test_features_required():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_horizon_bounds():
    with pytest.raises(ValidationError):
        PredictRequest(features=VALID_FEATURES, horizon=0)
    with pytest.raises(ValidationError):
        PredictRequest(features=VALID_FEATURES, horizon=25)
