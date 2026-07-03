import pandas as pd
from src.data.preprocess import time_aware_split


def test_time_aware_split():
    df = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=100, freq="D"), "val": range(100)}
    )

    train, val, test = time_aware_split(df, train_ratio=0.7, val_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Check temporal ordering
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
