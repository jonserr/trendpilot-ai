import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os


def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def time_aware_split(
    df: pd.DataFrame, date_col: str = "date", train_ratio: float = 0.7, val_ratio: float = 0.15
):
    """
    Splits the dataframe temporally to prevent data leakage.
    """
    df = df.sort_values(by=[date_col])

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def save_preprocessor(preprocessor, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str):
    return joblib.load(filepath)
