import pandas as pd
from src.utils.common import load_object
from src.components.data_preprocessing import drop_columns


def predict(test_df):
    pipeline = load_object("artifacts/rul_pipeline.pkl")

    preds = pipeline.predict(test_df)

    return preds