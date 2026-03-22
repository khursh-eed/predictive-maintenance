import pickle
import pandas as pd
from src.utils.common import load_object

# load pipeline
with open('/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/artifacts/rul_pipeline_20260322_210747.pkl', "rb") as f:
    pipeline = load_object(file_path=f)


def predict_from_csv(file_path: str):
    try:
        # load csvfile
        df = pd.read_csv(file_path)

        if df.empty:
            return {"error": "CSV file is empty"}

        predictions = pipeline.predict(df)

        df_last = df.groupby("unit").tail(1).copy()
        df_last["RUL_prediction"] = predictions

        return df_last[["unit", "cycle", "RUL_prediction"]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    result = predict_from_csv('/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/prediction_data/p1.csv')
    print(result)