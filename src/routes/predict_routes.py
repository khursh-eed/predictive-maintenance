from fastapi import APIRouter, UploadFile, File
import pandas as pd
from src.pipeline.predicting_pipeline import predict_from_df

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Convert uploaded file → DataFrame
        df = pd.read_csv(file.file)

        result = predict_from_df(df)

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }