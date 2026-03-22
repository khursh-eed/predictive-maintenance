import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
import os

from src.components.feature_engineering import FeatureEngineer
from src.components.data_preprocessing import prepare_data, drop_columns, ScalerWrapper, validate_dataframe
from src.components.model_trainer import get_model, train_model
from src.components.evaluator import evaluate
from src.utils.common import save_object

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RUL_Prediction")

# class RULPipeline:
#     def __init__(self, fe, scaler, model):
#         self.fe = fe
#         self.scaler = scaler
#         self.model = model

#     def predict(self, df):
#         df = self.fe.transform(df)
#         X = drop_columns(df)
#         X = self.scaler.transform(X)
#         return self.model.predict(X)

class RULPipeline:
    def __init__(self, fe, scaler, model):
        self.fe = fe
        self.scaler = scaler
        self.model = model

    def predict(self, df):
        # validation
        required_cols = [
            "unit", "cycle", "setting1", "setting2",
            "sensor2", "sensor3", "sensor4", "sensor6",
            "sensor7", "sensor8", "sensor9", "sensor11",
            "sensor12", "sensor13", "sensor14", "sensor15",
            "sensor17", "sensor20", "sensor21"
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if df.empty:
            raise ValueError("Input dataframe is empty")

        # preproceesing 
        df = df.sort_values(by=["unit", "cycle"])
        df = self.fe.transform(df)
        # df['cycle_norm'] = df['cycle'] / df.groupby('unit')['cycle'].transform('max')
        df_last = df.groupby("unit").tail(1)
        X = drop_columns(df_last)
        X = self.scaler.transform(X)
        preds = self.model.predict(X)

        return preds

# for testing - we need cal the rul only for the last cycle for every unit
def get_last_cycle(df):
    return df.groupby("unit").tail(1)


def run_training(train_df_url, test_df_url,rul_file_path,
                 model_name, model_params,
                 rolling_windows=None, lags=None, diffs=False,
                 use_scaling=False):

    with mlflow.start_run():

        train_df = pd.read_csv(train_df_url)
        test_df = pd.read_csv(test_df_url)

        # validating the dfs

        validate_dataframe(train_df, "Train DF")
        validate_dataframe(test_df, "Test DF")


        # feature engineering 
        fe = FeatureEngineer(rolling_windows, lags, diffs)

        train_df = fe.transform(train_df)
        test_df = fe.transform(test_df)

        # train_df['cycle_norm'] = train_df['cycle'] / train_df.groupby('unit')['cycle'].transform('max')
        # test_df['cycle_norm'] = test_df['cycle'] / test_df.groupby('unit')['cycle'].transform('max')
        # x and y split and removing the extra columns
        X_train, y_train = prepare_data(train_df)
        X_train = drop_columns(X_train)

        # clipping the higher RUL to avoid very high prediction
        y_train = y_train.clip(upper=125)
        y_train = np.log1p(y_train)

        # scaling 
        scaler = ScalerWrapper(use_scaling)
        X_train = scaler.fit_transform(X_train)

        # training model
        model = get_model(model_name, model_params)
        model = train_model(model, X_train, y_train)

        # applyign the same preprocessing steps on train 
        test_last = get_last_cycle(test_df)

        X_test = drop_columns(test_last)
        X_test = scaler.transform(X_test)

        y_pred = model.predict(X_test)
        y_pred = np.expm1(y_pred)


        # Load RUL file
        y_test_df = pd.read_csv(rul_file_path, header=None)
        y_test_df.columns = ["RUL"]


        # True values
        y_true = y_test_df["RUL"].clip(upper=125).values
        
        # checking the evaluation metrics
        metrics = evaluate(y_true, y_pred)

        # logging all of this in mlflow
        mlflow.log_param("model", model_name)
        mlflow.log_params(model_params)
        mlflow.log_param("rolling_windows", rolling_windows)
        mlflow.log_param("lags", lags)
        mlflow.log_param("diffs", diffs)
        mlflow.log_param("use_scaling", use_scaling)

        mlflow.log_metrics(metrics)

        # 🔹 Save pipeline
        pipeline = RULPipeline(fe, scaler, model)

        os.makedirs("/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/artifacts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/artifacts/rul_pipeline_{timestamp}.pkl"
        save_object(file_path, pipeline)

        print("Metrics:", metrics)

        return pipeline