import mlflow
import numpy as np
import pandas as pd

from src.components.feature_engineering import FeatureEngineer
from src.components.data_preprocessing import prepare_data, drop_columns, ScalerWrapper, validate_dataframe
from src.components.model_trainer import get_model, train_model
from src.components.evaluator import evaluate
from src.utils.common import save_object

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RUL_Prediction")

class RULPipeline:
    def __init__(self, fe, scaler, model):
        self.fe = fe
        self.scaler = scaler
        self.model = model

    def predict(self, df):
        df = self.fe.transform(df)
        X = drop_columns(df)
        X = self.scaler.transform(X)
        return self.model.predict(X)

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

        # x and y split and removing the extra columns
        X_train, y_train = prepare_data(train_df)
        X_train = drop_columns(X_train)

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


        # Load RUL file
        y_test_df = pd.read_csv(rul_file_path, header=None)
        y_test_df.columns = ["RUL"]


        # True values
        y_true = y_test_df["RUL"].values
        

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
        save_object("artifacts/rul_pipeline.pkl", pipeline)

        print("Metrics:", metrics)

        return pipeline