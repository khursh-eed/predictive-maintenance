from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_model(model_name, params):
    if model_name == "rf":
        return RandomForestRegressor(**params)
    elif model_name == "xgb":
        return XGBRegressor(**params)
    else:
        raise ValueError("Invalid model name")


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model