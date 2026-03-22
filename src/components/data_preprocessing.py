from sklearn.preprocessing import StandardScaler


class ScalerWrapper:
    def __init__(self, use_scaling=False):
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None

    # used for training data
    def fit_transform(self, X):
        if self.use_scaling:
            return self.scaler.fit_transform(X)
        return X

    # used for test data
    def transform(self, X):
        if self.use_scaling:
            return self.scaler.transform(X)
        return X

# seperatign x and y
def prepare_data(df):
    X = df.drop(columns=["RUL"])
    y = df["RUL"]
    return X, y

# dropping unit n cycle - not requied for training (might overfit)
def drop_columns(df):
    return df.drop(columns=["Unnamed: 0","unit", "cycle"], errors="ignore")

def validate_dataframe(df, name):
    required_cols = ["unit", "cycle"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{name} missing column: {col}")

    sensor_cols = [col for col in df.columns if "sensor" in col]
    
    if len(sensor_cols) == 0:
        raise ValueError(f"{name} has no sensor columns")

    print(f"{name} loaded successfully with {len(df)} rows")