class FeatureEngineer:
    def __init__(self, rolling_windows=None, lags=None, diffs=False):
        self.rolling_windows = rolling_windows
        self.lags = lags
        self.diffs = diffs

    def transform(self, df):
        df = df.copy()
        
        sensor_cols = [col for col in df.columns if "sensor" in col]

        # Rolling Features
        if self.rolling_windows:
            for window in self.rolling_windows:
                for col in sensor_cols:
                    df[f"{col}_roll_mean_{window}"] = (
                        df.groupby("unit")[col]
                        .transform(lambda x: x.rolling(window).mean())
                    )

        # Lag Features
        if self.lags:
            for lag in self.lags:
                for col in sensor_cols:
                    df[f"{col}_lag_{lag}"] = (
                        df.groupby("unit")[col].shift(lag)
                    )

        # Diff Features
        if self.diffs:
            for col in sensor_cols:
                df[f"{col}_diff"] = (
                    df.groupby("unit")[col].diff()
                )

        df = df.dropna()

        return df