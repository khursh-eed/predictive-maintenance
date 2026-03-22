from src.pipeline.training_pipeline import run_training

pipeline = run_training(
    train_df_url='/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/data/train_df.csv',
    test_df_url='/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/data/test_df.csv',
    rul_file_path='/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/data/CMaps/RUL_FD001.txt',
    
    model_name="xgb",
    model_params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8

    },

    rolling_windows=[5],
    lags=[2,5],
    diffs=False,
    use_scaling=True
)