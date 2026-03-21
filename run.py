from src.pipeline.training_pipeline import run_training

pipeline = run_training(
    train_df_url='/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/data/train_df.csv',
    test_df_url='/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/data/test_df.csv',
    rul_file_path='/Users/khursheedfatima/Documents/Projects/Predictive_Maintenance/data/CMaps/RUL_FD001.txt',
    
    model_name="rf",
    model_params={
        "n_estimators": 50,
        "max_depth": 5
    },

    rolling_windows=[5],
    lags=[1],
    diffs=True,
    use_scaling=False
)