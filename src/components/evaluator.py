import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def nasa_score(y_true, y_pred):
    score = 0

    for i in range(len(y_true)):
        d = y_pred[i] - y_true[i]

        if d < 0:
            score += np.exp(-d / 13) - 1
        else:
            score += np.exp(d / 10) - 1

    return score


def evaluate(y_true, y_pred):

    # checks if the size of predicted and true is the same (otherwise cant compare) - if wrong means somewhere in the preprocessing steps went wrong
    assert len(y_true) == len(y_pred), \
        f"Mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    score = nasa_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "nasa_score": score
    }