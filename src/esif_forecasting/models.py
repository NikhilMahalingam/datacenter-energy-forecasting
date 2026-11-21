from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HOUR


def estimator(model: str, value: float, seed: int):
    if model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=value))
    if model == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_leaf_nodes=value, max_iter=100, learning_rate=0.06,
            min_samples_leaf=20, early_stopping=False, random_state=seed,
        )
    raise ValueError(f"Unknown estimator: {model}")


def fit_predict(model: str, candidates: Sequence, train: pd.DataFrame, test: pd.DataFrame,
                columns: list[str], seed: int, gap_hours: int) -> tuple[np.ndarray, dict]:
    boundary = int(len(train) * 0.8)
    validation = train.iloc[boundary:]
    inner = train.loc[train["target_end"] <= validation.index[0] - gap_hours * HOUR]
    if len(inner) < 2:
        raise ValueError("Insufficient labeled history before internal validation")
    trials = []
    for parameter in candidates:
        candidate = estimator(model, parameter, seed)
        candidate.fit(inner[columns], inner["actual_kw"])
        prediction = candidate.predict(validation[columns])
        mae = float(np.abs(validation["actual_kw"] - prediction).mean())
        trials.append({"parameter": parameter, "mae_kw": mae})
    best = min(trials, key=lambda trial: trial["mae_kw"])["parameter"]
    final = estimator(model, best, seed)
    final.fit(train[columns], train["actual_kw"])
    details = {
        "selected": best, "validation_trials": trials,
        "inner_training_rows": len(inner), "inner_last_label": str(inner["target_end"].max()),
        "validation_rows": len(validation), "first_validation_issue": str(validation.index[0]),
        "last_validation_label": str(validation["target_end"].max()),
    }
    return final.predict(test[columns]), details
