from pathlib import Path

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from esif_forecasting.experiment import evaluate, metrics
from test_features import hourly_fixture


def test_models_and_ablations_share_origins():
    hourly = hourly_fixture(480)
    hourly.loc[hourly.index[370], "temperature_c"] = float("nan")
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
    with threadpool_limits(limits=1):
        predictions, report = evaluate(hourly, config)
    assert predictions.notna().all().all()
    assert set(predictions.horizon_hours) == {1, 24}
    assert not predictions.duplicated(["horizon_hours", "issue_time"]).any()
    assert not any(hourly.index[370] == predictions.issue_time)
    for fold in report["folds"]:
        for detail in fold["tuning"].values():
            assert pd.Timestamp(detail["inner_last_label"]) <= (
                pd.Timestamp(detail["first_validation_issue"]) - pd.Timedelta(hours=24))
    table = metrics(predictions, report, config)
    pooled = table.query("scope == 'pooled'")
    assert len(pooled) == 12
    for horizon, group in pooled.groupby("horizon_hours"):
        assert set(group.test_rows) == {(predictions.horizon_hours == horizon).sum()}
    baselines = pooled.query("horizon_hours == 24 and feature_set == 'baseline'")
    assert baselines.mae_kw.nunique() == 1
    assert baselines.mae_skill.eq(0).all()
