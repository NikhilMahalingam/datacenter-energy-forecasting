import pandas as pd
import pytest

from esif_forecasting.evaluation import Fold, scores, split
from esif_forecasting.features import forecast_table
from test_features import hourly_fixture


@pytest.mark.parametrize("horizon", (1, 24))
def test_train_labels_stop_before_first_test_issue(horizon):
    hourly = hourly_fixture()
    frame = forecast_table(hourly, horizon).dropna()
    fold = Fold("test", hourly.index[110], hourly.index[-1])
    train, test = split(frame, fold, 24)
    assert train["target_end"].max() <= test.index.min() - pd.Timedelta(hours=24)
    assert test["target_start"].min() == fold.start


def test_skill_defined_only_for_nonzero_persistence_error():
    actual = pd.Series([3.0, 3.0])
    baseline = pd.Series([1.0, 1.0])
    assert scores(actual, actual, baseline)["mae_skill"] == 1
    perfect = scores(actual, actual, actual)
    assert perfect["mae_skill"] is None
    assert perfect["r2"] is None


@pytest.mark.parametrize(("periods", "expected"), [
    (24 * 30, "chronological_75_25"),
    (24 * 365, "monthly_expanding"),
    (24 * 732, "last_complete_year"),
])
def test_protocol_follows_measured_coverage(periods, expected):
    from esif_forecasting.evaluation import evaluation_folds

    scheme, folds = evaluation_folds(hourly_fixture(periods), 4, 0.95)
    assert scheme == expected
    assert all(fold.start < fold.end for fold in folds)
