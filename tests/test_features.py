import numpy as np
import pandas as pd
import pytest

from esif_forecasting.features import HORIZONS, forecast_table, historical_features


def hourly_fixture(periods=160):
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    values = np.arange(periods, dtype=float)
    return pd.DataFrame({"target_kw": values, "it_power_kw": 2 * values,
                         "temperature_c": values / 10, "outside_air_humidity": 40}, index=index)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_target_alignment_and_no_future_features(horizon):
    hourly = hourly_fixture()
    frame = forecast_table(hourly, horizon)
    issue = hourly.index[30]
    row = frame.loc[issue]
    assert row["target_end"] == hourly.index[30 + horizon]
    assert row["target_start"] == hourly.index[29 + horizon]
    assert row["actual_kw"] == 30 + horizon
    assert row["persistence"] == 30
    assert row["seasonal_naive"] == 30 + horizon - 24

    changed = hourly.copy()
    changed.loc[changed.index > issue] = 999
    for group, features in historical_features(hourly).items():
        pd.testing.assert_frame_equal(
            features.loc[:issue], historical_features(changed)[group].loc[:issue]
        )
    if horizon == 24:
        np.testing.assert_array_equal(frame["persistence"], frame["seasonal_naive"])


def test_missing_hour_is_not_skipped():
    hourly = hourly_fixture()
    missing = hourly.index[31]
    frame = forecast_table(hourly.drop(missing), 1)
    assert pd.isna(frame.loc[hourly.index[30], "actual_kw"])
