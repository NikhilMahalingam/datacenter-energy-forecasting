import pandas as pd
import pytest

from esif_forecasting.data import clean, hourly, validate


def test_right_labeled_hours_and_insufficient_samples():
    time = pd.date_range("2024-01-01", periods=7, freq="15min").delete(5)
    raw = pd.DataFrame({"ts": time, "cooling_kw": [1, 2, 3, 4, 100, 200]})
    result = hourly(clean(validate(raw, ["cooling_kw"]), ["cooling_kw"]), 0.75)
    assert result.iloc[0, 0] == 2.5
    assert result.index[0] == pd.Timestamp("2024-01-01 01:00")
    assert pd.isna(result.iloc[1, 0])


def test_conflicts_and_exact_repeats():
    raw = pd.DataFrame({"ts": ["2024-01-01"] * 2 + ["2024-01-02"] * 2,
                        "cooling_kw": [1, 2, 3, 3]})
    result = clean(validate(raw, ["cooling_kw"]), ["cooling_kw"])
    assert result["cooling_kw"].tolist() == [3]
    assert result.index.tz is None


def test_invalid_schema():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(pd.DataFrame({"ts": ["2024-01-01"]}), ["cooling_kw"])


def test_manual_zipped_csv(tmp_path):
    from esif_forecasting.data import input_path, read_table

    path = tmp_path / "manual.csv.zip"
    original = pd.DataFrame({"ts": ["2024-01-01"], "cooling_kw": [3.0]})
    original.to_csv(path, index=False)
    pd.testing.assert_frame_equal(read_table(input_path(tmp_path, "power", path)), original)


def test_prepare_and_accounting(tmp_path):
    from esif_forecasting.data import POWER, prepare

    timestamps = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    power = pd.DataFrame({"ts": timestamps, **dict(zip(POWER, [100, 3, 4, 5])),
                          "plug_and_light_kw": 2, "pue": 1.14})
    weather = pd.DataFrame({"ts": timestamps[4:], "outside_air_temp": 50,
                            "outside_air_humidity": 45})
    power_path, weather_path = tmp_path / "power.parquet", tmp_path / "weather.parquet"
    power.to_parquet(power_path)
    weather.to_parquet(weather_path)
    hourly, report = prepare(power_path, weather_path, "mechanical", 0.75)
    assert hourly["target_kw"].tolist() == [12, 12]
    assert hourly["temperature_c"].iloc[-1] == 10
    assert pd.isna(hourly["temperature_c"].iloc[0])
    assert report["exact_raw_timestamp_matches"] == 4
    assert report["power"]["timezone"] == "UTC"
    assert report["hourly"]["longest_incomplete_stretch_hours"] == 1
    residuals = report["pue_accounting"]["absolute_pue_residual"]
    assert residuals["including_pump"]["median"] == pytest.approx(0)
    assert residuals["excluding_pump"]["median"] == pytest.approx(0.05)
