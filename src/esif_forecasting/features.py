import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
HORIZONS = (1, 24)
LAGS = (0, 1, 2, 3, 6, 24)
WINDOWS = (6, 24)
POWER_HISTORY = ("target_kw", "it_power_kw")
WEATHER_HISTORY = ("temperature_c", "outside_air_humidity")
BASELINES = ("persistence", "seasonal_naive")


def forecast_table(hourly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError(f"Expected a forecast horizon in {HORIZONS}")
    index = hourly.index
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Hourly index must be sorted and unique")
    hourly = hourly.asfreq("h")
    target = hourly["target_kw"]
    result = pd.DataFrame(index=hourly.index.rename("issue_time"))
    result["target_end"] = result.index + horizon * HOUR
    result["target_start"] = result["target_end"] - HOUR
    result["actual_kw"] = target.shift(-horizon)
    result["persistence"] = target
    result["seasonal_naive"] = target.shift(24 - horizon)
    return result


def historical_features(hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    hourly = hourly.asfreq("h")
    groups = {}
    for name, signals in (("power", POWER_HISTORY), ("weather", WEATHER_HISTORY)):
        features = {}
        for signal in signals:
            values = hourly[signal]
            for lag in LAGS:
                features[f"{signal}_lag{lag}"] = values.shift(lag)
            for window in WINDOWS:
                features[f"{signal}_mean{window}"] = values.rolling(window).mean()
        groups[name] = pd.DataFrame(features, index=hourly.index)
    for label, values, period in (("hour", hourly.index.hour, 24),
                                  ("weekday", hourly.index.dayofweek, 7)):
        phase = 2 * np.pi * values / period
        groups["power"][f"{label}_sin"] = np.sin(phase)
        groups["power"][f"{label}_cos"] = np.cos(phase)
    return groups
