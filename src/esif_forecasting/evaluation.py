from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import BASELINES, HOUR, HORIZONS, forecast_table


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def evaluation_folds(hourly: pd.DataFrame, initial_months: int,
                     year_completeness: float) -> tuple[str, list[Fold]]:
    coverage = hourly[["target_kw", "it_power_kw", "temperature_c",
                       "outside_air_humidity"]].notna().all(axis=1)
    if not coverage.any():
        raise ValueError("Power and weather have no complete hourly overlap")
    coverage.index = coverage.index - HOUR
    observed = coverage[coverage].index
    first, last = observed[0], observed[-1]
    complete_years = []
    for year in range(first.year + 1, last.year + 1):
        start = pd.Timestamp(year, 1, 1, tz=first.tz)
        end = pd.Timestamp(year + 1, 1, 1, tz=first.tz)
        counts = coverage.loc[(coverage.index >= start) & (coverage.index < end)].sum()
        prior = coverage.loc[coverage.index < start].sum()
        spans_year = first <= start and last + HOUR >= end
        if spans_year and counts / ((end - start) / HOUR) >= year_completeness and prior >= 7008:
            complete_years.append(Fold(str(year), start, end))
    if complete_years:
        return "last_complete_year", complete_years[-1:]

    if last - first >= pd.Timedelta(days=270):
        start = first.normalize().replace(day=1) + pd.offsets.MonthBegin(initial_months)
        boundaries = pd.date_range(start, last + HOUR, freq="MS")
        if len(boundaries) >= 2:
            return "monthly_expanding", [
                Fold(start.strftime("%Y-%m"), start, end)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]

    boundary = (first + (last - first) * 0.75).ceil("h")
    return "chronological_75_25", [Fold("holdout", boundary, last + HOUR)]


def split(frame: pd.DataFrame, fold: Fold, gap_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame.loc[(frame["target_start"] >= fold.start) & (frame["target_end"] <= fold.end)]
    if test.empty:
        return frame.iloc[:0], test
    cutoff = test.index.min() - gap_hours * HOUR
    train = frame.loc[frame["target_end"] <= cutoff]
    return train, test


def scores(actual: pd.Series, predicted: pd.Series, persistence: pd.Series) -> dict:
    error = (predicted - actual).to_numpy()
    mae = float(np.abs(error).mean())
    squared = float(np.square(error).sum())
    total = float(np.square(actual - actual.mean()).sum())
    reference = float((persistence - actual).abs().mean())
    return {
        "mae_kw": mae, "rmse_kw": float(np.sqrt(squared / len(actual))),
        "r2": 1 - squared / total if len(actual) > 1 and total else None,
        "mae_skill": 1 - mae / reference if reference else None,
    }


def baseline_predictions(hourly: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    predictions = []
    for horizon in HORIZONS:
        horizon_predictions = []
        frame = forecast_table(hourly, horizon).dropna(subset=["actual_kw", *BASELINES])
        for fold in folds:
            _, test = split(frame, fold, 0)
            if not test.empty:
                horizon_predictions.append(test.assign(horizon_hours=horizon, fold=fold.name))
        if not horizon_predictions:
            raise ValueError(f"No baseline observations at h={horizon} in test windows")
        predictions.extend(horizon_predictions)
    return pd.concat(predictions).reset_index()
