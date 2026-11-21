from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from time import perf_counter

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .data import TARGETS, input_path, prepare
from .download import CATALOG, LICENSE
from .evaluation import evaluation_folds, scores, split
from .features import BASELINES, HORIZONS, forecast_table, historical_features
from .models import fit_predict

MODEL_PARAMETERS = {"ridge": "ridge_alpha", "hist_gradient_boosting": "boosting_leaves"}


def configurations(config: dict) -> None:
    horizons = config["horizon_hours"]
    if not horizons or len(set(horizons)) != len(horizons) or not set(horizons) <= set(HORIZONS):
        raise ValueError("horizon_hours must contain distinct values from [1, 24]")
    if config["gap_hours"] < 0 or config["initial_months"] < 1:
        raise ValueError("Expected a nonnegative embargo and positive initial_months")
    if not 0 < config["year_completeness"] <= 1 or config["target"] not in TARGETS:
        raise ValueError("Invalid year completeness or target")
    if not config["ridge_alpha"] or not all(v > 0 for v in config["ridge_alpha"]):
        raise ValueError("Expected positive ridge_alpha candidates")
    if not config["boosting_leaves"] or not all(isinstance(v, int) and v >= 2
                                               for v in config["boosting_leaves"]):
        raise ValueError("Expected integer boosting_leaves candidates of at least two")


def evaluate(hourly: pd.DataFrame, config: dict):
    configurations(config)
    scheme, folds = evaluation_folds(hourly, config["initial_months"], config["year_completeness"])
    groups = historical_features(hourly)
    features = groups["power"].join(groups["weather"])
    sets = {"power": groups["power"].columns.tolist(),
            "power_weather": features.columns.tolist()}
    predictions, records = [], []
    for horizon in config["horizon_hours"]:
        observations = forecast_table(hourly, horizon)
        frame = observations.join(features).dropna()
        horizon_results = []
        for fold in folds:
            train, test = split(frame, fold, config["gap_hours"])
            detail = {
                "horizon_hours": horizon, "fold": fold.name,
                "test_window_start": str(fold.start), "test_window_end": str(fold.end),
                "grid_origins": len(frame.index.union(observations.index)),
                "eligible_origins": len(frame), "train_rows": len(train), "test_rows": len(test),
            }
            records.append(detail)
            if test.empty:
                detail["skipped"] = "no eligible test origins"
                continue
            if len(train) < 2:
                raise ValueError(f"Insufficient training history for {fold.name}, h={horizon}")
            detail.update(train_first_issue=str(train.index[0]), train_last_issue=str(train.index[-1]),
                          train_last_label=str(train["target_end"].max()), tuning={})
            output = test[observations.columns].copy()
            for model, parameter in MODEL_PARAMETERS.items():
                for feature_set, columns in sets.items():
                    name = f"{model}__{feature_set}"
                    output[name], detail["tuning"][name] = fit_predict(
                        model, config[parameter], train, test, columns,
                        config["seed"], config["gap_hours"],
                    )
            horizon_results.append(output.assign(horizon_hours=horizon, fold=fold.name))
        if not horizon_results:
            raise ValueError(f"No eligible test origins for h={horizon}")
        predictions.extend(horizon_results)
    return pd.concat(predictions).rename_axis("issue_time").reset_index(), {
        "scheme": scheme, "folds": records, "feature_columns": sets,
    }


def metric_rows(group, horizon, scope, label, models):
    for column in models:
        model, _, feature_set = column.partition("__")
        yield {
            "horizon_hours": horizon, "scope": scope, "fold": label, "model": model,
            "feature_set": feature_set or "baseline", "test_rows": len(group),
            "first_issue": group.issue_time.min(), "last_issue": group.issue_time.max(),
            "first_target_start": group.target_start.min(), "last_target_end": group.target_end.max(),
            **scores(group.actual_kw, group[column], group.persistence),
        }


def metrics(predictions: pd.DataFrame, evaluation: dict, config: dict) -> pd.DataFrame:
    models = [*BASELINES, *(col for col in predictions if "__" in col)]
    rows = []
    for horizon, group in predictions.groupby("horizon_hours"):
        rows.extend(metric_rows(group, horizon, "pooled", "all", models))
        for fold, group_fold in group.groupby("fold"):
            rows.extend(metric_rows(group_fold, horizon, "fold", fold, models))
        season = group.target_start.dt.month.mod(12).floordiv(3).map(
            {0: "DJF", 1: "MAM", 2: "JJA", 3: "SON"})
        for label, seasonal_group in group.groupby(season):
            rows.extend(metric_rows(seasonal_group, horizon, "season", label, models))
    trials = pd.DataFrame(evaluation["folds"]).assign(scope="fold")
    columns = ["horizon_hours", "fold", "scope", "train_rows", "train_first_issue",
               "train_last_issue", "train_last_label"]
    result = pd.DataFrame(rows).merge(trials[columns], how="left", on=columns[:3])
    return result.assign(seed=config["seed"], gap_hours=config["gap_hours"], target=config["target"])


def run(config_path: Path, power_path: Path | None, weather_path: Path | None) -> None:
    from .cli import write_json

    started = perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configurations(config)
    directory = Path(config["data_dir"])
    power = input_path(directory, "power", power_path)
    weather = input_path(directory, "weather", weather_path)
    hourly, audit = prepare(power, weather, config["target"], config["minimum_hour_fraction"])
    processed = Path(config["processed_dir"])
    write_json(processed / "audit.json", audit)
    hourly.to_parquet(processed / "hourly.parquet")
    with threadpool_limits(limits=1):
        predictions, evaluation = evaluate(hourly, config)
    results = Path(config["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    summary = metrics(predictions, evaluation, config)
    predictions.to_csv(results / "test_predictions.csv", index=False)
    summary.to_csv(results / "metrics.csv", index=False)
    metadata = {
        **evaluation, "config": config, "source": CATALOG, "license": LICENSE,
        "audit": audit, "target_fields_kw": TARGETS[config["target"]],
        "python": python_version(), "versions": {name: version(name) for name in
            ("numpy", "pandas", "scikit-learn", "pyarrow", "matplotlib", "threadpoolctl", "pyyaml")},
        "completed_utc": datetime.now(timezone.utc).isoformat(), "seconds": perf_counter() - started,
    }
    write_json(results / "run_metadata.json", metadata)
    print(f"{evaluation['scheme']}: {len(predictions)} out-of-sample forecasts in {results}")
