from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .features import HOUR

POWER_LABEL = "Mechanical/cooling-plant auxiliary power (kW)"
DISPLAY = {
    "persistence": "Persistence", "seasonal_naive": "Previous day",
    "ridge__power": "Ridge · power", "ridge__power_weather": "Ridge · power + weather",
    "hist_gradient_boosting__power": "Boosting · power",
    "hist_gradient_boosting__power_weather": "Boosting · power + weather",
}


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def representative_period(test: pd.DataFrame) -> pd.DataFrame:
    test = test.sort_values("target_start")
    breaks = test.target_start.diff().ne(HOUR) | test.fold.ne(test.fold.shift())
    stretches = list(test.groupby(breaks.cumsum(), sort=False))
    for _, group in stretches:
        if len(group) >= 168:
            return group.iloc[:168]
    return max(stretches, key=lambda item: len(item[1]))[1]


def forecast_trace(test: pd.DataFrame, horizon: int, directory: Path) -> dict:
    segment = representative_period(test)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, sharey=True, layout="constrained")
    for axis, models in zip(axes, (("persistence", "seasonal_naive"),
                                   ("ridge__power_weather", "hist_gradient_boosting__power_weather"))):
        axis.plot(segment.target_start, segment.actual_kw, color="0.15", label="Observed", lw=1.5)
        for model, color, style in zip(models, ("#0072B2", "#D55E00"), ("--", ":")):
            axis.plot(segment.target_start, segment[model], color=color,
                      linestyle=style, label=DISPLAY[model], lw=1.4)
        axis.legend(fontsize=8, ncols=3, loc="upper left")
    axes[0].set_title(f"{horizon}-hour-ahead forecasts")
    axes[-1].set_xlabel("Target hour start (source clock)")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.supylabel(POWER_LABEL, fontsize=10)
    save(fig, directory / f"forecast_{horizon}h.png")
    return {"start": str(segment.target_start.iloc[0]), "end": str(segment.target_end.iloc[-1]),
            "hours": len(segment), "fold": segment.fold.iloc[0]}


def skill_plot(metrics: pd.DataFrame, directory: Path) -> None:
    groups = list(metrics.groupby("horizon_hours"))
    fig, axes = plt.subplots(1, len(groups), figsize=(8, 3.5), sharey=True,
                             squeeze=False, layout="constrained")
    for ax, (horizon, frame) in zip(axes[0], groups):
        ax.axvline(0, color="0.6", lw=1)
        for i, column in enumerate(DISPLAY):
            model, _, feature_set = column.partition("__")
            values = frame.loc[(frame.model == model) & (frame.feature_set == (feature_set or "baseline"))]
            fold = values.loc[values.scope == "fold", "mae_skill"]
            pooled = values.loc[values.scope == "pooled", "mae_skill"]
            ax.scatter(fold, np.full(len(fold), i), color="0.55", s=18, label="Fold" if i == 0 else None)
            ax.scatter(pooled, [i], color="#0072B2", marker="D", s=24, label="Pooled" if i == 0 else None)
        ax.set_yticks(range(len(DISPLAY)), DISPLAY.values())
        ax.set_title(f"{horizon} h")
        ax.set_xlabel("MAE skill vs. persistence")
    axes[0, 0].invert_yaxis()
    axes[0, -1].legend(fontsize=8)
    save(fig, directory / "mae_skill.png")


def weather_ablation(metrics: pd.DataFrame, directory: Path) -> None:
    pooled = metrics.loc[(metrics.scope == "pooled") & (metrics.feature_set != "baseline")]
    table = pooled.pivot(index=["horizon_hours", "model"], columns="feature_set", values="mae_kw")
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    positions = np.arange(len(table))
    ax.bar(positions - 0.18, table.power, width=0.36, color="#0072B2", label="Power history + time")
    ax.bar(positions + 0.18, table.power_weather, width=0.36, color="#E69F00", label="With weather")
    labels = {"ridge": "Ridge", "hist_gradient_boosting": "Boosting"}
    ax.set_xticks(positions, [f"{h} h · {labels[m]}" for h, m in table.index])
    ax.set_ylabel("Mean absolute error (kW)")
    ax.legend()
    save(fig, directory / "weather_ablation.png")


def temperature_response(hourly: pd.DataFrame, directory: Path) -> None:
    data = hourly[["temperature_c", "target_kw"]].dropna()
    lower = 5 * np.floor(data.temperature_c / 5)
    bins = data.target_kw.groupby(lower.rename("lower_c")).agg(["count", "median", "mean"])
    bins.to_csv(directory / "temperature_bins.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4), layout="constrained")
    ax.scatter(data.temperature_c, data.target_kw, alpha=0.12, s=5, color="#0072B2",
               edgecolors="none", label="Hourly observation")
    ax.scatter(bins.index + 2.5, bins["median"], color="#D55E00", marker="D",
               s=25, label="Median in 5 °C bin")
    ax.set(xlabel="Outdoor temperature (°C)", ylabel=POWER_LABEL)
    ax.legend()
    save(fig, directory / "auxiliary_power_vs_temperature.png")


def make_figures(hourly, predictions, metrics, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    periods = {str(horizon): forecast_trace(group, horizon, directory)
               for horizon, group in predictions.groupby("horizon_hours")}
    skill_plot(metrics, directory)
    weather_ablation(metrics, directory)
    temperature_response(hourly, directory)
    return periods
