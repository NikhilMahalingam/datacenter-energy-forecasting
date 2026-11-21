import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .download import BASE, FILES

POWER = ["it_power_kw", "cooling_kw", "hvac_kw", "pump_kw"]
WEATHER = ["outside_air_temp", "outside_air_humidity"]
TARGETS = {"mechanical": POWER[1:], "cooling_meter": ["cooling_kw"]}


def input_path(directory: Path, kind: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for name in FILES[kind]:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {kind} in {directory}. Run download or pass --{kind} PATH.")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".zip", ".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Expected Parquet, CSV or zipped CSV: {path}")


def validate(raw: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = set(["ts", *required]) - set(raw)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Input file contains no records")
    if pd.api.types.is_numeric_dtype(raw["ts"]):
        raise ValueError("Numeric timestamps require a documented epoch unit")
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], errors="raise", format="mixed")
    if frame["ts"].isna().any() or not pd.api.types.is_datetime64_any_dtype(frame["ts"]):
        raise ValueError("Expected nonmissing timestamps with a consistent timezone")
    for name in required:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"Expected numeric values in {name}")
    return frame


def clean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # Retain exact repeats once. Discard timestamps with conflicting records.
    frame = frame.drop_duplicates().drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    frame = frame[columns].replace([np.inf, -np.inf], np.nan)
    for column in columns:
        if column.endswith("_kw"):
            frame[column] = frame[column].where(frame[column] >= 0)
    if "outside_air_humidity" in columns:
        frame["outside_air_humidity"] = frame["outside_air_humidity"].where(
            frame["outside_air_humidity"].between(0, 100))
        frame["outside_air_temp"] = frame["outside_air_temp"].where(
            frame["outside_air_temp"].between(-100, 150))
    return frame


def provenance(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    sidecar = path.with_name(path.name + ".json")
    source = {"catalog_file_url": BASE + path.name} if any(
        path.name in names for names in FILES.values()) else {}
    if sidecar.is_file():
        source.update(json.loads(sidecar.read_text(encoding="utf-8")))
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest, **source}


def audit(raw: pd.DataFrame, frame: pd.DataFrame, path: Path) -> dict:
    unique = pd.DatetimeIndex(frame["ts"].unique()).sort_values()
    intervals = unique.to_series().diff().dropna()
    numeric = frame.select_dtypes(include="number")
    return {
        **provenance(path), "rows": len(frame), "dtypes": raw.dtypes.astype(str).to_dict(),
        "start": str(unique.min()), "end": str(unique.max()), "timezone": str(unique.tz),
        "coverage_years": (unique.max() - unique.min()) / pd.Timedelta(days=365.25),
        "monotonic_input": frame["ts"].is_monotonic_increasing,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_timestamp_rows": int(frame["ts"].duplicated(keep=False).sum()),
        "conflicting_timestamps": int(frame.drop_duplicates()["ts"].duplicated().sum()),
        "missing": frame.isna().sum().to_dict(),
        "infinite": np.isinf(numeric).sum().to_dict(),
        "negative": (numeric < 0).sum().to_dict(),
        "outside_bounds": {
            key: int((frame[key].notna() & ~frame[key].between(*bounds)).sum())
            for key, bounds in {"outside_air_temp": (-100, 150),
                                "outside_air_humidity": (0, 100)}.items() if key in frame
        },
        "top_intervals": {str(k): int(v) for k, v in intervals.value_counts().head(10).items()},
        "longest_interval": str(intervals.max()),
    }


def hourly(frame: pd.DataFrame, minimum_fraction: float) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No usable rows after resolving duplicates")
    intervals = frame.index.to_series().diff().dropna()
    if intervals.empty:
        raise ValueError("At least two distinct timestamps are required to infer cadence")
    cadence = intervals.value_counts().idxmax()
    expected = max(1, round(pd.Timedelta(hours=1) / cadence))
    minimum = max(1, int(np.ceil(expected * minimum_fraction)))
    groups = frame.resample("h", label="right", closed="left")
    return groups.mean().where(groups.count() >= minimum)


def pue_accounting(power: pd.DataFrame) -> dict:
    if not {"pue", "plug_and_light_kw"}.issubset(power):
        return {"status": "PUE or plug-and-light field absent"}
    data = power[[*POWER, "pue", "plug_and_light_kw"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data["it_power_kw"] > 0]
    if data.empty:
        return {"status": "No complete rows with positive IT power"}
    recorded = data["pue"]
    overhead = data[[*POWER[1:], "plug_and_light_kw"]].sum(axis=1)
    residuals = {
        "including_pump": (1 + overhead / data["it_power_kw"] - recorded).abs(),
        "excluding_pump": (1 + (overhead - data["pump_kw"]) / data["it_power_kw"] - recorded).abs(),
    }
    return {"rows": len(data), "absolute_pue_residual": {
        key: {"mean": float(err.mean()), "median": float(err.median()),
              "p95": float(err.quantile(0.95))}
        for key, err in residuals.items()
    }}


def prepare(power_path: Path, weather_path: Path, target: str,
            minimum_fraction: float) -> tuple[pd.DataFrame, dict]:
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_hour_fraction must be in (0, 1]")
    tables, reports = {}, {}
    for kind, path, columns in (("power", power_path, POWER), ("weather", weather_path, WEATHER)):
        raw = read_table(path)
        validated = validate(raw, columns)
        reports[kind] = audit(raw, validated, path)
        if kind == "power":
            reports["pue_accounting"] = pue_accounting(validated)
        tables[kind] = clean(validated, columns)
    power, weather = tables["power"], tables["weather"]
    if str(power.index.tz) != str(weather.index.tz):
        raise ValueError("Power and weather timestamps must use the same timezone")
    reports["exact_raw_timestamp_matches"] = len(power.index.intersection(weather.index))
    power, weather = hourly(power, minimum_fraction), hourly(weather, minimum_fraction)
    frame = power.join(weather)
    frame["target_kw"] = frame[TARGETS[target]].sum(axis=1, min_count=len(TARGETS[target]))
    frame["temperature_c"] = (frame["outside_air_temp"] - 32) * (5 / 9)
    frame = frame.drop(columns="outside_air_temp")
    by_period_start = frame.set_axis(frame.index - pd.Timedelta(hours=1))
    monthly = by_period_start.resample("MS")
    counts = monthly.count().assign(grid_hours=monthly.size())
    complete = by_period_start[["target_kw", "it_power_kw", "temperature_c",
                                "outside_air_humidity"]].notna().all(axis=1)
    counts["complete"] = complete.resample("MS").sum()
    edges = complete.ne(complete.shift()).cumsum()
    gaps = (~complete).groupby(edges).sum()
    reports["hourly"] = {
        "target": target, "minimum_hour_fraction": minimum_fraction, "grid_hours": len(frame),
        "missing": frame.isna().sum().to_dict(),
        "first_period_start": str(by_period_start.index.min()),
        "last_period_end": str(frame.index.max()),
        "longest_incomplete_stretch_hours": int(gaps.max()),
        "monthly": {str(date.date()): values for date, values in counts.to_dict("index").items()},
    }
    return frame, reports
