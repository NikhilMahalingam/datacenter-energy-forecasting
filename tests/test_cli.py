import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from esif_forecasting.cli import main
from test_features import hourly_fixture


def test_cli_from_manual_files_to_figures(tmp_path, monkeypatch):
    fixture = hourly_fixture(384).rename_axis("ts").reset_index()
    power = fixture.rename(columns={"target_kw": "cooling_kw"}).assign(hvac_kw=3, pump_kw=1)
    weather = fixture.assign(outside_air_temp=fixture.temperature_c * 1.8 + 32)
    power_path = tmp_path / "power.csv.zip"
    weather_path = tmp_path / "weather.parquet"
    power[["ts", "cooling_kw", "it_power_kw", "hvac_kw", "pump_kw"]].to_csv(power_path, index=False)
    weather[["ts", "outside_air_temp", "outside_air_humidity"]].to_parquet(weather_path)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
    results = tmp_path / "results"
    processed = tmp_path / "processed"
    config.update(results_dir=str(results), processed_dir=str(processed))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    paths = ["--power", str(power_path), "--weather", str(weather_path)]
    monkeypatch.setattr(sys, "argv", ["esif", "run", "--config", str(cfg), *paths])
    main()
    metadata = json.loads((results / "run_metadata.json").read_text())
    predictions = pd.read_csv(results / "test_predictions.csv")
    assert metadata["audit"]["power"]["rows"] == len(fixture)
    assert metadata["target_fields_kw"] == ["cooling_kw"]
    assert set(predictions.horizon_hours) == {1, 24}
    assert (processed / "hourly.parquet").is_file()
    assert (results / "metrics.csv").is_file()
    for name in ("forecast_1h", "forecast_24h", "mae_skill", "weather_ablation",
                 "auxiliary_power_vs_temperature"):
        assert (results / "figures" / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert set(metadata["figure_intervals"]) == {"1", "24"}


def test_earliest_contiguous_trace():
    from esif_forecasting.plots import representative_period

    index = pd.date_range("2024-01-01", periods=400, freq="h").delete(10)
    frame = pd.DataFrame({"target_start": index, "fold": "a"})
    segment = representative_period(frame)
    assert len(segment) == 168
    assert segment.target_start.iloc[0] == index[10]
