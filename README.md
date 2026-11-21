# ESIF auxiliary-power forecasting

Direct 1-hour and 24-hour forecasts of mean mechanical/cooling-plant auxiliary power at the National Laboratory of the Rockies ESIF high-performance computing facility.

## Setup

Use Python 3.11 or later. Create a virtual environment with `python -m venv .venv`, activate it, and run:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m esif_forecasting.cli download
python -m esif_forecasting.cli inspect
```

The [OEDI dataset catalog](https://data.openei.org/submissions/8616) and [NLR catalog](https://data.nlr.gov/submissions/300) each list ESIF DC Power Metrics and Outside Weather Station files in Parquet and zipped CSV formats. If automatic retrieval fails, open either catalog, download one file for each series, and keep the original names under `data/raw/`. For renamed files or another directory, supply both paths explicitly:

```sh
python -m esif_forecasting.cli inspect --power path/to/power.parquet --weather path/to/weather.csv.zip
```

For the tested Python 3.11 dependency versions, use `python -m pip install -c requirements-tested.txt -e ".[dev]"`. Run commands from the repository root with the virtual environment active.

## Forecasts and outputs

```sh
python -m esif_forecasting.cli baseline
python -m esif_forecasting.cli run --config configs/default.yaml
```

Both commands accept the same explicit `--power` and `--weather` paths as `inspect`. The standalone baseline writes its eligible naive forecasts to `results/baseline_predictions.csv`. Use the baselines in the full run for comparisons with ML because the full run restricts every model to the same eligible origins within a horizon.

| Output of the full run | Contents |
| --- | --- |
| `data/processed/audit.json`, `hourly.parquet` | Source hashes, schema and quality checks, monthly counts, PUE residuals, and the hourly table indexed by completed hour end. |
| `results/test_predictions.csv` | One row per horizon, fold and forecast issue time. Includes target interval start/end, observed kW and all six predictions. |
| `results/metrics.csv` | Fold, season and pooled MAE, RMSE, R² and same-sample persistence skill. Undefined scores have empty fields. |
| `results/run_metadata.json` | Feature names, configuration, source audit, versions, runtime, fold and tuning details, and trace intervals. |
| `results/figures/` | Two forecast traces, skill and weather comparisons, and an outdoor temperature diagnostic with a companion bin summary. |

**Data access currently blocks empirical results.** The configured target is the outdoor cooling equipment meter, which includes freeze protection. Review the audit before choosing the wider three-meter target. See the [data audit](docs/data_audit.md), [experimental methods and figure captions](docs/experiment_notes.md), and [related work](docs/related_work.md).
