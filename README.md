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

See [data audit](docs/data_audit.md) for source attribution, reported access errors, and target selection.
