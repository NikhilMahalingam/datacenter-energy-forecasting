# ESIF data audit

## Provenance and availability

Struan Clark and Justin Strelka, *NLR HPC Facility Power Usage Effectiveness (PUE) Data*, OEDI, January 29, 2026. The NLR catalog dates the initial release to 2025 and gives DOI 10.7799/3015212. OEDI listed CC BY 4.0 when checked on November 21, 2026. [OEDI record](https://data.openei.org/submissions/8616) · [NLR record](https://doi.org/10.7799/3015212) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Series | Parquet filename | Zipped CSV filename |
| --- | --- | --- |
| Power | `1757103411-esif.influx.buildingData.PUE.combined.parquet` | `1757105566-esif.influx.buildingData.PUE.combined.csv.zip` |
| Weather | `1757105566-esif.influx.buildingData.outside.combined_2.parquet` | `1757105566-esif.influx.buildingData.outside.combined.csv.zip` |

All four catalog paths on `data.nrel.gov` failed DNS resolution from the execution host. The same four on `data.nlr.gov` returned HTTP 503, as did both Parquet paths on `data-stage.nlr.gov`. Follow the manual instructions in [README](../README.md) once an endpoint is available.

**Measured audit pending.** No telemetry files were acquired during these attempts. The `inspect` command writes `data/processed/audit.json` with file hashes, sizes, dtypes, source timezone, coverage, timestamp interval distributions, missingness, duplicate and validity counts, raw timestamp matches, monthly hourly completeness, and PUE accounting comparisons. Manually supplied files can have a same-name `.json` sidecar containing `url` and `retrieved_utc` to capture download provenance. Inspect the meter signals and accounting before finalizing the target.

## Target and cleaning rules

The conservative default uses `cooling_kw` alone. It covers outdoor cooling fans, freeze-protection trace heaters, and the dedicated tower filter pump. HVAC and other pumps are kept as separate fields. The catalog explicitly excludes the 2.67 kW filter pump from `pump_kw`. The optional `mechanical` target sums the three fields after the meter-accounting decision. The audit compares PUE residuals with and without the published `pump_kw` contribution. Those residuals are a consistency check with acknowledged uncertainty about the PUE denominator and other overheads.

Preserve naive timestamps if the source has no zone information. Reject missing or inconsistent timestamps. Keep exact repeated records once and remove conflicting records at the same timestamp. Mask negative kW, infinite values, humidity outside 0–100%, and outdoor temperature outside the broad −100°F to 150°F plausibility bounds. Determine each series’ nominal cadence from its modal positive interval after deduplication.

Use arithmetic means in clock-hour bins `[start, end)`, labeled by `end`. Each signal needs at least 75% of the expected sample count by default. Align power and weather on identical hour ends with zero tolerance and no filling. Keep the full power hour grid, including gaps. Convert outdoor °F to °C with `(°F − 32) × 5/9`. The target is mean kW.
