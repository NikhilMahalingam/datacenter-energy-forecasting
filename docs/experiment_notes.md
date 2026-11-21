# Forecasting mechanical/cooling-plant auxiliary power at ESIF

**Empirical results blocked, November 21, 2026.** The OEDI catalog and license were accessible. Neither measured series could be retrieved. Coverage, sampling cadence, timestamp convention, seasonal behavior, forecast errors, and the effect of adding weather remain unmeasured. Automated tests use deterministic fixtures in temporary directories. There is no manuscript results paragraph or empirical figure set yet. [Access attempts and meter definitions](data_audit.md)

## Research question and study population

How accurately can an operator forecast hourly mean auxiliary electrical power one hour and one day ahead using completed measurements? Weather history could add information about fans and freeze protection beyond recent power demand. Persistence could also perform well when operation is stable. The sample is one HPC facility. The design measures forecast accuracy and associations. It cannot establish a change in electricity use, water use, or emissions.

## Issue time, target, and information set

Let `T` be the clock time when the operator issues a forecast. The most recent observed power covers `[T−1h, T)`. A forecast at horizon h predicts mean kW in `[T + (h−1) hours, T + h hours)`, where h is the integer horizon in hours. For example, a forecast issued at 09:00 predicts the 09:00–10:00 hour at h=1 and the following day's 08:00–09:00 hour at h=24. The target field defaults to `cooling_kw` until meter accounting can be reviewed. The combined-meter sensitivity option and sampling rules are in the [data audit](data_audit.md).

Persistence uses the latest completed hour for both horizons. Previous-day seasonal naive uses the observation from exactly 24 hours before the target period. At h=24 its predictions coincide with persistence by construction.

The two fitted models use target and IT power at lags 0, 1, 2, 3, 6 and 24 hours, plus trailing 6-hour and 24-hour means ending at issue time. Sine/cosine pairs encode the issue hour and weekday. The weather ablation adds identically constructed historical outdoor temperature and relative humidity features. Each horizon uses complete cases across all feature sets, both baselines and the label. A missing weather record can therefore reduce the sample even for the power-only comparison. Neither PUE nor energy reuse enters the predictors. Observed history becomes available as each test origin arrives. Actual target-hour weather is unavailable to all models.

## Evaluation protocol

Choose folds once from the overlapping complete hourly measurements. A calendar year qualifies for final testing when the series spans that year, at least 95% of its hours have all four signals, and at least 7,008 complete hourly observations precede it. Use the latest qualifying year and fit only on earlier history. Otherwise, when measured overlap spans at least 270 days, use expanding training windows and full calendar-month test windows beginning four months after the first observed month's start. With shorter overlap, reserve the last quarter of the elapsed time span for testing. A short holdout may represent just one seasonal regime. Empty test folds are recorded. Insufficient training or validation history stops the run.

A target interval must lie wholly within its test window. Training labels must end at least 24 hours before the earliest retained test issue time under the default embargo. The inner validation set is the last 20% of eligible training origins by row count, with the same embargo on inner training labels. Select the candidate with lowest validation MAE, taking the first candidate on ties, and refit on the whole outer training sample. Each feature set uses exactly the same rows. Fold models stay fixed during their test window.

| Estimator | Candidates and fitting |
| --- | --- |
| Ridge | Standardize features on the current training partition. Compare alpha 1 and 10. |
| Histogram gradient boosting | Compare 7 and 15 maximum leaf nodes. Use 100 iterations, learning rate 0.06, minimum 20 samples per leaf, no early stopping, seed 42. |

Report MAE and RMSE in kW. Compute pooled errors directly from all held-out predictions. Skill is one minus model MAE divided by persistence MAE on identical rows. Positive values indicate lower error, negative values higher error. Skill is undefined at zero persistence MAE. R² is undefined for a constant target or a single observation. Seasonal scores group by meteorological season of the target start in the source clock. Overlapping lag windows and successive targets can produce correlated errors.

## Figure captions for the planned run

| Filename in `results/figures/` | Caption |
| --- | --- |
| `forecast_1h.png`, `forecast_24h.png` | Observed and forecast hourly mean auxiliary power during the first uninterrupted seven-day test segment within a fold. If no such segment exists, use the earliest longest contiguous segment. Both weather-enabled ML models appear in the lower panel. At 24 hours the two baseline lines overlap. |
| `mae_skill.png` | MAE skill versus same-horizon persistence. Circles show individual test folds and diamonds pool all test predictions. A zero denominator leaves the corresponding symbol absent. |
| `weather_ablation.png` | Pooled MAE of each fitted model with power and time features and with observed outdoor weather added, evaluated on matched origins within each horizon. |
| `auxiliary_power_vs_temperature.png` | Concurrent outdoor temperature and auxiliary power for valid paired hours across the full record. Diamonds indicate medians in 5 °C bins. Counts, medians and means are saved in `temperature_bins.csv`. Cold-weather changes can be consistent with trace-heater operation, but the available category includes other equipment. |

## Completing the manuscript analysis

Run the [reproduction commands](../README.md), verify the timestamp and cadence audit, compare PUE residuals and the measured meter traces, and record the final target choice before comparing scores. Use the pooled metrics for the headline table and the fold and season rows to describe variation. Inspect bin counts and temporal coverage before interpreting the temperature curve. Report weather ablations and negative skill wherever observed. The study can support a single-site forecasting benchmark. Generalization to dedicated AI training facilities requires external validation.
