# Related work

Gao used a neural network trained on operational telemetry to model data center plant performance and predict PUE. This is an established example of ML-based efficiency prediction using facility measurements. [Gao, 2014](https://research.google/pubs/machine-learning-applications-for-data-center-optimization/)

Li and colleagues used experimental measurements to compare physical, neural, and hybrid models for energy consumption prediction in an integrated heat pipe cooling system. Their work addresses cooling-system modeling with a different system boundary. [Li et al., 2024](https://doi.org/10.1016/j.enbuild.2024.114170)

The public `datacenter-pue-forecasting` code repository reports walk-forward PUE forecasts at 1-hour and 24-hour horizons using the same ESIF release. It is research code. This experiment benchmarks direct forecasts of auxiliary electrical power at a single facility, with persistence, seasonal naive, and matched weather ablations. [ESIF PUE forecasting code](https://github.com/00y300/datacenter-pue-forecasting)
