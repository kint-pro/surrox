# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Problem layer: declarative problem definition with variables, objectives, constraints, scenarios
- Surrogate layer: ensemble training with XGBoost, LightGBM, Gaussian Process, TabICL
- Optimizer layer: pymoo-based multi-objective optimization with auto algorithm selection
- Analysis layer: SHAP, PDP/ICE, feature importance, what-if, scenario comparison
- Conformal prediction for uncertainty quantification
- Facade API with `run()` and `run_scenarios()` entry points
