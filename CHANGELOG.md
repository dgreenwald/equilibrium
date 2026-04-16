# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-16

### Added
- **Bayesian MCMC Estimation** (`equilibrium.estimation` subpackage): Full posterior estimation pipeline for DSGE models
  - `Prior`: Flexible prior distribution specification (Beta, Gamma, Normal, Uniform, etc.)
  - `RWMC`: Random Walk Metropolis-Hastings sampler with adaptive jump scaling, parameter blocking, mode-finding, and Hessian-based proposal covariance
  - `StateSpaceModel` / `build_state_space()`: Assembles the Kalman-filter state-space representation from a linearized model and a list of observable variable names
  - `log_likelihood()` / `log_likelihood_ssm()`: Gaussian log-likelihood evaluation via the Kalman filter
  - `EstimParam`: Typed specification for an estimated parameter (prior, bounds, initial value)
  - `estimate()`: High-level entry point — validates inputs, finds the posterior mode, computes the Hessian-based proposal, runs `n_chains` MCMC chains, and saves results
  - `EstimationResult`: Container for mode, Hessian, chains, and metadata
  - `load_estimation()`: Reload a saved estimation from disk
- **`RunItem`**: New `RunItem` dataclass for bundling a model specification with experiment specifications, used to group multi-spec runs
- **`PathResult.to_dataframe()`**: Convert simulation path results to a pandas DataFrame (requires `pandas` optional dependency)
- **Linear sequence save/load**: `solve_sequence_linear()` now saves results by default; `load_linear_sequence_result()` loads from the `sequences_linear/` subdirectory; `plot_deterministic_results()` supports `result_kind="linear_sequence"` for label-based loading
- **Derived parameters**: `Model` supports derived parameter definitions that are computed from other parameters
- **`initial_guess_from_label`** parameter on `solve_steady()` to seed the solver from a previously saved steady state
- **Secant method** support in calibration scalar solvers
- **Warm start** option for calibration to initialize from a prior solution
- **`calibrate_initial`** argument in calibration to control whether the initial parameter vector is used as a starting point
- **`hlines` / `vlines`** options in `plot_paths()` for adding horizontal and vertical reference lines
- **Plot group style controls**: `style_level` parameter and fallback resolution for plot group styles
- **`result_kind`** included in auto-generated plot subdirectory names to avoid collisions between result types
- **`write_latex_property_list()`** promoted to public API
- **`.env` file discovery** now walks parent directories, so projects do not need `.env` at the exact working directory

### Changed
- Plotting code reorganized: path preparation split into `plot/preparation.py` for reuse outside the main plot functions
- `debt_block` updated to support detrending
- MCMC estimation performance: Kalman filter inner loop, numerical helpers (`robust_cholesky`, `bound_transform`), and MCMC proposal step all optimized

### Fixed
- **`opt.root` with `hybr`/`lm` methods** was passing `maxiter` instead of `maxfev` to MINPACK, causing `OptimizeWarning` on every vector root calibration call
- Multi-regime exogenous state handoff now correctly preserves shock persistence across regime boundaries
- `log_transform` was incorrectly wrapping optimality and calibration rule RHS expressions
- Calibration bounds enforcement and spurious convergence detection improved
- Custom y-limits (`ylims`) in plotting now applied correctly
- `hlines`/`vlines` ordering in `plot_paths()` fixed

## [0.2.0] - 2026-02-07

### Added
- **Calibrated Parameter I/O**: New functions for reading and saving calibrated parameter values
  - `read_calibrated_param()` and `read_calibrated_params()` for loading parameter values by label and regime
  - `save_calibrated_params()` for persisting calibration results
- **PathResult Indexing**: Added indexing support to `PathResult` for easier access to simulation data
- **Block Suffix Control**: New `suffix_before` parameter in `Model.add_block()` to control suffix placement
  - Allows inserting suffix before placeholder terms like `_AGENT`, `_ATYPE`, `_INSTRUMENT`
  - Example: `C_AGENT` with `suffix="_firm"` and `suffix_before=["_AGENT"]` becomes `C_firm_AGENT` instead of `C_AGENT_firm`
- **Block Variable Transformation**: Support for applying variable transformations (suffix, rename) to `ModelBlock` instances
- **Calibration Path Transforms**: Series transforms can now be applied to paths before evaluating distance from calibration targets

### Changed
- **BREAKING**: Refactored `calibrate()` function API with typed input classes
  - Replaced generic `param_to_model` function parameter with specific typed classes
  - New classes: `RegimeParam`, `ModelParam`, `ShockParam` for clearer parameter specification
  - Provides better type safety and more explicit calibration configuration
- Improved `plot_paths()` to print variable values for easier inspection
- Suppressed unnecessary console output during calibration for cleaner logs
- Refactored internal IRF dict construction for improved code maintainability
- Streamlined `SequenceResult` splicing implementation
- Reorganized `model.py` file structure for better modularity

### Fixed
- Fixed suffix application behavior in block transformation edge cases
- Various linting and code quality improvements

## [0.1.0] - 2026-01-28

### Added
- **Dynare Export**: New `Model.to_dynare()` method and `io.dynare` module for exporting models to Dynare .mod file format with comprehensive test coverage
- **Data Overlay on Plots**: New `overlay_data` parameter in `plot_deterministic_results()` for overlaying empirical data or alternative model outputs on simulation plots
  - Support for dict and array overlay formats
  - Customizable styling via `overlay_kwargs`, `overlay_style`, and `overlay_color` parameters
  - Helper function `overlay_to_result()` for advanced overlay workflows
- **Joint Derivative Computation**: New `DerivativeResult` container class with convenient slicing and indexing by variable name or argnum
  - `Model.compute_derivatives()` method for efficient joint derivative calculation
  - `Model.steady_state_derivatives()` method for computing derivatives at steady state
  - Option to compute derivatives with respect to parameters (`include_params`)
- **JAX Pytree Registration**: State NamedTuple now registered as JAX pytree for efficient tree operations
  - Enables `jax.tree.map()` and other pytree operations on State objects
  - Reduces compilation overhead for state construction (200+ primitives → ~20)
- **Lazy Derivative Evaluation**: Optional deferred derivative computation in `FunctionBundle` for improved performance
- **Persistent Function Caching**: Disk-based caching of compiled JAX functions to speed up repeated model operations
- **Calibration Flag Tracking**: New `_calibrated` flag on Model to track whether steady state was solved with calibration
  - Automatic detection and re-solving when calibration flag mismatch detected in `solve_sequence()`
  - Warning messages when cached steady state doesn't match requested calibration setting
- **Enhanced Verbose Iteration Logging**: Improved logging for steady state solution iterations with file retention management
- **Diagnostic Information**: New diagnostic output for steady state solution convergence analysis

### Changed
- **Generalized Newton Solver**: Internal Newton solver can now be used more broadly with scipy as configurable fallback
- **Settings Initialization**: Settings now initialized early in `__init__.py` to configure JAX before any JAX code imports
- **Iteration Logging**: Verbose iteration logging now only logs first attempt and manages file retention automatically

### Performance
- **Vectorized State Construction**: Optimized `array_to_state()` function using `jnp.split()` instead of individual scalar extractions
  - Reduces trace graph size significantly
  - Decreases compilation count for state operations
- **Function Bundle Sharing**: Models created with `update_copy()` share function bundles, resulting in zero additional compilations for second linearization
- **Compilation Optimization**: Various improvements reducing total compilation count in standard workflows

### Fixed
- Removed duplicate `tabulate` dependency from `requirements-dev.txt`
- Clarified comment for `derived_vars` definition (was incorrectly marked as TODO)

### Documentation
- Added `CLAUDE.md` with comprehensive project guidance and development patterns
- Added `docs/performance-analysis.md` with detailed compilation analysis
- Added `docs/REMAINING_OPTIMIZATIONS.md` documenting future optimization opportunities

## [0.0.2] - 2026-01-20

### Added
- New smoke test script for verifying TestPyPI installations (`scripts/smoke_test_testpypi.sh`)
- Additional Makefile targets for development workflow

### Changed
- Updated README.md with improved documentation and formatting
- Modified pyproject.toml configuration to improve package setup

### Fixed
- Restored `black` code formatter to development dependencies
- Updated requirements files to ensure consistent dependency versions

## [0.0.1] - Initial Release

### Added
- Initial release of Equilibrium: JAX-based dynamic general-equilibrium solver
- Core model specification and code generation framework
- Rule-based economic model definition system
- Automatic differentiation and JIT compilation with JAX
- Model block system for reusable economic components
- Deterministic and stochastic simulation capabilities
- Steady-state solver and linearization tools
- IRF (Impulse Response Function) analysis
- Plotting utilities for model results
- Project scaffolding utility (`equilibrium init`)
- Comprehensive test suite
- Documentation and examples

[Unreleased]: https://github.com/dgreenwald/equilibrium/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dgreenwald/equilibrium/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dgreenwald/equilibrium/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/dgreenwald/equilibrium/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/dgreenwald/equilibrium/releases/tag/v0.0.1
