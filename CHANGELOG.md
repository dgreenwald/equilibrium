# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/yourusername/equilibrium/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/equilibrium/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/yourusername/equilibrium/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/yourusername/equilibrium/releases/tag/v0.0.1
