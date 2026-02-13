# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Equilibrium is a JAX-based dynamic general-equilibrium solver for economic models. The codebase emphasizes:
- High-performance numerical computing with JAX's automatic differentiation and JIT compilation
- Rule-based model specification with automatic code generation
- Functional programming patterns with immutable state management
- Modular model blocks for reusable economic components

## Development Commands

### Installation
```bash
# Development installation (recommended)
pip install -e .[dev]

# Alternative: conda environment
conda env create -f environment.yml
conda activate equilibrium-env
```

The project uses a `src` layout: package sources are in `src/equilibrium/`, tests in `tests/`. Always install in editable mode (`pip install -e .[dev]`) or set `PYTHONPATH=src` before running scripts.

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_deterministic.py

# Run tests with JAX compilation logging
JAX_LOG_COMPILES=1 python tests/test_deterministic.py

# Run individual test files directly
python tests/test_deterministic.py
python tests/test_compilation_analysis.py
```

### Code Quality
```bash
# Format code with Black (88-char limit)
black .

# Lint with Ruff
ruff check .
ruff check --fix .  # Auto-fix issues

# Pre-commit hooks (recommended)
pre-commit install
pre-commit run --all-files
```

### Project Scaffolding
```bash
# Create a new project with a working RBC example
equilibrium init my_project

cd my_project
python main.py  # Run the example immediately
```

The scaffolding utility creates a minimal, flat project structure:
```
my_project/
├── main.py              # Main execution script
├── model.py             # Model specification (RBC example)
├── parameters.py        # Parameter values and guesses
├── constants.py         # Plotting configuration
├── .env                 # Environment variable configuration (optional)
├── README.md            # Project documentation
└── .gitignore           # Git ignore patterns
```

**Features:**
- Working RBC example that runs immediately (matches README Quick Start)
- Flat structure ideal for small-to-medium projects
- Well-documented with inline comments explaining each section
- Shows calibration example (commented out but ready to use)
- Demonstrates steady state, IRFs, and deterministic experiments
- Easy to modify: replace RBC equations with your own model

**Growing your project:**
As projects evolve, you may want to:
- Add `specifications.py` for complex model variant configurations
- Create separate `plot_irfs.py`, `analyze_*.py` scripts for reusable analysis
- Add `blocks.py` for custom reusable model components
- Move to `model/` subdirectory structure for larger codebases (10+ files)

## Architecture Overview

### Code Generation Pipeline

Equilibrium's core innovation is automatic code generation from symbolic economic rules:

1. **Rule Definition**: Economic equations defined as symbolic tuples `(variable_name, expression_string)`
2. **Rule Processing** (`core/rules.py`): Dependency resolution, variable classification, topological sorting
3. **Code Generation** (`core/codegen.py`): Converts processed rules to Python functions using Jinja2 templates
4. **Module Compilation**: Dynamically compiles generated code into executable JAX functions
5. **Function Bundling** (`utils/jax_function_bundle.py`): Caches JIT-compiled functions and their derivatives

**Key insight**: Generated functions operate on `State` NamedTuples containing all model variables. The code generator automatically determines variable dependencies and generates optimal computation order.

### Variable Classification System

Variables are automatically classified into categories:

- **u**: Unknown/endogenous variables (solved by optimality conditions)
- **x**: State variables (evolve via transition equations)
- **z**: Exogenous shock processes
- **params**: Model parameters
- **E**: Expectation variables (forward-looking)

Rule categories match the solution algorithm:
- **`intermediate`**: Within-period calculations and identities
- **`expectations`**: Forward-looking equations (use `_NEXT` suffix for next-period variables)
- **`transition`**: State evolution equations
- **`optimality`**: First-order conditions that determine unknowns
- **`calibration`**: Parameter calibration equations
- **`analytical_steady`**: Closed-form steady state solutions

### State NamedTuple and Pytree Registration

The generated `State` NamedTuple is automatically registered as a JAX pytree, enabling efficient tree operations and reducing compilation overhead.

**Structure**:
- **Core variables** (u, x, z, E, params): Fundamental state that gets differentiated through
- **Derived variables** (intermediate, read_expectations): Computed from core vars via `intermediate_variables()`

**Pytree behavior**:
- Only core variables are pytree children (leaves)
- Derived variables excluded from pytree (reconstructed as NaN during unflatten)
- After tree operations, call `intermediate_variables()` to recompute derived vars

**Key functions** (in generated `inner_functions` module):
```python
# Convert between arrays and states
st = mod.inner_functions.array_to_state(arr)  # Array → State
arr = mod.inner_functions.state_to_array(st)  # State → Array (core vars only)

# Compute intermediate variables
st_full = mod.inner_functions.intermediate_variables(st)

# Tree operations (pytree-enabled)
import jax
st_scaled = jax.tree.map(lambda x: x * 2.0, st)  # Scale all core vars
st_sum = jax.tree.map(lambda x, y: x + y, st1, st2)  # Add states element-wise
```

**Performance impact**:
- Reduces trace graph size for state construction (200+ primitives → ~20)
- Enables efficient vectorized operations on states
- Maintains full backward compatibility with existing code

**Important**: Derived variables become NaN after tree operations. Always recompute:
```python
st_scaled = jax.tree.map(lambda x: x * 2.0, st)  # Core vars scaled, derived vars = NaN
st_scaled = mod.inner_functions.intermediate_variables(st_scaled)  # Recompute derived
```

### Model Block System

**Location**: `src/equilibrium/blocks/`

Model blocks are reusable components that encapsulate economic mechanisms:

- **`blocks/macro.py`**: Structural macroeconomic blocks (investment, bonds, debt pricing)
- **`blocks/symbolic.py`**: SymPy-based automatic differentiation for utility functions

Blocks are created using the `@model_block` decorator and accept a `ModelBlock` instance which they modify with rules, parameters, and flags. Blocks can have boolean flags to enable/disable features (e.g., `tv_inv_efficiency`, `include_lag`).

**Important**: When modifying or adding model blocks, understand that they use placeholder names like `AGENT`, `ATYPE`, `INSTRUMENT` which get replaced during model assembly. The symbolic preference block uses SymPy to automatically compute marginal utilities from utility function specifications.

### FunctionBundle and Compilation Optimization

**Critical for performance**: The `FunctionBundle` class (`utils/jax_function_bundle.py`) caches JIT-compiled functions and their Jacobians:

```python
bundle = FunctionBundle(
    f=my_function,
    argnums=[0, 1],  # Differentiate w.r.t. arguments 0 and 1
    static_argnums=(2,),  # Static arguments for JIT
)

# Cached compiled functions
result = bundle.f_jit(x, y, config)
jacobian = bundle.jacobian_fwd_jit[0](x, y, config)
```

**Key behavior**: When creating model variants with `model.update_copy(params=...)`, function bundles are shared between models, resulting in 0 additional compilations for the second linearization. This is verified in `tests/test_bundle_sharing.py`.

Current compilation baseline (from `tests/test_compilation_analysis.py`):
- Full workflow: 46 compilations
- First steady state: 25
- First linearization: 34
- Second linearization after `update_copy()`: 0 (perfect reuse!)

## Common Development Tasks

### Creating a New Model

```python
from equilibrium import Model

model = Model()

# Set parameters
model.params.update({'alp': 0.6, 'bet': 0.95})

# Set initial guesses
model.steady_guess.update({'I': 0.5, 'log_K': np.log(6.0)})

# Define rules by category
model.rules['intermediate'] += [
    ('K', 'np.exp(log_K)'),
    ('y', 'Z * (K ** alp)'),
]

model.rules['transition'] += [
    ('log_K', 'np.log(K_new)'),
]

model.rules['optimality'] += [
    ('I', 'E_Om_K - 1.0'),
]

# Add exogenous processes
model.add_exog('Z_til', pers=0.95, vol=0.1)

# CRITICAL: Must finalize before solving
model.finalize()

# Solve and linearize
model.solve_steady(calibrate=False)
model.linearize()
```

### Working with Generated Code

Debug generated code by examining the output:

```python
# During development, check generated code in debug directory
# Default: ~/.local/share/EQUILIBRIUM/debug/

from equilibrium.settings import get_settings
settings = get_settings()
print(settings.paths.debug_dir)
```

Generated functions follow consistent patterns:
- Use `st` for current state, `st_new` for next period
- All functions take `(state, params)` as arguments
- Return either tuples of values or updated State objects via `_upd()`

### Adding Model Blocks

When adding blocks to `blocks/macro.py` or creating new block modules:

1. Use the `@model_block` decorator
2. Accept `block` as first parameter (type: `ModelBlock`)
3. Modify `block.rules`, `block.params`, `block.steady_guess`, `block.flags`
4. Return the block at the end
5. Use placeholder suffixes (`_AGENT`, `_ATYPE`, `_INSTRUMENT`) for substitution

Example pattern:
```python
@model_block
def my_block(block, *, enable_feature: bool = False) -> ModelBlock:
    block.rules["intermediate"] += [
        ("var_AGENT", "expression with param_ATYPE"),
    ]

    if enable_feature:
        block.rules["optimality"] += [
            ("control_AGENT", "foc_equation"),
        ]

    return block
```

### Using Model Blocks with Suffix and Rename

Model blocks support suffix and rename transformations with precise control over variable naming:

```python
from equilibrium import Model

model = Model()

# Block with placeholder variables
agent_block_rules = {
    "intermediate": {
        "C_AGENT": "wage_AGENT * hours",
        "S_AGENT": "income - C_AGENT",
    }
}

# Add with suffix before placeholders (for better naming)
model.add_block(
    rules=agent_block_rules,
    suffix="_worker",
    suffix_before=["_AGENT"],  # Insert suffix BEFORE _AGENT
    rename={"AGENT": "h"},
)

# Result: C_AGENT → C_worker_AGENT → C_worker_h (not C_AGENT_worker_h)
```

**suffix_before parameter**:
- Controls where suffix is inserted within variable names
- If variable ends with term(s) from `suffix_before + ["_NEXT"]`, suffix goes before that block
- Accepts `list[str]`, `str`, or `None` (default: no special terms besides `_NEXT`)
- Applied before `rename`, so use pre-rename placeholder names

**Common usage**:
```python
# Standard placeholders
model.add_block(
    block,
    suffix="_firm",
    suffix_before=["_AGENT", "_ATYPE", "_INSTRUMENT"],
    rename={"AGENT": "h", "ATYPE": "saver"},
)

# Without suffix_before (old behavior):
# C_AGENT → C_AGENT_firm → C_h_firm

# With suffix_before (new behavior):
# C_AGENT → C_firm_AGENT → C_firm_h
```

### Excluding Variables from Blocks

When you want most of a block but need custom implementations for specific variables:

```python
from equilibrium import Model
from equilibrium.blocks.macro import investment_block

model = Model()

# Add block but exclude K_new to define it ourselves
model.add_block(
    investment_block(),
    suffix="_firm",
    suffix_before=["_AGENT"],
    rename={"AGENT": "household"},
    exclude_vars={"K_new_firm_household"}  # Use final name after all transforms
)

# Provide custom transition equation
model.rules['transition'] += [
    ('K_new_firm_household', 'custom_capital_accumulation_formula'),
]
```

**exclude_vars parameter**:
- Specify variable names AFTER all transformations (suffix → suffix_before → rename)
- Applies to rules and exog_list, not params/steady_guess/flags
- Takes precedence over `overwrite=True`
- Accepts `set[str]`, `list[str]`, or `None` (default: no exclusion)

### Configuration and Settings

Equilibrium uses Pydantic-based centralized configuration (`settings.py`):

```python
from equilibrium.settings import get_settings

settings = get_settings()
# Access configured paths:
# - settings.paths.data_dir (default: ~/.local/share/EQUILIBRIUM/)
# - settings.paths.save_dir (data_dir/cache)
# - settings.paths.plot_dir (data_dir/plots)
# - settings.paths.debug_dir (data_dir/debug)
# - settings.paths.log_dir (data_dir/logs)
```

Override via environment variables:
```bash
export EQUILIBRIUM_PATHS__DATA_DIR=/custom/path
export EQUILIBRIUM_LOGGING__ENABLED=true
export EQUILIBRIUM_LOGGING__LEVEL=DEBUG
```

**Important**: Always use `settings.paths` or `resolve_output_path()` instead of hard-coding file paths.

### Deterministic Simulations with DetSpec

`DetSpec` (`solvers/det_spec.py`) specifies multi-regime scenarios:

```python
from equilibrium.solvers.det_spec import DetSpec

spec = DetSpec()

# Add regimes with different parameters
spec.add_regime(0, preset_par_regime={"tau": 0.3})
spec.add_regime(1, preset_par_regime={"tau": 0.35}, time_regime=20)

# Add shocks to specific regimes
spec.add_shock(0, "z_tfp", per=0, val=0.01)

# Build paths for simulation
z_path = spec.build_exog_paths(model, Nt=100, regime=0)
```

Results are stored in `DeterministicResult` and `SequenceResult` containers (`solvers/results.py`) with save/load support for NPZ, JSON, and CSV formats.

### Parameter Calibration

**Location**: `src/equilibrium/solvers/calibration.py`

Equilibrium provides a flexible calibration system for finding parameter values that match target moments. The `calibrate()` function uses typed parameter specifications for clear, type-safe calibration workflows.

#### Basic Calibration with Model Rules

For simple calibration, add calibration rules to the model:

```python
# Add calibration equations
model.rules['calibration'] += [
    ('bet', 'K - 6.0'),  # Calibrate discount factor to match capital target
]

# Solve with calibration enabled
model.solve_steady(calibrate=True)
```

#### Advanced Calibration API

For multi-regime calibration or complex scenarios, use the `calibrate()` function with typed inputs:

```python
from equilibrium import calibrate, PointTarget, RegimeParam, ModelParam, ShockParam

# Define calibration targets
targets = [
    PointTarget(var_name='K', target=6.0, regime=0),
    PointTarget(var_name='Y', target=1.5, regime=0),
    PointTarget(var_name='C', target=1.0, regime=1),
]

# Specify parameters to calibrate
params_to_calibrate = [
    RegimeParam(name='bet', regime=0, initial_guess=0.95),  # Regime-specific
    ModelParam(name='delta', initial_guess=0.1),            # Model-wide
    ShockParam(name='log_Z_til', param='VOL', initial_guess=0.01),  # Shock parameter
]

# Run calibration
result = calibrate(
    model=model,
    spec=det_spec,
    targets=targets,
    params_to_calibrate=params_to_calibrate,
    Nt=100,
    series_transforms={'log_K': SeriesTransform(log_to_level=True)},  # Optional
)

# Access results
print(f"Calibrated values: {result.calibrated_params}")
print(f"Final distance: {result.final_distance}")
```

**Parameter Types**:
- **`RegimeParam`**: Calibrate a parameter specific to a regime (e.g., tax rate in regime 0)
- **`ModelParam`**: Calibrate a model-wide parameter (e.g., depreciation rate)
- **`ShockParam`**: Calibrate shock process parameters (e.g., volatility, persistence)

**Target Types**:
- **`PointTarget`**: Match a variable to a specific value in a regime
- **`FunctionalTarget`**: Match a custom function of simulation paths

#### Saving and Loading Calibrated Parameters

Persist calibrated parameters for reuse across sessions:

```python
from equilibrium import save_calibrated_params, read_calibrated_param, read_calibrated_params

# Save calibrated parameters with a label and regime
save_calibrated_params(
    params={'bet': 0.96, 'delta': 0.08},
    label='baseline_model',
    regime=0,
)

# Read a single calibrated parameter
bet_value = read_calibrated_param('bet', label='baseline_model', regime=0)

# Read multiple calibrated parameters at once (returns dict)
params = read_calibrated_params(['bet', 'delta'], label='baseline_model', regime=0)
```

**Important**: These I/O functions are separate from `read_steady_value()` / `read_steady_values()`, which load endogenous variable values. Use `read_calibrated_param()` for parameter values, `read_steady_value()` for state/control variables.

### Overlaying Data on Plots

The `plot_deterministic_results()` function supports overlaying external data (e.g., empirical observations, alternative model outputs) on simulation plots for direct visual comparison. This is useful for model validation, calibration, and comparing simulations with real-world data.

**Basic Usage with Default Styling:**

```python
from equilibrium.plot import plot_deterministic_results

# Simulate model
model.solve_steady(calibrate=True)
model.linearize()
simulation_result = deterministic.solve(model, z_path)

# Empirical data as dict (variable name -> array)
empirical_data = {
    'consumption': np.array([1.0, 1.05, 1.08, 1.12, 1.15]),
    'output': np.array([2.0, 2.1, 2.15, 2.22, 2.3]),
}

# Plot with overlay (default: black dash-dot line)
paths = plot_deterministic_results(
    [simulation_result],
    overlay_data=empirical_data,
    overlay_name="Empirical Data",
    result_names=["Model"],
    include_list=['consumption', 'output'],
    plot_dir="output/comparison",
)
```

**Array Overlay:**

```python
# Overlay as numpy array (requires variable names)
data_array = np.column_stack([consumption_data, output_data])

paths = plot_deterministic_results(
    [simulation_result],
    overlay_data=data_array,
    overlay_var_names=['consumption', 'output'],
    overlay_name="Historical Data",
    result_names=["Model"],
)
```

**Custom Styling with overlay_kwargs:**

The `overlay_kwargs` parameter provides direct control over overlay appearance with standard matplotlib kwargs:

```python
paths = plot_deterministic_results(
    [simulation_result],
    overlay_data=empirical_data,
    overlay_name="Data",
    overlay_kwargs={
        'linestyle': ':',      # Dotted line
        'linewidth': 2.5,      # Thicker line
        'color': 'red',        # Red color
        'alpha': 0.8,          # Slight transparency
    },
)
```

**Style Presets:**

Use built-in presets for common overlay styles:

```python
# Available presets: 'dashdot', 'dashed', 'dotted', 'markers'
paths = plot_deterministic_results(
    [simulation_result],
    overlay_data=empirical_data,
    overlay_name="Data",
    overlay_style='markers',    # Scatter plot with markers
    overlay_color='navy',        # Navy blue
)
```

**Full Control with PlotSpec:**

For maximum control, use `PlotSpec` (highest priority):

```python
from equilibrium.plot import PlotSpec

overlay_spec = PlotSpec(
    group_colors={'Empirical Data': 'darkgreen'},
    group_styles={'Empirical Data': '-.'},
    marker_styles={'Empirical Data': 's'},  # Square markers
)

paths = plot_deterministic_results(
    [simulation_result],
    overlay_data=empirical_data,
    overlay_name="Empirical Data",
    overlay_spec=overlay_spec,
)
```

**Styling Priority:**

When multiple styling parameters are provided, priority is:
1. `overlay_spec` (highest - full PlotSpec control)
2. `overlay_kwargs` (direct matplotlib kwargs)
3. `overlay_style` + `overlay_color` (preset combinations)
4. Defaults (black dash-dot line if nothing specified)

**Advanced Use Cases:**

```python
# Standalone overlay (no simulation results)
paths = plot_deterministic_results(
    overlay_data=empirical_data,
    overlay_name="Historical Data",
    include_list=['consumption', 'output'],
)

# Multiple simulation results + overlay
paths = plot_deterministic_results(
    [baseline_result, alternative_result],
    overlay_data=empirical_data,
    overlay_name="Data",
    result_names=["Baseline", "Alternative"],
    overlay_style='dashed',
    overlay_color='red',
)

# With SequenceResult
spec = DetSpec(n_regimes=2, time_list=[10])
seq_result = deterministic.solve_sequence(spec, model, Nt=30)

paths = plot_deterministic_results(
    [seq_result],
    overlay_data=empirical_data,
    overlay_name="Data",
    T_max=25,  # Truncate sequence
)

# With series transforms (applied to both simulation and overlay)
paths = plot_deterministic_results(
    [result],
    overlay_data={'log_Y': np.log([1.0, 1.1, 1.2])},
    series_transforms={'log_Y': SeriesTransform(log_to_level=True)},
)
```

**Edge Cases and Behavior:**

- **Partial variables**: If overlay has only some variables, simulation-only variables show NaN for overlay
- **Extra variables**: If overlay has variables not in simulation, they appear with NaN for simulation series
- **Different lengths**: Plot truncates to minimum length across all series
- **Missing var_names**: Array overlay without `overlay_var_names` raises `ValueError`

**Helper Function:**

For advanced use cases, `overlay_to_result()` converts external data to `DeterministicResult` format:

```python
from equilibrium.plot import overlay_to_result

overlay_result = overlay_to_result(
    overlay_data=empirical_data,
    overlay_name="Empirical",
    reference_result=simulation_result,  # Align variables
    fill_missing=True,  # Fill missing vars with NaN
)

# Now manually include in plot
paths = plot_deterministic_results([simulation_result, overlay_result])
```

## JAX-Specific Guidelines

### Core Principles
- Always import `jax.numpy` as `np` in generated code and JAX-heavy modules
- Functions must be pure (no side effects) for JIT compilation
- Prefer immutable operations and functional patterns
- Use `State._replace(var=new_value)` for state updates, never mutation
- Avoid Python loops; use JAX array operations (vmap, lax.scan, etc.)

### Common Patterns
```python
# State updates (immutable)
new_state = state._replace(consumption=new_c, capital=new_k)

# Avoid this (mutable):
# state.consumption = new_c  # WRONG - State is a NamedTuple

# Array operations over loops
result = jax.vmap(func)(array)  # Good
# result = [func(x) for x in array]  # Bad for JIT
```

### Performance Considerations
- Profile with `JAX_LOG_COMPILES=1` to track compilations
- Function bundles prevent recompilation when sharing across model copies
- Consider memory layout for large arrays
- Use `@jax.jit` selectively; the codebase has configurable JIT flags

## Testing Patterns

### Test Organization
- `test_deterministic.py`: Main regression tests with compilation tracking
- `test_compilation_analysis.py`: Step-by-step compilation breakdown
- `test_bundle_sharing.py`: Verify function bundle reuse
- `test_jax_function_bundle.py`: Unit tests for FunctionBundle

### Writing Tests
- Use `np.allclose()` for numerical comparisons (JAX precision)
- Seed random number generators for reproducibility
- Compare against saved benchmark data for regression tests
- Test both steady-state and dynamic solutions

### CompilationCounter Pattern
Tests use `CompilationCounter` to track JAX compilations:
```python
from tests.utils import CompilationCounter

with CompilationCounter() as counter:
    # Operations to monitor
    model.linearize()

print(f"Compilations: {counter.count}")
```

## Code Style

- **Line length**: 88 characters (Black default)
- **Type hints**: Required for function signatures (Python 3.10+ syntax preferred)
- **Imports**: Use relative imports within package (`.core`, `.model`, not `equilibrium.core`)
- **Docstrings**: NumPy-style format with Parameters/Returns/Examples sections
- **Naming**: Descriptive variable names; follow existing conventions for generated code

## Anti-Patterns to Avoid

### JAX-Related
- Don't mix standard NumPy and JAX arrays without conversion
- Don't use side effects in JIT-compiled functions
- Don't create new FunctionBundle instances repeatedly; cache and reuse
- Don't use Python loops where JAX vectorization applies

### Model Development
- Don't skip `model.finalize()` before solving
- Don't modify parameters after finalization without re-finalizing or using `update_copy()`
- Don't create circular dependencies in rules (dependency resolution will fail)
- Don't hard-code file paths; use `settings.paths` or `resolve_output_path()`

### Code Generation
- Don't modify generated modules directly; update templates instead
- Don't hard-code variable names in templates; use the provided lists
- Don't bypass RuleProcessor dependency resolution

## Import Management

The codebase uses relative imports within the package. If you need to update imports:

```bash
python scripts/relativize_imports.py
```

## Related Documentation

- `tests/README.md`: Testing guide with compilation count baselines
- `.github/copilot-instructions.md`: Comprehensive development patterns (source of truth for advanced patterns)
- `README.md`: User-facing documentation and examples
- `src/equilibrium/migration/README.md`: Migration utilities

## Notes on Recent Changes

Recent work has focused on compilation optimization through function bundle sharing. The `@model_block` decorator system is relatively new and provides a clean API for defining reusable model components. When working with blocks, refer to existing examples in `blocks/macro.py` and `blocks/symbolic.py` for patterns.
