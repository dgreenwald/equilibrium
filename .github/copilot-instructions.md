# Copilot Instructions for Danare

## Project Overview

Danare is a dynamic general-equilibrium solver built on JAX, designed for solving and analyzing economic models. The project emphasizes high-performance numerical computing with automatic differentiation and JIT compilation.

## General Best Practices

### Code Style and Standards
- Follow PEP 8 conventions with 88-character line limit (Black formatting)
- Use type hints throughout (Python 3.10+ syntax preferred)
- Prefer explicit imports over wildcard imports
- Use relative imports within the package (`.core`, `.model`, etc.)
- Maintain consistent docstring format with proper parameter descriptions (NumPy-style)

### JAX-Specific Guidelines
- Always use `jax.numpy` as `np` in generated code and JAX-heavy modules
- Prefer immutable operations and functional programming patterns
- Use `@jax.jit` decorators judiciously - the codebase includes configurable JIT compilation
- Be mindful of JAX's pure function requirements (no side effects)
- Use JAX's automatic differentiation (`jax.jacfwd`, `jax.grad`) for derivatives
- Use `FunctionBundle` for consistent JAX compilation and derivative caching

### Performance Considerations
- Leverage JAX's vectorization capabilities
- Avoid Python loops in favor of JAX array operations
- Use NamedTuple for structured data (see `State` class pattern)
- Consider memory layout for large array operations
- Use `FunctionBundle` to avoid recompilation overhead

## Project Structure and Conventions

### Package Organization
```
src/
└── danare/
    ├── __init__.py           # Public API exports (Model, LinearModel, plot functions, I/O)
    ├── core/                  # Core functionality
    │   ├── codegen.py        # Code generation from symbolic rules
    │   └── rules.py          # Rule processing and dependency management
    ├── model/                 # Model definitions and operations
    │   ├── model.py          # Main Model class
    │   └── linear.py         # LinearModel class
    ├── solvers/               # Numerical solvers
    │   ├── deterministic.py  # Deterministic path solver
    │   ├── det_spec.py       # Deterministic scenario specification (DetSpec)
    │   ├── newton.py         # Newton solver utilities
    │   ├── perturb.py        # Perturbation methods
    │   ├── linear.py         # Linear system solvers
    │   └── results.py        # Result containers (DeterministicResult, SequenceResult)
    ├── templates/             # Jinja2 templates for code generation
    │   ├── base.py.jinja     # Common imports and structure
    │   └── functions.py.jinja # Function generation template
    ├── io/                    # Input/output utilities
    │   ├── __init__.py       # Exports: resolve_output_path, save_results, load_results
    │   └── results.py        # Results I/O functions (npz, json, csv formats)
    ├── plot/                  # Plotting utilities
    │   ├── __init__.py
    │   └── plot.py           # plot_paths, plot_deterministic_results
    ├── utils/                 # Utility functions and data structures
    │   ├── containers.py     # MyOrderedDict and custom containers
    │   ├── utilities.py      # General utilities
    │   ├── io.py             # Legacy I/O utilities
    │   └── jax_function_bundle.py  # FunctionBundle for JAX compilation caching
    ├── settings.py            # Centralized configuration via Pydantic
    ├── logger.py              # Logging configuration with rotating handlers
    └── modspec.py             # Model feature specification (ModSpec)
```

### Key Architecture Patterns

#### Code Generation Pipeline
The project uses a sophisticated code generation system:

1. **Rule Definition**: Economic rules are defined symbolically
2. **Rule Processing** (`RuleProcessor`): Transforms rules and resolves dependencies
3. **Code Generation** (`CodeGenerator`): Converts processed rules to Python functions
4. **Template Rendering**: Uses Jinja2 templates to generate clean Python code
5. **Module Compilation**: Dynamically compiles generated code into executable modules

**When working with code generation:**
- Understand the `State` NamedTuple pattern - it's central to the architecture
- Variables are categorized as: `core_vars`, `derived_vars`, `intermediate`, etc.
- Templates use `st` (current state) and `st_new` (next period state) conventions
- Generated functions follow consistent patterns for arguments and returns

#### Variable Naming Conventions
- **Core variables**: Primary model variables (u, x, z, E, params)
- **Intermediate variables**: Computed within periods
- **Derived variables**: Additional computed values
- **Rule categories**: `transition`, `optimality`, `intermediate`, `calibration`, `analytical_steady`

#### State Management
```python
# State objects are NamedTuples with all model variables
class State(NamedTuple):
    var1: np.ndarray
    var2: np.ndarray
    # ...

# Pure functional updates using _replace
new_state = state._replace(var1=new_value)
```

#### FunctionBundle for JAX Compilation
The `FunctionBundle` class caches JIT-compiled functions and their derivatives:

```python
from danare.utils.jax_function_bundle import FunctionBundle

# Create a bundle with cached derivatives
bundle = FunctionBundle(
    f=my_function,
    argnums=[0, 1],  # Differentiate w.r.t. arguments 0 and 1
    static_argnums=(2,),  # Static arguments for JIT
)

# Use cached compiled functions
result = bundle.f_jit(x, y, config)
jacobian_x = bundle.jacobian_fwd_jit[0](x, y, config)
jacobian_y = bundle.jacobian_rev_jit[1](x, y, config)
```

### Configuration and Settings

#### Centralized Settings (settings.py)
Danare uses Pydantic for centralized configuration:

```python
from danare.settings import get_settings

settings = get_settings()
# Access paths
settings.paths.data_dir    # ~/.local/share/DANARE/
settings.paths.save_dir    # data_dir/cache
settings.paths.log_dir     # data_dir/logs
settings.paths.debug_dir   # data_dir/debug
settings.paths.plot_dir    # data_dir/plots
```

**Environment Variable Overrides:**
- `DANARE_PATHS__DATA_DIR` - Override data directory
- `DANARE_LOGGING__ENABLED=true` - Enable logging
- `DANARE_LOGGING__LEVEL=DEBUG` - Set log level

#### Logging Configuration
```python
from danare.logger import get_logger, configure_logging

# Get a logger
logger = get_logger("mymodule")  # Returns danare.mymodule logger

# Manual configuration
configure_logging()  # Uses settings from get_settings()
```

### Model Development Patterns

#### Model Definition Workflow
1. Initialize `Model` with parameters and rules
2. Add exogenous processes with `add_exog()`
3. Define rules in categories (transition, optimality, etc.)
4. Call `finalize()` to compile the model
5. Solve steady state and linearize

#### Rule Definition Best Practices
- Use clear, descriptive variable names
- Separate intermediate calculations for readability
- Follow the `(variable_name, expression)` tuple format
- Use `_NEXT` suffix for forward-looking variables
- Leverage analytical steady-state solutions when available

### Results and I/O

#### Result Containers
Use `DeterministicResult` and `SequenceResult` for storing solver outputs:

```python
from danare.solvers.results import DeterministicResult, SequenceResult

# DeterministicResult stores a single path
result = DeterministicResult(
    UX=path_array,          # Shape (Nt, N_ux)
    Z=exog_array,           # Shape (Nt, N_z)
    model_label="my_model",
    var_names=["c", "k", "y"],
    exog_names=["z_tfp"],
)

# Save and load
result.save("path/to/results.npz")
loaded = DeterministicResult.load("path/to/results.npz")

# SequenceResult for multi-regime sequences
seq_result = SequenceResult(regimes=[result1, result2], time_list=[20])
spliced = seq_result.splice(T_max=50)  # Combine into single DeterministicResult
```

#### I/O Utilities
```python
from danare.io import resolve_output_path, save_results, load_results

# Resolve path with smart defaults
path = resolve_output_path(
    result_type="irfs",
    model_label="baseline",
    timestamp=True,
)

# Save arbitrary data
save_results({"irfs": irf_array}, path, format="npz", metadata={"model": "RBC"})

# Load results
data = load_results(path)
```

### Plotting

#### Plotting Utilities
```python
from danare.plot import plot_paths, plot_deterministic_results

# Plot time-series paths
plot_paths(
    path_vals=array,           # Shape (groups, periods, variables)
    full_list=all_var_names,
    include_list=["c", "k"],
    title_str="IRFs",
    x_str="Period",
    prefix="irf",
    group_names=["Baseline", "Alternative"],
    plot_dir="./plots",
)

# Plot DeterministicResult objects directly
plot_deterministic_results(
    results=[result1, result2],
    include_list=["consumption", "output"],
    plot_dir="./plots",
    result_names=["Baseline", "Alternative"],
)
```

### Deterministic Scenario Specification

#### DetSpec for Multi-Regime Simulations
```python
from danare.solvers.det_spec import DetSpec

# Create a scenario with parameter regimes and shocks
spec = DetSpec()
spec.add_regime(0, preset_par_regime={"tau": 0.3})
spec.add_regime(1, preset_par_regime={"tau": 0.35}, time_regime=20)
spec.add_shock(0, "z_tfp", per=0, val=0.01)

# Build exogenous paths for simulation
z_path = spec.build_exog_paths(model, Nt=100, regime=0)
```

#### Testing Patterns
- Create deterministic test cases with known benchmarks
- Use `np.allclose()` for numerical comparisons
- Test both steady-state and dynamic solutions
- Include regression tests with saved benchmark data
- Seed stochastic routines for reproducible JAX tracing

## Code Generation Specifics

### Template Variables and Context
When working with Jinja2 templates:
- `funcs`: List of function dictionaries with `name`, `args`, `body`, `returns`
- `core_vars`: List of core variable names
- `derived_vars`: List of derived variable names
- `jit`: Boolean flag for JIT compilation

### Function Generation Patterns
```python
# Generated functions follow this pattern:
@jax.jit  # If jit=True
def function_name(st, params):
    # Intermediate calculations
    var1 = st.input1 + params.param1
    var2 = var1 ** 2

    # Return pattern depends on function type
    return var1, var2  # Multiple returns
    # OR
    return _upd(st, var1=var1, var2=var2)  # State update
```

### Debug and Development
- Use `debug_dir` in `CodeGenerator` to save generated code
- Set `display_source=True` to see generated code
- Generated code is formatted with Black for consistency
- Review generated modules in settings.paths.debug_dir

## Development Workflow

### Setting Up Development Environment

There are three recommended ways to set up the development environment:

#### Option 1: Using pip (Recommended for most users)
This is the fastest and most straightforward method:
```bash
# Install the package in editable mode with development dependencies
pip install -e .[dev]

# Or install from requirements files
pip install -r requirements-dev.txt
pip install -e .
```

#### Option 2: Using conda (Recommended for managing Python versions)
Use conda to create an isolated environment with all dependencies:
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate danare-env
```
The environment.yml file is cross-platform and will work on Linux, macOS, and Windows.

#### Option 3: Using conda with pip (Maximum compatibility)
Combine conda for base packages and pip for the package installation:
```bash
# Create a minimal conda environment
conda create -n danare-env python=3.10
conda activate danare-env

# Install dependencies via pip
pip install -r requirements-dev.txt
pip install -e .
```

### Running Tests
```bash
# Run all tests with pytest
pytest tests/

# Run specific test file
pytest tests/test_deterministic.py

# Run with JAX compilation logging
JAX_LOG_COMPILES=1 python tests/test_deterministic.py
```

Because the repository uses a `src` layout, run tests after `pip install -e .[dev]` or set `PYTHONPATH=src` for ad-hoc execution.

### Code Quality Tools
- **Black**: Code formatting (88-character limit)
- **Ruff**: Fast Python linting (`ruff check --fix` for auto-fixes)
- **MyPy**: Type checking

```bash
# Lint
ruff check .

# Format
black .

# Type check (when configured)
mypy danare
```

### Import Management
Use the provided script for relative imports:
```bash
python scripts/relativize_imports.py
```

## Common Anti-Patterns to Avoid

### JAX-Related
- Don't use Python loops where JAX array operations would work
- Avoid side effects in JIT-compiled functions
- Don't mix NumPy and JAX arrays unnecessarily
- Be careful with shape broadcasting in array operations
- Don't create new `FunctionBundle` instances repeatedly - cache them

### Code Generation
- Don't modify generated modules directly - update templates instead
- Avoid hardcoding variable names in templates - use the provided lists
- Don't bypass the dependency resolution system in `RuleProcessor`
- Keep template logic simple - complex logic belongs in Python

### Model Development
- Don't skip the `finalize()` step - it's required for compilation
- Avoid circular dependencies between rules
- Don't modify model parameters after `finalize()` without updating
- Use `update_copy()` for parameter changes rather than in-place modification

### Settings and Configuration
- Don't hard-code file paths - use `settings.paths` or `resolve_output_path()`
- Don't create files outside the configured directories
- Use environment variables for deployment-specific configuration

## Integration Guidelines

### Adding New Solvers
1. Create new module in `solvers/` directory
2. Follow the established interface patterns
3. Use JAX operations for performance
4. Include appropriate error handling and convergence checks
5. Return results using `DeterministicResult` or `SequenceResult`

### Extending Code Generation
1. Modify templates in `templates/` directory
2. Update `CodeGenerator` class if new template variables needed
3. Ensure generated code follows project conventions
4. Test with `display_source=True` during development

### Adding New Model Types
1. Inherit from base `Model` class or create parallel structure
2. Follow the rule-based architecture
3. Ensure compatibility with existing solvers
4. Update public API exports in `__init__.py` if appropriate

### Adding New I/O Formats
1. Add format handler in `io/results.py`
2. Update `save_results()` and `load_results()` functions
3. Follow the existing pattern for metadata handling
4. Update result container `save()` and `load()` methods

## Performance Optimization

### JAX Best Practices
- Profile with JAX's built-in tools
- Consider array memory layout for large problems
- Use `jax.vmap` for batch operations
- Leverage JAX's automatic vectorization
- Use `FunctionBundle` to cache compiled functions

### Compilation Optimization
- Compilation counts should be monitored (see `tests/test_compilation_analysis.py`)
- Function bundles enable sharing compiled functions across model copies
- Second linearization of copied models should trigger 0 new compilations
- Use `JAX_LOG_COMPILES=1` to debug compilation issues

### Code Generation Optimization
- Minimize redundant calculations in generated code
- Use intermediate variables judiciously
- Consider computational graph efficiency
- Profile generated functions separately

## Documentation Standards

### Docstring Format
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function purpose.

    Parameters
    ----------
    param1 : type
        Description of parameter.
    param2 : type
        Description of parameter.

    Returns
    -------
    return_type
        Description of return value.

    Examples
    --------
    >>> result = function_name(value1, value2)
    """
```

### Comments
- Use comments sparingly but effectively
- Explain complex mathematical transformations
- Document JAX-specific considerations
- Clarify template logic in Jinja2 files

This guide should help maintain consistency and quality while working on the Danare codebase. Always consider the economic modeling context and the performance requirements of numerical computing when making changes.
