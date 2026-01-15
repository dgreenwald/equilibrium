# Equilibrium

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Equilibrium** is a high-performance dynamic general-equilibrium solver built on JAX. It provides a rule-based framework for defining and solving economic models with automatic code generation, JAX-accelerated computations, and support for both steady-state and linear approximation solutions.

## Features

- **Rule-based model specification**: Define economic models through intuitive rules for variables, expectations, transitions, and optimality conditions
- **Automatic code generation**: Converts model rules into optimized Python functions with JAX compilation
- **High-performance computing**: Leverages JAX for automatic differentiation, JIT compilation, and GPU acceleration
- **Multiple solution methods**: Supports Klein and AIM methods for linearization around steady states
- **Deterministic and stochastic simulations**: Compute impulse response functions and simulate model dynamics
- **Multi-regime scenarios**: Define complex policy experiments with `DetSpec` for piecewise-constant parameter regimes
- **Flexible parameter calibration**: Built-in support for parameter calibration during steady-state solving
- **Type-safe operations**: Uses NamedTuple-based state management for reliable computations
- **Results I/O**: Save and load model results in multiple formats (NPZ, JSON, CSV)
- **Plotting utilities**: Built-in functions for visualizing IRFs and deterministic paths
- **Centralized configuration**: Pydantic-based settings with environment variable support

## Installation

### Using pip

```bash
pip install equilibrium
```

### Development Installation

For the latest development version or to contribute:

```bash
git clone https://github.com/dgreenwald/equilibrium.git
cd equilibrium

# Option 1: Install with pip (recommended)
pip install -e .[dev]

# Option 2: Install from requirements files
pip install -r requirements-dev.txt
pip install -e .

# Option 3: Using conda for environment management
conda env create -f environment.yml
conda activate equilibrium-env
```

The project uses a `src` layout: package sources live under `src/equilibrium/`, while the pytest suite resides in `tests/`; install in editable mode (`pip install -e .[dev]`) or export `PYTHONPATH=src` before running scripts or tests.

### Dependencies

Equilibrium requires Python 3.10+ and the following packages:
- JAX and JAXlib (≥0.4)
- NumPy (≥1.26)
- SciPy
- NetworkX
- Jinja2
- Pydantic (≥2) and pydantic-settings (≥2)
- Matplotlib (for plotting)

## Creating a New Project

The easiest way to start a new equilibrium project is with the scaffolding utility:

```bash
# Create a new project with a working RBC example
equilibrium init my_project

# Navigate to the project and run immediately
cd my_project
python main.py
```

This creates a minimal project structure with a working example:
- **main.py**: Main execution script (solve steady state, compute IRFs, plot results)
- **model.py**: Model specification with RBC example (matches Quick Start below)
- **parameters.py**: Parameter values and steady-state guesses
- **constants.py**: Plotting configuration (variables, titles, styling)
- **.env**: Environment variable configuration (optional)

All files are well-documented with inline comments explaining how to customize for your own model. The RBC example runs immediately so you can see the full workflow in action.

### Environment Variables

The scaffolded project includes a `.env` file you can edit to override settings without changing code.
You can also export variables in your shell before running:

```bash
# Override data/output locations
export EQUILIBRIUM_PATHS__DATA_DIR=/custom/path

# Enable logging
export EQUILIBRIUM_LOGGING__ENABLED=true
export EQUILIBRIUM_LOGGING__LEVEL=INFO
```

## Modular Model Blocks

Equilibrium is designed to be modular: you can assemble models from reusable
`ModelBlock` components. A model block bundles rule definitions and variables so
you can compose larger models without duplicating equations. This is especially
helpful for swapping policy regimes or reusing standard production, preference,
and financial frictions across projects.

```python
from equilibrium.model import Model, ModelBlock, model_block

@model_block
def production_block(mod: Model) -> ModelBlock:
    mod.rules["intermediate"] += [
        ("y", "Z * (K ** alp)"),
        ("w", "(1 - alp) * y / L"),
    ]
    return ModelBlock(mod)

model = Model()
model.add_block(production_block(model))
```

## Quick Start

Here's a simple example showing how to define and solve a basic RBC (Real Business Cycle) model:

```python
import numpy as np
from equilibrium import Model

# Create a new model
model = Model()

# Set model parameters
model.params.update({
    'alp': 0.6,     # Capital share
    'bet': 0.95,    # Discount factor
    'delta': 0.1,   # Depreciation rate
    'gam': 2.0,     # Risk aversion
    'Z_bar': 0.5,   # Baseline productivity
})

# Set initial guesses for steady state
model.steady_guess.update({
    'I': 0.5,                    # Investment
    'log_K': np.log(6.0),        # Log capital
})

# Define model equations through rules
model.rules['intermediate'] += [
    ('K_new', 'I + (1.0 - delta) * K'),                 # Capital law of motion
    ('Z', 'Z_bar + Z_til'),                             # Total productivity
    ('fk', 'alp * Z * (K ** (alp - 1.0))'),            # Marginal product of capital
    ('y', 'Z * (K ** alp)'),                            # Output
    ('c', 'y - I'),                                     # Consumption
    ('uc', 'c ** (-gam)'),                              # Marginal utility
    ('K', 'np.exp(log_K)'),                             # Capital level
]

# Expectation equations (Euler equation)
model.rules['expectations'] += [
    ('E_Om_K', 'bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))'),
]

# State transition equations
model.rules['transition'] += [
    ('log_K', 'np.log(K_new)'),
]

# Optimality conditions
model.rules['optimality'] += [
    ('I', 'E_Om_K - 1.0'),
]

# Add exogenous process for productivity shock
model.add_exog('Z_til', pers=0.95, vol=0.1)

# Finalize the model (compiles rules into functions)
model.finalize()

# Solve for steady state
model.solve_steady(calibrate=False)

# Linearize around steady state
model.linearize()

# Compute impulse response functions
model.compute_linear_irfs(Nt_irf=20)
irfs = model.linear_mod.irf  # Access the computed IRFs

# Simulate the model
simulation = model.simulate_linear(Nt=100)
```

## Model Components

### Rule Types

Equilibrium organizes model equations into several rule categories:

- **`intermediate`**: Definitions of intermediate variables and identities
- **`expectations`**: Forward-looking equations (use `_NEXT` suffix for next period variables)
- **`transition`**: State evolution equations
- **`optimality`**: First-order conditions and equilibrium conditions
- **`calibration`**: Equations used for parameter calibration

### Variable Classifications

The solver automatically classifies variables into:

- **`u`**: Unknown/endogenous variables to be solved
- **`x`**: State variables that evolve over time
- **`z`**: Exogenous shock processes
- **`params`**: Model parameters

## API Reference

### Core Classes

#### `Model`

The main class for defining and solving DSGE models.

**Key Methods:**

- `finalize()`: Compiles model rules into executable functions
- `solve_steady(calibrate=False)`: Solves for steady-state values
- `linearize()`: Linearizes model around steady state
- `simulate_linear(Nt, s_init=None, shocks=None)`: Simulates linearized model
- `compute_linear_irfs(Nt_irf)`: Computes impulse response functions (stored in `model.linear_mod.irf`)
- `add_exog(var_name, pers=0.0, vol=0.0)`: Adds exogenous AR(1) process

## Advanced Usage

### Parameter Calibration

Enable automatic parameter calibration during steady-state solving:

```python
# Add calibration equations
model.rules['calibration'] += [
    ('bet', 'K - 6.0'),  # Calibrate discount factor to match capital target
]

# Solve with calibration enabled
model.solve_steady(calibrate=True)
```

### Working with Model Variants

Create model variants with different parameters:

```python
# Create a variant with different parameter
params_new = {'bet': model.params['bet'] + 0.01}
model_variant = model.update_copy(params=params_new)
model_variant.solve_steady(calibrate=False)
model_variant.linearize()
```

### Deterministic Simulations

Use `DetSpec` to define policy experiments with one or multiple parameter regimes:

```python
from equilibrium.solvers.det_spec import DetSpec

# Create a scenario specification
spec = DetSpec()

# Add regime 0 with baseline parameters and a shock
spec.add_regime(0, preset_par_regime={"tau": 0.3})
spec.add_shock(0, "z_tfp", per=0, val=0.01)  # TFP shock at period 0

# Add regime 1 with different parameters, starting at period 20
spec.add_regime(1, preset_par_regime={"tau": 0.35}, time_regime=20)

# Build exogenous paths for simulation
z_path = spec.build_exog_paths(model, Nt=100, regime=0)
```

### Plotting Results

Visualize simulation results and IRFs:

```python
from equilibrium import plot_deterministic_results
from equilibrium.plot import plot_model_irfs

# Plot deterministic results
plot_deterministic_results(
    results=[result1, result2],
    include_list=["c", "k", "y"],
    plot_dir="./plots",
    result_names=["Baseline", "Policy Change"],
)

# Plot model IRFs with automatic save paths
plot_model_irfs(
    model=model,
    shock="z_tfp",
    Nt=40,
    plot_dir=None,  # defaults to settings.paths.plot_dir / "irfs" / model.label
    include_list=["c", "k", "y"],
)
```

### Saving and Loading Results

Read steady-state values saved with the model label:

```python
from equilibrium import read_steady_value, read_steady_values

# Read a single steady-state value by name
steady_c = read_steady_value("c", label="my_model")

# Read multiple values at once (returns a dict)
steady_vals = read_steady_values(["c", "k", "y"], label="my_model")
```

## Performance Tips

- **Use JAX arrays**: Work with JAX numpy arrays for best performance
- **JIT compilation**: The solver automatically JIT-compiles functions for speed
- **GPU acceleration**: JAX can automatically use GPUs when available
- **Batch operations**: Use vectorized operations over manual loops
- **Function bundle reuse**: Model copies via `update_copy()` share compiled functions

## Configuration

Equilibrium uses a centralized configuration system based on Pydantic:

```python
from equilibrium.settings import get_settings

settings = get_settings()

# Access configured paths
print(settings.paths.data_dir)   # ~/.local/share/EQUILIBRIUM/
print(settings.paths.save_dir)   # data_dir/cache
print(settings.paths.plot_dir)   # data_dir/plots
print(settings.paths.log_dir)    # data_dir/logs
```

### Environment Variables

Override settings via environment variables:

```bash
# Override data directory
export EQUILIBRIUM_PATHS__DATA_DIR=/custom/path

# Enable logging
export EQUILIBRIUM_LOGGING__ENABLED=true
export EQUILIBRIUM_LOGGING__LEVEL=DEBUG
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes with proper tests
4. Follow the existing code style (Black formatting, type hints)
5. Submit a pull request

### Development Setup

```bash
# Clone and set up development environment
git clone https://github.com/dgreenwald/equilibrium.git
cd equilibrium

# Install with pip (recommended)
pip install -e .[dev]

# Or install from requirements files
pip install -r requirements-dev.txt
pip install -e .

# Or use conda
conda env create -f environment.yml
conda activate equilibrium-env

# Set up pre-commit hooks (recommended)
pre-commit install

# Run tests
pytest
# Or run a specific test file
pytest tests/test_deterministic.py
# Or run directly with python
python tests/test_deterministic.py
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to ensure code quality. The hooks automatically run on each commit to check and format your code.

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run

# Update hooks to the latest versions
pre-commit autoupdate
```

The pre-commit hooks include:
- **Black**: Code formatting (88-character limit)
- **Ruff**: Linting and import sorting (runs repo-wide to mirror CI)
- **Trailing whitespace removal**
- **End-of-file fixer**
- **YAML/TOML validation**

### Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting (88-character limit)
- Include type hints for function signatures
- Follow NumPy-style docstrings
- Use relative imports within the package
- Pre-commit hooks will automatically format and check your code

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Equilibrium in your research, please cite:

```bibtex
@software{equilibrium,
  title={Equilibrium: Dynamic General-Equilibrium Solver in JAX},
  author={Daniel L. Greenwald},
  url={https://github.com/dgreenwald/equilibrium},
  year={2026}
}
```

## Support

For questions, bug reports, or feature requests, please open an issue on [GitHub](https://github.com/dgreenwald/equilibrium/issues).
