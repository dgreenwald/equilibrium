"""Project scaffolding utilities."""

import sys
from pathlib import Path


def get_template_dir() -> Path:
    """Get the path to the template directory."""
    # Templates are in equilibrium/templates/project_scaffold
    current_file = Path(__file__)
    equilibrium_root = current_file.parent.parent
    template_dir = equilibrium_root / "templates" / "project_scaffold"

    if not template_dir.exists():
        raise RuntimeError(f"Template directory not found: {template_dir}")

    return template_dir


def create_from_template(template_path: Path, output_path: Path):
    """Copy a template file to output path, removing .template extension.

    Parameters
    ----------
    template_path : Path
        Path to template file
    output_path : Path
        Destination path for the file
    """
    with open(template_path, "r") as f:
        content = f.read()

    with open(output_path, "w") as f:
        f.write(content)


def scaffold_project(project_name: str):
    """Create a new equilibrium project with recommended structure.

    Parameters
    ----------
    project_name : str
        Name of the project directory to create
    """
    # Get absolute path for the project
    project_path = Path.cwd() / project_name

    # Check if directory already exists
    if project_path.exists():
        print(f"Error: Directory '{project_name}' already exists")
        sys.exit(1)

    print(f"Creating equilibrium project: {project_name}")
    print(f"Location: {project_path}")
    print()

    # Get template directory
    template_dir = get_template_dir()

    # Create project directory structure
    try:
        # Create main project directory
        project_path.mkdir(parents=True)
        print(f"✓ Created {project_name}/")

        # Create files from templates (flat structure)
        files_to_create = [
            ("main.py.template", project_path / "main.py"),
            ("model.py.template", project_path / "model.py"),
            ("parameters.py.template", project_path / "parameters.py"),
            ("constants.py.template", project_path / "constants.py"),
            (".env.template", project_path / ".env"),
        ]

        for template_name, output_path in files_to_create:
            template_path = template_dir / template_name
            if not template_path.exists():
                print(f"Warning: Template not found: {template_path}")
                continue

            create_from_template(template_path, output_path)
            print(f"✓ Created {output_path.relative_to(project_path.parent)}")

        # Create a simple README
        readme_path = project_path / "README.md"
        readme_content = f"""# {project_name}

A minimal working example using the equilibrium toolbox.

**Current model**: Simple RBC with TFP shocks and depreciation regime change (replace with your model!)

## Quick Start

```bash
# Run the example
python main.py

# Output:
# - Steady state solution (with calibrated discount factor)
# - TFP shock IRFs
# - Low depreciation regime change transition path
# - All plots saved to plots/ directory
```

Customize what runs by editing `run_list` in `main.py`:
```python
run_list = [
    "steady",              # Solve steady state
    "irfs",                # Compute IRFs
    "plot_irfs",           # Plot IRFs
    "deterministic",       # Run deterministic transition experiment
    "plot_deterministic",  # Plot deterministic results
]
```

## Project Structure

```
{project_name}/
├── main.py              # Main execution script
├── model.py             # Model specification (equations, structure)
├── parameters.py        # Parameter values and steady state guesses
├── constants.py         # Plotting configuration
├── .env                 # Environment variables (optional)
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## Customization Guide

1. **Replace the example model** in `model.py`:
   - Keep the structure, replace equations with yours
   - Follow commented examples for different rule categories
   - See CLAUDE.md in equilibrium repository for rule categories

2. **Update parameters** in `parameters.py`:
   - Replace example calibration values
   - Adjust steady state guesses

3. **Configure plotting** in `constants.py`:
   - Add your variables to `plot_vars`
   - Customize `var_titles` for readable axis labels
   - Use `series_transforms` for unit conversions

4. **Control execution** in `main.py`:
   - Modify `run_list` to skip/include steps
   - Uncomment deterministic section for regime-switching

## Example: Parameter Calibration

The template includes an active calibration example that calibrates the discount factor
to match a steady-state capital target of 2.5.

In `model.py`:
```python
mod.rules["calibration"] += [
    ("bet", "K - 2.5"),  # Calibrate bet so K = 2.5 in steady state
]
```

In `main.py`:
```python
mod.solve_steady(calibrate=True)  # Calibration is enabled
```

To disable calibration, set `calibrate=False` or comment out the calibration rules.

## Example: Deterministic Experiments

The template includes a working low depreciation experiment. Results are automatically saved
and can be loaded for plotting using labels:

```python
# Configure and solve experiment (regime change)
det_spec = DetSpec(label="low_depreciation")
det_spec.add_regime(0, preset_par_regime={{"delta": 0.05}})
res = deterministic.solve_sequence(det_spec, mod, Nt)

# Plot using labels (loads from cache)
plot_deterministic_results(
    result_labels=[("baseline", "low_depreciation")],  # (model_label, experiment_label)
    ...
)

# Or try a shock experiment
# det_spec = DetSpec(label="tfp_shock")
# det_spec.add_shock(0, "log_Z_til", 0, 0.01)
```

## Growing Your Project

As your project evolves, consider:

- **Multiple variants**: Extract configuration logic to `specifications.py` (see CLAUDE.md)
- **Custom analysis**: Create separate `plot_irfs.py`, `analyze_*.py` scripts
- **Reusable components**: Create `blocks.py` for custom model blocks
- **Large codebase**: Organize into subdirectories as needed

See `CLAUDE.md` in the equilibrium repository for detailed patterns.

## Environment Configuration

The `.env` file contains optional environment variables. All settings are commented out by default:

- **EQUILIBRIUM_PATHS__DATA_DIR**: Override default data directory
- **EQUILIBRIUM_LOGGING__ENABLED**: Enable logging (set to `true`)
- **EQUILIBRIUM_LOGGING__LEVEL**: Set log level (DEBUG, INFO, etc.)
- **JAX_LOG_COMPILES**: Enable JAX compilation logging

## Next Steps

- See [equilibrium documentation](https://github.com/dgreenwald/equilibrium)
- Explore the RBC example to understand the workflow
- Modify equations in `model.py` to build your own model
"""

        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"✓ Created {readme_path.relative_to(project_path.parent)}")

        # Create a .gitignore
        gitignore_path = project_path / ".gitignore"
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# Equilibrium outputs
*.pdf
*.png
*.npz
plots/
cache/

# OS
.DS_Store
Thumbs.db
"""

        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        print(f"✓ Created {gitignore_path.relative_to(project_path.parent)}")

        print()
        print("=" * 60)
        print("✓ Project created successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print(f"  1. cd {project_name}")
        print("  2. python main.py          # Run the RBC example")
        print("  3. Edit model.py           # Replace with your model")
        print("  4. Edit parameters.py      # Update parameter values")
        print()
        print("For more information, see README.md or visit:")
        print("  https://github.com/dgreenwald/equilibrium")
        print()

    except Exception as e:
        print(f"Error creating project: {e}")
        # Clean up partially created directory
        if project_path.exists():
            import shutil

            shutil.rmtree(project_path)
        sys.exit(1)
