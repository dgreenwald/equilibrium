# Repository Guidelines

Contributors help maintain the high-performance JAX pipeline in Equilibrium; the points below keep changes aligned and reproducible.

## Project Structure & Module Organization
- `src/equilibrium/` contains the installable package: `core/` manages rule transforms and code generation, `model/` exposes the `Model` API plus linearization, `solvers/` hosts deterministic/Newton/perturbation routines, `templates/` stores Jinja snippets, and `utils/` provides typed containers and helpers.
- `tests/` holds the pytest suite; reuse fixtures from the RBC examples and keep large artifacts isolated (see `tests/UX_benchmark.npy`).
- `scripts/relativize_imports.py` is a maintenance helper for path hygiene; touch it only when reorganizing imports.

## Environment Setup
- Require Python 3.10+. Install tooling with `pip install -e .[dev]` or `conda env create -f environment.yml` then `conda activate equilibrium-env`.
- JAX selects accelerators at runtime; export `XLA_PYTHON_CLIENT_PREALLOCATE=false` during GPU debug sessions to avoid memory grabs.
- Keep experiment configuration in shell exports or `.env` files—never hard-code local paths in package modules.

## Build, Test, and Development Commands
- `pytest` (or `python -m pytest`) runs the full suite with verbose output configured via `pyproject.toml`; run after `pip install -e .[dev]` or export `PYTHONPATH=src`.
- `ruff check equilibrium tests` enforces lint rules; run `ruff check --fix` when the automatic edits are safe.
- `black .` formats the tree at 88 columns, matching CI expectations.
- `mypy equilibrium` validates type hints across the NamedTuple-heavy APIs.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; let `black` manage layout and trailing commas in multi-line literals.
- Use `snake_case` for functions, variables, and rule keys; reserve `PascalCase` for classes and `UPPER_SNAKE` for constants.
- Align generated function names with template placeholders (e.g., `*_bundle`, `*_rule`) so auto-generated code stays traceable.
- Type-hint public APIs and new utilities; prefer `NamedTuple` or `typing.Protocol` to describe solver state.

## Testing Guidelines
- Mirror module names in `tests/test_<module>.py`; seed stochastic routines so JAX tracing stays deterministic.
- Add regression fixtures under `tests/data/` (create if needed) and document provenance in a README beside the data.
- Cover new solver paths with both shape and dtype assertions to catch JAX tracing regressions early.

## Commit & Pull Request Guidelines
- Commit subjects follow an imperative voice and stay concise (`Fix ruff linting errors`, `Update README...`); keep bodies focused on context and side-effects.
- Reference related issues in commit bodies or PR descriptions, and note API or performance implications explicitly.
- PRs should summarise behaviour changes, list validation commands run locally, and attach screenshots for doc-facing work. Request review only once CI is green and docs/tests reflect the change.
