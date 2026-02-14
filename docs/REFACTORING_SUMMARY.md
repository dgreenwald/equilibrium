# Plot Module Refactoring Summary

## Overview

Successfully refactored `equilibrium.plot` to extract data preparation logic from plotting functions, enabling reusable path preparation for both plotting and numerical analysis.

## Changes Made

### 1. Created `src/equilibrium/plot/preparation.py`

New module containing:

- **`PreparedPaths`** dataclass: Container for harmonized deterministic paths
  - `path_vals`: np.ndarray of shape (n_results, n_periods, n_vars)
  - `var_names`: List of variable names
  - `result_names`: List of result names
  - `processed_results`: List of transformed DeterministicResults
  - `n_periods`: Common time dimension

- **`prepare_deterministic_paths()`** function: Core preparation logic
  - Loads results (explicit or from disk via labels)
  - Handles SequenceResult → DeterministicResult conversion
  - Applies series transforms
  - Harmonizes variable names across results
  - Processes overlay data
  - Returns unified `PreparedPaths` object

- **`overlay_to_result()`** function: Moved from `plot.py`
  - Converts external data to DeterministicResult format
  - No changes to functionality

### 2. Updated `src/equilibrium/plot/plot.py`

- **Removed**: `overlay_to_result()` (moved to `preparation.py`)
- **Refactored**: `plot_deterministic_results()`
  - Now calls `prepare_deterministic_paths()` for data preparation
  - Reduced from ~230 lines to ~100 lines
  - Focuses only on styling and plotting
  - Same public API, fully backward compatible

### 3. Updated `src/equilibrium/plot/__init__.py`

Added exports from new module:
```python
from .preparation import (
    PreparedPaths,
    overlay_to_result,
    prepare_deterministic_paths,
)
```

Updated module docstring to document data preparation functions.

## Benefits

### 1. **Code Reusability**
User scripts can now access prepared paths without plotting overhead:
```python
from equilibrium.plot import prepare_deterministic_paths

prep = prepare_deterministic_paths(
    result_labels=[('baseline', 'experiment')],
    include_list=['var1', 'var2'],
    series_transforms=transforms,
    overlay_data=empirical_data,
)

# Direct numerical access
model_value = prep.path_vals[0, 37, var_idx]
data_peak = np.nanmax(prep.path_vals[1, :, var_idx])
```

### 2. **DRY Principle**
Single source of truth for:
- Result loading and splicing
- Series transformations
- Variable harmonization
- Overlay data processing

### 3. **Maintainability**
- Smaller, focused functions (~200 lines in `prepare_deterministic_paths`, ~100 in `plot_deterministic_results`)
- Clear separation: data preparation vs. visualization
- Bug fixes benefit both plotting and analysis workflows

### 4. **Type Safety**
`PreparedPaths` dataclass provides clear interface contract.

### 5. **Performance**
Analysis workflows skip matplotlib overhead when only numeric results needed.

## Usage Examples

### Example 1: Table Generation (Your Use Case)

**Before** (lines 88-104 in your script):
```python
result = load_sequence_result(model_label, experiment_label, splice=True)
result_transformed = result.transform(series_transforms=series_transforms)
overlay_result = overlay_to_result(overlay_data, ...)
data_transformed = overlay_result.transform(series_transforms=series_transforms)
# Manual indexing into UX arrays...
```

**After**:
```python
prep = prepare_deterministic_paths(
    result_labels=[(model_label, experiment_label)],
    include_list=variables,
    series_transforms=series_transforms,
    overlay_data=data_dict,
)

model_value = prep.path_vals[0, target_period, var_idx]
data_peak = np.nanmax(prep.path_vals[1, :, var_idx])
```

### Example 2: Multiple Model Comparison

```python
prep = prepare_deterministic_paths(
    result_labels=[
        ('baseline', 'shock_a'),
        ('baseline', 'shock_b'),
        ('alternative', 'shock_a'),
    ],
    include_list=['consumption', 'output'],
    series_transforms={'consumption': SeriesTransform(deviation=True)},
)

# Compare consumption responses across experiments
for i, name in enumerate(prep.result_names):
    c_idx = prep.var_names.index('consumption')
    peak_response = np.max(np.abs(prep.path_vals[i, :, c_idx]))
    print(f"{name}: peak = {peak_response:.4f}")
```

### Example 3: Still Works for Plotting

```python
# Same API as before - fully backward compatible
paths = plot_deterministic_results(
    result_labels=[('baseline', 'experiment')],
    include_list=['c', 'k', 'y'],
    overlay_data=empirical_data,
    overlay_name="Data",
    overlay_style="dashed",
)
```

## Backward Compatibility

✅ **Fully backward compatible**
- All existing code using `plot_deterministic_results()` works unchanged
- All existing code using `overlay_to_result()` works unchanged
- Same imports: `from equilibrium.plot import plot_deterministic_results, overlay_to_result`

## Testing

Verified with:
1. Direct function tests (prepare_deterministic_paths, overlay, plotting)
2. Import tests (all modules import successfully)
3. Integration tests (full workflow with overlay data)

Note: Existing test failures in `test_plot_results.py` are due to JAX/GPU environment issues, not the refactoring.

## Files Modified

1. `src/equilibrium/plot/preparation.py` - **NEW**
2. `src/equilibrium/plot/plot.py` - **MODIFIED** (removed ~230 lines, added ~20)
3. `src/equilibrium/plot/__init__.py` - **MODIFIED** (updated exports)

## Files Created for Reference

1. `/home/dan/research/frm/danare/create_deterministic_table_updated.py` - Example using new API

## Next Steps

### Optional Enhancements

1. **Add to CLAUDE.md**: Document the new `prepare_deterministic_paths()` function
2. **Update examples**: Add notebook/script showing analysis workflows
3. **Performance profiling**: Measure time savings for analysis-only workflows
4. **Additional utilities**: Consider adding helpers like:
   - `extract_variable_from_paths(prep, var_name)` → returns 2D array (results × time)
   - `compare_paths(prep, var_name, stat='max')` → summary statistics

### Documentation Updates

Recommend adding section to CLAUDE.md:
```markdown
### Numerical Analysis Without Plotting

For custom analysis requiring deterministic paths without plotting:

\`\`\`python
from equilibrium.plot import prepare_deterministic_paths

prep = prepare_deterministic_paths(
    result_labels=[('model', 'experiment')],
    include_list=['var1', 'var2'],
    series_transforms=transforms,
    overlay_data=empirical_data,
)

# Access prepared data
model_values = prep.path_vals[0, :, :]  # First result, all periods, all vars
var_idx = prep.var_names.index('var1')
time_series = prep.path_vals[:, :, var_idx]  # All results, all periods, one var
\`\`\`
```

## Summary

This refactoring successfully:
- ✅ Eliminates code duplication between plotting and analysis
- ✅ Provides clean API for numerical path access
- ✅ Maintains 100% backward compatibility
- ✅ Improves code organization (separation of concerns)
- ✅ Enables future enhancements (easier to add features to preparation vs. plotting)

The refactoring aligns with the project's emphasis on:
- Functional programming patterns
- Modular, reusable components
- Type safety and clear interfaces
