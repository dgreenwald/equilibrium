"""Multidimensional index built from 1D levels."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..levels import Levels


class IndexBlock(NamedTuple):
    """Container for a single level combination."""

    levels: tuple[int, ...]
    basis_indices: tuple[NDArray[np.int64], ...]
    grid_indices: tuple[NDArray[np.int64], ...]


class Index:
    """Combine 1D levels into a sparse multidimensional index."""

    def __init__(
        self,
        levels: Sequence[Levels],
        *,
        max_total_level: int | None,
        max_levels: Sequence[int] | int | None = None,
    ) -> None:
        if not levels:
            raise ValueError("levels must be non-empty")

        self._levels = tuple(levels)
        self._dimension = len(self._levels)

        self._max_levels = self._normalize_max_levels(max_levels)
        self._max_total_level = self._validate_total_level(max_total_level)
        self._level_vectors = tuple(self._generate_level_vectors())

        # Build grid_ix_ and basis_ix_ arrays
        self._grid_ix, self._basis_ix, self._size = self._build_index_arrays()

    @property
    def levels(self) -> tuple[Levels, ...]:
        return self._levels

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_levels(self) -> tuple[int, ...]:
        return self._max_levels

    @property
    def max_total_level(self) -> int | None:
        return self._max_total_level

    @property
    def level_vectors(self) -> tuple[tuple[int, ...], ...]:
        """All level tuples satisfying the configured constraints."""
        return self._level_vectors

    @property
    def size(self) -> int:
        """Total number of points in the sparse grid."""
        return self._size

    def iter_level_vectors(self) -> Iterator[tuple[int, ...]]:
        """Iterate over all valid level tuples."""
        yield from self._level_vectors

    def iter_blocks(self) -> Iterator[IndexBlock]:
        """Yield per-level blocks with basis and grid indices."""
        for level_tuple in self._level_vectors:
            basis_indices = tuple(
                level.basis_indices(level_value)
                for level, level_value in zip(self._levels, level_tuple)
            )
            grid_indices = tuple(
                level.grid_indices(level_value)
                for level, level_value in zip(self._levels, level_tuple)
            )
            yield IndexBlock(level_tuple, basis_indices, grid_indices)

    def _normalize_max_levels(
        self, max_levels: Sequence[int] | int | None
    ) -> tuple[int, ...]:
        if max_levels is None:
            values = tuple(level.level for level in self._levels)
        elif isinstance(max_levels, int):
            if max_levels < 0:
                raise ValueError("max_levels must be >= 0")
            values = tuple(max_levels for _ in range(self._dimension))
        else:
            if len(max_levels) != self._dimension:
                raise ValueError("max_levels sequence must match number of dimensions")
            values = tuple(int(value) for value in max_levels)
            if any(value < 0 for value in values):
                raise ValueError("max_levels must be >= 0")

        for idx, (value, level) in enumerate(zip(values, self._levels)):
            if value > level.level:
                raise ValueError(
                    f"max_levels[{idx}]={value} exceeds level limit {level.level}"
                )

        return values

    def _validate_total_level(self, max_total_level: int | None) -> int | None:
        if max_total_level is None:
            return None
        if max_total_level < 0:
            raise ValueError("max_total_level must be >= 0")
        return max_total_level

    def _generate_level_vectors(self) -> Iterator[tuple[int, ...]]:
        """Recursively generate admissible level combinations.

        The recursion avoids exploring branches whose partial sums already
        exceed the maximum total level, mirroring the C++ implementation.
        """

        partial: list[int] = []

        def recurse(dim_idx: int, level_sum: int) -> Iterator[tuple[int, ...]]:
            if dim_idx == self._dimension:
                yield tuple(partial)
                return

            max_level = self._max_levels[dim_idx]
            upper = max_level
            if self._max_total_level is not None:
                remaining = self._max_total_level - level_sum
                if remaining < 0:
                    return
                upper = min(upper, remaining)

            for level_value in range(upper + 1):
                partial.append(level_value)
                yield from recurse(dim_idx + 1, level_sum + level_value)
                partial.pop()

        yield from recurse(0, 0)

    def _build_index_arrays(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
        """Build flattened grid_ix and basis_ix arrays.

        Returns:
            Tuple of (grid_ix, basis_ix, total_size)
        """
        grid_ix_list = []
        basis_ix_list = []
        total_size = 0

        for block in self.iter_blocks():
            # Compute tensor product grid indices for this block
            # Each block has n_points = product of len(grid_indices[d]) for all d
            block_indices = block.grid_indices
            block_basis_indices = block.basis_indices

            # Generate all combinations using meshgrid
            grids = np.meshgrid(*block_indices, indexing="ij")
            bases = np.meshgrid(*block_basis_indices, indexing="ij")

            # Flatten and stack
            # Shape: (n_points_in_block, dimension)
            block_grid_flat = np.column_stack([g.ravel() for g in grids])
            block_basis_flat = np.column_stack([b.ravel() for b in bases])

            n_points_block = block_grid_flat.shape[0]

            # Append to lists
            grid_ix_list.append(block_grid_flat)
            basis_ix_list.append(block_basis_flat)
            total_size += n_points_block

        # Concatenate all blocks
        if grid_ix_list:
            grid_ix_2d = np.vstack(grid_ix_list)  # Shape: (total_size, dimension)
            basis_ix_2d = np.vstack(basis_ix_list)  # Shape: (total_size, dimension)

            # Flatten to 1D with layout [dim * idx + dim_idx]
            grid_ix = grid_ix_2d.ravel()  # Ravels in C-order (row-major)
            basis_ix = basis_ix_2d.ravel()
        else:
            grid_ix = np.array([], dtype=np.int64)
            basis_ix = np.array([], dtype=np.int64)
            total_size = 0

        return grid_ix, basis_ix, total_size

    @property
    def grid_ix(self) -> NDArray[np.int64]:
        """Flat array of grid indices for sparse grid construction."""
        return self._grid_ix

    @property
    def basis_ix(self) -> NDArray[np.int64]:
        """Flat array of basis indices for sparse basis construction."""
        return self._basis_ix

    def __repr__(self) -> str:
        return (
            f"Index(dimension={self._dimension}, max_total_level={self._max_total_level},"
            f" max_levels={self._max_levels})"
        )
