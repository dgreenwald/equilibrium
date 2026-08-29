"""Function approximation using sparse grids and basis functions."""

from .bases import (
    Basis1d,
    ChebyshevBasis1d,
    HierarchicalHatBasis1d,
    HierarchicalHatBasisInterior1d,
    ModifiedHierarchicalHatBasis1d,
    ModifiedUniformHatBasis1d,
    UniformHatBasis1d,
    UniformHatBasisInterior1d,
)
from .core import Index, IndexBlock, Scheme
from .core.function import Function
from .grids import (
    ChebyshevLobattoGrid1d,
    Grid1d,
    UniformGrid1d,
    UniformGridWithBoundary1d,
)
from .jax_eval import (
    JaxApproximationData,
    evaluate_bases_jax,
    evaluate_jax,
    make_jax_data,
)
from .levels import Levels, SmolyakInteriorLevels, SmolyakLevels, TensorProductLevels
from .presets import (
    VALID_FUNCAPPROX_NAMES,
    create_approximation,
    make_funcapprox,
    make_smolyak_chebyshev,
    make_smolyak_hat,
    make_smolyak_hierarchical_hat,
    make_smolyak_modified_hat,
    make_tensor_chebyshev,
    make_tensor_hat,
    make_tensor_modified_hat,
    make_tensor_uniform_hat,
)

__all__ = [
    # Grids
    "Grid1d",
    "ChebyshevLobattoGrid1d",
    "UniformGrid1d",
    "UniformGridWithBoundary1d",
    # Bases
    "Basis1d",
    "ChebyshevBasis1d",
    "HierarchicalHatBasis1d",
    "HierarchicalHatBasisInterior1d",
    "ModifiedHierarchicalHatBasis1d",
    "ModifiedUniformHatBasis1d",
    "UniformHatBasis1d",
    "UniformHatBasisInterior1d",
    "Levels",
    "SmolyakLevels",
    "SmolyakInteriorLevels",
    "TensorProductLevels",
    "Index",
    "IndexBlock",
    "Scheme",
    "Function",
    "make_smolyak_chebyshev",
    "make_smolyak_hierarchical_hat",
    "make_smolyak_hat",
    "make_smolyak_modified_hat",
    "make_tensor_chebyshev",
    "make_tensor_uniform_hat",
    "make_tensor_hat",
    "make_tensor_modified_hat",
    "make_funcapprox",
    "VALID_FUNCAPPROX_NAMES",
    "create_approximation",
    "JaxApproximationData",
    "make_jax_data",
    "evaluate_bases_jax",
    "evaluate_jax",
]
