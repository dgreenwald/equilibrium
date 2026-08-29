"""Package-level tests for the in-tree approximation API."""

import importlib
import inspect
import pkgutil
import sys

import equilibrium.approx as approx

EXPECTED_PUBLIC_API = {
    "Basis1d",
    "ChebyshevBasis1d",
    "ChebyshevLobattoGrid1d",
    "Function",
    "Grid1d",
    "HierarchicalHatBasis1d",
    "HierarchicalHatBasisInterior1d",
    "Index",
    "IndexBlock",
    "JaxApproximationData",
    "Levels",
    "ModifiedHierarchicalHatBasis1d",
    "ModifiedUniformHatBasis1d",
    "Scheme",
    "SmolyakInteriorLevels",
    "SmolyakLevels",
    "TensorProductLevels",
    "UniformGrid1d",
    "UniformGridWithBoundary1d",
    "UniformHatBasis1d",
    "UniformHatBasisInterior1d",
    "VALID_FUNCAPPROX_NAMES",
    "create_approximation",
    "evaluate_bases_jax",
    "evaluate_jax",
    "make_funcapprox",
    "make_jax_data",
    "make_smolyak_chebyshev",
    "make_smolyak_hat",
    "make_smolyak_hierarchical_hat",
    "make_smolyak_modified_hat",
    "make_tensor_chebyshev",
    "make_tensor_hat",
    "make_tensor_modified_hat",
    "make_tensor_uniform_hat",
}


def test_public_api_matches_documented_port():
    assert set(approx.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(approx, name) for name in EXPECTED_PUBLIC_API)


def test_benchmark_package_is_not_exported():
    assert "benchmark" not in approx.__all__
    assert not hasattr(approx, "benchmark")


def test_all_modules_import_without_external_funcapprox():
    module_names = [
        module.name
        for module in pkgutil.walk_packages(approx.__path__, f"{approx.__name__}.")
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        assert "from funcapprox" not in source
        assert "import funcapprox" not in source

    assert "funcapprox" not in sys.modules
    assert not any(name.startswith("funcapprox.") for name in sys.modules)
