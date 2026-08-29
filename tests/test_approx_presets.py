"""Tests for preset helper factories."""

import pytest

from equilibrium.approx import (
    VALID_FUNCAPPROX_NAMES,
    make_funcapprox,
    make_smolyak_chebyshev,
    make_smolyak_hierarchical_hat,
    make_smolyak_modified_hat,
    make_tensor_chebyshev,
    make_tensor_modified_hat,
    make_tensor_uniform_hat,
)
from equilibrium.approx.levels import SmolyakInteriorLevels


def test_make_smolyak_chebyshev_basic():
    func = make_smolyak_chebyshev(
        dimension=2,
        max_levels=(3, 3),
        max_total_level=4,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    assert func.scheme.dimension == 2
    assert func.scheme.bases[0].basis_type == "chebyshev"
    assert func.get_grid_points().shape[1] == 2


def test_smolyak_level_override():
    func = make_smolyak_chebyshev(
        dimension=1,
        max_levels=2,
        max_total_level=2,
        lb=[-1.0],
        ub=[1.0],
        level=3,
    )
    assert func.scheme.index.levels[0].level == 3


def test_make_smolyak_hierarchical_hat():
    func = make_smolyak_hierarchical_hat(
        dimension=1,
        max_levels=3,
        max_total_level=3,
        lb=[0.0],
        ub=[1.0],
    )
    assert func.scheme.bases[0].basis_type.startswith("hierarchical")


def test_make_smolyak_modified_hat():
    func = make_smolyak_modified_hat(
        dimension=1,
        max_levels=3,
        max_total_level=3,
        lb=[0.0],
        ub=[1.0],
    )
    assert func.scheme.bases[0].basis_type == "modified_hierarchical_hat"
    assert isinstance(func.scheme.index.levels[0], SmolyakInteriorLevels)


def test_make_tensor_chebyshev():
    func = make_tensor_chebyshev(
        dimension=2,
        n_points=(3, 4),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    assert func.scheme.dimension == 2
    assert func.scheme.bases[0].basis_type == "chebyshev"


def test_make_tensor_uniform_hat():
    func = make_tensor_uniform_hat(
        dimension=1,
        n_points=3,
        lb=[0.0],
        ub=[1.0],
    )
    assert func.scheme.bases[0].basis_type == "uniform_hat"


def test_make_tensor_modified_hat():
    func = make_tensor_modified_hat(
        dimension=1,
        n_points=3,
        lb=[0.0],
        ub=[1.0],
    )
    assert func.scheme.bases[0].basis_type == "modified_uniform_hat"


def test_make_funcapprox_dispatch():
    func = make_funcapprox(
        " Smolyak_Chebyshev ",
        dimension=1,
        max_levels=2,
        max_total_level=2,
        lb=[-1.0],
        ub=[1.0],
    )
    assert func.scheme.bases[0].basis_type == "chebyshev"


def test_make_funcapprox_invalid_name():
    with pytest.raises(ValueError, match="smolyak_chebyshev"):
        make_funcapprox(
            "not_a_real_preset",
            dimension=1,
            max_levels=2,
            max_total_level=2,
            lb=[-1.0],
            ub=[1.0],
        )


def test_valid_funcapprox_names_contains_smolyak_chebyshev():
    assert "smolyak_chebyshev" in VALID_FUNCAPPROX_NAMES
    assert "smolyak_modified_hat" in VALID_FUNCAPPROX_NAMES
    assert "tensor_modified_hat" in VALID_FUNCAPPROX_NAMES
