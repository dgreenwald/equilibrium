from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax

logger = logging.getLogger(__name__)


class _LazyDerivativeDict:
    """
    A dictionary-like object that lazily creates JAX derivatives on first access.

    This avoids the overhead of creating all possible derivatives (grad, jacobian,
    hessian) upfront when only a subset will actually be used.
    """

    def __init__(
        self,
        factory: Callable[[int], Callable],
        argnums_list: List[int],
    ):
        """
        Parameters
        ----------
        factory : Callable[[int], Callable]
            A function that takes an argnum and returns the compiled derivative.
        argnums_list : List[int]
            List of valid argnums for validation.
        """
        self._factory = factory
        self._argnums_list = argnums_list
        self._cache: Dict[int, Callable] = {}

    def __getitem__(self, argnum: int) -> Callable:
        if argnum not in self._cache:
            if argnum not in self._argnums_list:
                raise KeyError(
                    f"argnum {argnum} not in configured argnums: {self._argnums_list}"
                )
            self._cache[argnum] = self._factory(argnum)
        return self._cache[argnum]

    def __contains__(self, argnum: int) -> bool:
        return argnum in self._argnums_list

    def get(self, argnum: int, default: Any = None) -> Any:
        try:
            return self[argnum]
        except KeyError:
            return default

    def keys(self):
        return self._argnums_list

    def items(self):
        """Iterate over all argnums, lazily creating derivatives as needed."""
        for argnum in self._argnums_list:
            yield argnum, self[argnum]

    def values(self):
        """Iterate over all derivatives, lazily creating them as needed."""
        for argnum in self._argnums_list:
            yield self[argnum]

    def __len__(self):
        return len(self._argnums_list)

    def __repr__(self):
        cached = list(self._cache.keys())
        return f"_LazyDerivativeDict(argnums={self._argnums_list}, cached={cached})"


@dataclass
class FunctionBundle:
    """
    Wrap a function f and provide consistently jitted transforms.

    Can differentiate w.r.t. single or multiple positional arguments.
    - If argnums is an int, stores jacobians/hessians for that single argument
    - If argnums is a list, stores jacobians/hessians for all specified arguments

    The primal function is always stored once (independent of argnums).

    Derivatives are created lazily on first access to avoid unnecessary
    compilation overhead. Only the derivatives you actually use will be compiled.
    """

    f: Callable
    argnums: Union[int, List[int]] = 0
    has_aux: bool = False

    # Configure what's static for JIT; keep stable to avoid recompiles.
    static_argnums: Optional[Tuple[int, ...]] = None
    static_argnames: Optional[Tuple[str, ...]] = None

    # Extra jit kwargs (e.g., donate_argnums, inline)
    jit_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Compiled primal function (filled in __post_init__)
    f_jit: Callable = field(init=False)

    # Internal: normalized list of argnums
    _argnums_list: List[int] = field(init=False)

    # Lazy dictionaries of compiled callables keyed by argnum
    grad_jit: _LazyDerivativeDict = field(init=False)
    value_and_grad_jit: _LazyDerivativeDict = field(init=False)
    jacobian_jit: _LazyDerivativeDict = field(init=False)
    jacobian_fwd_jit: _LazyDerivativeDict = field(init=False)
    jacobian_rev_jit: _LazyDerivativeDict = field(init=False)
    hessian_jit: _LazyDerivativeDict = field(init=False)

    def __post_init__(self):
        # Primal function is always needed, compile immediately
        self.f_jit = self._jit(self.f)

        # Normalize argnums to list
        self._argnums_list = (
            [self.argnums] if isinstance(self.argnums, int) else list(self.argnums)
        )

        # Create lazy derivative dictionaries
        self.grad_jit = _LazyDerivativeDict(
            self._make_grad, self._argnums_list
        )
        self.value_and_grad_jit = _LazyDerivativeDict(
            self._make_value_and_grad, self._argnums_list
        )
        self.jacobian_fwd_jit = _LazyDerivativeDict(
            self._make_jacobian_fwd, self._argnums_list
        )
        self.jacobian_rev_jit = _LazyDerivativeDict(
            self._make_jacobian_rev, self._argnums_list
        )
        # Default jacobian aliases to rev for backward compatibility
        self.jacobian_jit = self.jacobian_rev_jit
        self.hessian_jit = _LazyDerivativeDict(
            self._make_hessian, self._argnums_list
        )

    def _jit(self, fn: Callable) -> Callable:
        return jax.jit(
            fn,
            static_argnums=self.static_argnums,
            static_argnames=self.static_argnames,
            **self.jit_kwargs,
        )

    def _make_grad(self, argnum: int) -> Callable:
        """Create a jitted grad function for the given argnum."""
        if self.has_aux:
            raise ValueError(
                "grad_jit is not available when has_aux=True. "
                "Use value_and_grad_jit instead."
            )
        return self._jit(jax.grad(self.f, argnums=argnum))

    def _make_value_and_grad(self, argnum: int) -> Callable:
        """Create a jitted value_and_grad function for the given argnum."""
        return self._jit(
            jax.value_and_grad(self.f, argnums=argnum, has_aux=self.has_aux)
        )

    def _make_jacobian_fwd(self, argnum: int) -> Callable:
        """Create a jitted forward-mode jacobian function for the given argnum."""
        return self._jit(jax.jacfwd(self.f, argnums=argnum))

    def _make_jacobian_rev(self, argnum: int) -> Callable:
        """Create a jitted reverse-mode jacobian function for the given argnum."""
        return self._jit(jax.jacrev(self.f, argnums=argnum))

    def _make_hessian(self, argnum: int) -> Callable:
        """Create a jitted hessian function for the given argnum."""
        return self._jit(jax.hessian(self.f, argnums=argnum))

    # Convenience methods for backward compatibility
    def jacobian_fwd(self, argnum: Optional[int] = None) -> Callable:
        """Get forward-mode jacobian for specified argnum (or default if single argnum)."""
        if argnum is None:
            argnum = self._argnums_list[0]
        return self.jacobian_fwd_jit[argnum]

    def jacobian_rev(self, argnum: Optional[int] = None) -> Callable:
        """Get reverse-mode jacobian for specified argnum (or default if single argnum)."""
        if argnum is None:
            argnum = self._argnums_list[0]
        return self.jacobian_rev_jit[argnum]


# Example usage
if __name__ == "__main__":
    logger.info(
        "FunctionBundle example code has been moved to tests/test_jax_function_bundle.py"
    )
    logger.info("Run: pytest tests/test_jax_function_bundle.py")
