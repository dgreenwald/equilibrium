from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax

logger = logging.getLogger(__name__)


@dataclass
class FunctionBundle:
    """
    Wrap a function f and provide consistently jitted transforms.

    Can differentiate w.r.t. single or multiple positional arguments.
    - If argnums is an int, stores jacobians/hessians for that single argument
    - If argnums is a list, stores jacobians/hessians for all specified arguments

    The primal function is always stored once (independent of argnums).
    """

    f: Callable
    argnums: Union[int, List[int]] = 0
    has_aux: bool = False

    # Configure what's static for JIT; keep stable to avoid recompiles.
    static_argnums: Optional[Tuple[int, ...]] = None
    static_argnames: Optional[Tuple[str, ...]] = None

    # Extra jit kwargs (e.g., donate_argnums, inline)
    jit_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Compiled callables (filled in __post_init__)
    f_jit: Callable = field(init=False)

    # Dictionaries of compiled callables keyed by argnum
    grad_jit: Dict[int, Callable] = field(init=False, default_factory=dict)
    value_and_grad_jit: Dict[int, Callable] = field(init=False, default_factory=dict)
    jacobian_jit: Dict[int, Callable] = field(init=False, default_factory=dict)
    jacobian_fwd_jit: Dict[int, Callable] = field(init=False, default_factory=dict)
    jacobian_rev_jit: Dict[int, Callable] = field(init=False, default_factory=dict)
    hessian_jit: Dict[int, Callable] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Primal function (independent of argnums)
        self.f_jit = self._jit(self.f)

        # Normalize argnums to list
        argnums_list = [self.argnums] if isinstance(self.argnums, int) else self.argnums

        # Create derivatives for each argnum
        for argnum in argnums_list:
            # value_and_grad works with or without aux
            self.value_and_grad_jit[argnum] = self._jit(
                jax.value_and_grad(self.f, argnums=argnum, has_aux=self.has_aux)
            )

            # Plain grad (only if no aux)
            if not self.has_aux:
                self.grad_jit[argnum] = self._jit(jax.grad(self.f, argnums=argnum))

            # Cache forward and reverse mode jacobians
            self.jacobian_fwd_jit[argnum] = self._jit(
                jax.jacfwd(self.f, argnums=argnum)
            )
            self.jacobian_rev_jit[argnum] = self._jit(
                jax.jacrev(self.f, argnums=argnum)
            )

            # Default jacobian (alias to rev for backward compatibility)
            self.jacobian_jit[argnum] = self.jacobian_rev_jit[argnum]

            # Hessian wrt argnum
            self.hessian_jit[argnum] = self._jit(jax.hessian(self.f, argnums=argnum))

    def _jit(self, fn: Callable) -> Callable:
        return jax.jit(
            fn,
            static_argnums=self.static_argnums,
            static_argnames=self.static_argnames,
            **self.jit_kwargs,
        )

    # Optional: provide convenience methods for backward compatibility
    def jacobian_fwd(self, argnum: Optional[int] = None) -> Callable:
        """Get forward-mode jacobian for specified argnum (or default if single argnum)."""
        if argnum is None:
            # For backward compatibility, return the first (or only) jacobian
            if isinstance(self.argnums, int):
                argnum = self.argnums
            else:
                argnum = self.argnums[0]
        return self.jacobian_fwd_jit[argnum]

    def jacobian_rev(self, argnum: Optional[int] = None) -> Callable:
        """Get reverse-mode jacobian for specified argnum (or default if single argnum)."""
        if argnum is None:
            # For backward compatibility, return the first (or only) jacobian
            if isinstance(self.argnums, int):
                argnum = self.argnums
            else:
                argnum = self.argnums[0]
        return self.jacobian_rev_jit[argnum]


# Example usage
if __name__ == "__main__":
    logger.info(
        "FunctionBundle example code has been moved to tests/test_jax_function_bundle.py"
    )
    logger.info("Run: pytest tests/test_jax_function_bundle.py")
