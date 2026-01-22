#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:21:30 2022

@author: dan
"""

import itertools
import logging
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from scipy.optimize import root as scipy_root
from tabulate import tabulate

from ..core.codegen import CodeGenerator
from ..core.rules import RuleProcessor
from ..model.linear import LinearModel
from ..settings import get_settings
from ..solvers.newton import root as newton_root
from ..utils import io
from ..utils.containers import MyOrderedDict, PresetDict
from ..utils.jax_function_bundle import FunctionBundle
from ..utils.utilities import initialize_if_none

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def trace_args(name, *args):
    """
    Log diagnostic information about function arguments for debugging.

    Parameters
    ----------
    name : str
        Name of the function being traced.
    *args : array_like
        Variable number of arguments to inspect.
    """
    logger.debug("%s inputs:", name)
    for i, a in enumerate(args):
        logger.debug(
            "  arg%s: shape=%s, dtype=%s, type=%s",
            i,
            getattr(a, "shape", None),
            getattr(a, "dtype", None),
            type(a),
        )


def standardize_args(*args):
    """
    Convert arguments to JAX arrays with float64 dtype.

    Parameters
    ----------
    *args : array_like
        Variable number of arguments to standardize.

    Returns
    -------
    list[jax.Array]
        List of arguments converted to JAX arrays with float64 dtype.

    Raises
    ------
    AssertionError
        If any converted argument is not a JAX Array.
    """
    std = [jnp.array(arg, dtype=jnp.float64) for arg in args]
    for i, a in enumerate(std):
        assert isinstance(a, jax.Array), f"Arg {i} is not a jax.Array: {type(a)}"
    return std


class BaseModelBlock:
    """Flexible base container for core model configuration artifacts.

    This class accepts ``rule_keys`` as a parameter for maximum flexibility.
    For most use cases, prefer using :class:`ModelBlock` which automatically
    uses the standard rule keys from :class:`Model`.
    """

    def __init__(
        self,
        *,
        flags=None,
        params=None,
        steady_guess=None,
        rules=None,
        exog_list=None,
        rule_keys=(),
    ):
        self.flags = initialize_if_none(flags, {})
        self.params = PresetDict(initialize_if_none(params, {}))
        self.steady_guess = PresetDict(initialize_if_none(steady_guess, {}))
        self.exog_list = initialize_if_none(exog_list, [])
        self.rule_keys = tuple(rule_keys)

        rules = initialize_if_none(rules, {})
        normalized = {}
        for key in self.rule_keys:
            raw = rules.get(key)
            if isinstance(raw, MyOrderedDict):
                normalized[key] = raw
            else:
                normalized[key] = MyOrderedDict(raw or {})
        self.rules = normalized

    @staticmethod
    def _validate_replacements(replacements):
        if not replacements:
            return []
        for key in replacements:
            if not isinstance(key, str) or not key:
                raise ValueError("Replacement keys must be non-empty strings.")
        keys = list(replacements.keys())
        for i, key_i in enumerate(keys):
            for key_j in keys[i + 1 :]:
                if key_i in key_j or key_j in key_i:
                    raise ValueError(
                        "Replacement patterns conflict: '{}' and '{}' overlap".format(
                            key_i, key_j
                        )
                    )
        return sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True)

    @staticmethod
    def _apply_replacements(text, ordered_replacements):
        if not isinstance(text, str):
            return text
        for old, new in ordered_replacements:
            text = text.replace(old, new)
        return text

    def with_replacements(self, replacements):
        ordered = self._validate_replacements(replacements)
        if not ordered:
            return self

        def replace(text: str) -> str:
            return self._apply_replacements(text, ordered)

        def replace_mapping(mapping, mapping_name):
            replaced = {}
            for key, value in mapping.items():
                new_key = replace(key)
                if new_key in replaced:
                    raise ValueError(
                        f"Replacement results in duplicate keys for {mapping_name}: '{new_key}'"
                    )
                replaced[new_key] = value
            return replaced

        new_flags = replace_mapping(self.flags, "flags")

        def replace_preset(preset, mapping_name):
            replaced = {}
            for key, value in preset.items():
                new_key = replace(key)
                if new_key in replaced:
                    raise ValueError(
                        f"Replacement results in duplicate keys for {mapping_name}: '{new_key}'"
                    )
                replaced[new_key] = value
            return replaced

        new_params = replace_preset(self.params, "params")
        new_steady = replace_preset(self.steady_guess, "steady_guess")

        new_exog = []
        for exog in self.exog_list:
            new_exog_name = replace(exog)
            if new_exog_name in new_exog:
                raise ValueError(
                    f"Replacement results in duplicate exogenous variables: '{new_exog_name}'"
                )
            new_exog.append(new_exog_name)

        new_rules = {}
        for key in self.rule_keys:
            od = MyOrderedDict()
            for rule_name, expression in self.rules.get(key, {}).items():
                new_name = replace(rule_name)
                if new_name in od:
                    raise ValueError(
                        f"Replacement results in duplicate rule names in '{key}': '{new_name}'"
                    )
                new_expr = replace(expression)
                od[new_name] = new_expr
            new_rules[key] = list(od.items())

        return BaseModelBlock(
            flags=new_flags,
            params=new_params,
            steady_guess=new_steady,
            rules=new_rules,
            exog_list=new_exog,
            rule_keys=self.rule_keys,
        )

    def add_block(
        self,
        block: "BaseModelBlock | None" = None,
        *,
        flags=None,
        params=None,
        steady_guess=None,
        rules=None,
        exog_list=None,
        overwrite: bool = False,
        rename: dict[str, str] | None = None,
    ):
        """Merge another block of configuration into this block.

        Parameters
        ----------
        block : BaseModelBlock, optional
            Pre-built block to merge. Provide either this or the keyword
            components below.
        flags, params, steady_guess, rules, exog_list : optional
            Components used to construct a temporary block if ``block`` is not
            supplied.
        overwrite : bool, default False
            When True, new values replace existing entries. When False,
            existing values win and only missing keys are appended.
        rename : dict, optional
            Mapping of substring replacements to apply prior to merging.

        Returns
        -------
        BaseModelBlock
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If both a block and component keywords are provided.
        TypeError
            If block is not a BaseModelBlock instance.
        ValueError
            If the rule_keys of the two blocks do not match.
        """

        if block is not None and any(
            value is not None
            for value in (flags, params, steady_guess, rules, exog_list)
        ):
            raise ValueError(
                "Provide either a BaseModelBlock or component keywords, not both."
            )

        if block is None:
            block = BaseModelBlock(
                flags=flags,
                params=params,
                steady_guess=steady_guess,
                rules=rules,
                exog_list=exog_list,
                rule_keys=self.rule_keys,
            )
        else:
            if not isinstance(block, BaseModelBlock):
                raise TypeError("block must be a BaseModelBlock instance")
            if block.rule_keys != self.rule_keys:
                raise ValueError(
                    f"Cannot merge blocks with different rule_keys: "
                    f"{self.rule_keys} vs {block.rule_keys}"
                )

        if rename:
            block = block.with_replacements(rename)

        # Merge flags
        if block.flags:
            if overwrite:
                self.flags.update(block.flags)
            else:
                for key, value in block.flags.items():
                    self.flags.setdefault(key, value)

        # Merge params & steady guesses using PresetDict helpers
        if block.params:
            if overwrite:
                self.params.overwrite_update(block.params)
            else:
                self.params.update(block.params)

        if block.steady_guess:
            if overwrite:
                self.steady_guess.overwrite_update(block.steady_guess)
            else:
                self.steady_guess.update(block.steady_guess)

        # Merge exogenous list while avoiding duplicates unless overwrite requested
        if block.exog_list:
            for exog in block.exog_list:
                if overwrite and exog in self.exog_list:
                    self.exog_list.remove(exog)
                if exog not in self.exog_list:
                    self.exog_list.append(exog)

        # Merge rules category by category
        for key in self.rule_keys:
            incoming = block.rules.get(key)
            if not incoming:
                continue
            destination = self.rules[key]
            for rule_name, expression in incoming.items():
                if not overwrite and rule_name in destination:
                    continue
                destination[rule_name] = expression

        return self

    def __add__(self, other: "BaseModelBlock") -> "BaseModelBlock":
        """Combine two BaseModelBlocks using the + operator.

        Creates a new block containing the merged contents of both blocks.
        The left operand's values take precedence (no overwrite).

        Parameters
        ----------
        other : BaseModelBlock
            The block to add to this one.

        Returns
        -------
        BaseModelBlock
            A new block containing the merged contents.

        Raises
        ------
        TypeError
            If other is not a BaseModelBlock instance.
        ValueError
            If the rule_keys of the two blocks do not match.
        """
        if not isinstance(other, BaseModelBlock):
            return NotImplemented

        # Create a copy of self to avoid modifying the original
        result = BaseModelBlock(
            flags=dict(self.flags),
            params=dict(self.params),
            steady_guess=dict(self.steady_guess),
            rules={key: list(val.items()) for key, val in self.rules.items()},
            exog_list=list(self.exog_list),
            rule_keys=self.rule_keys,
        )
        # Merge the other block into the copy
        result.add_block(other, overwrite=False)
        return result


class ModelBlock(BaseModelBlock):
    """Container for core model configuration artifacts using standard rule keys.

    This class automatically uses the standard rule keys defined in
    :attr:`Model.RULE_KEYS`. For custom rule keys, use :class:`BaseModelBlock`.
    """

    def __init__(
        self,
        *,
        flags=None,
        params=None,
        steady_guess=None,
        rules=None,
        exog_list=None,
    ):
        super().__init__(
            flags=flags,
            params=params,
            steady_guess=steady_guess,
            rules=rules,
            exog_list=exog_list,
            rule_keys=Model.RULE_KEYS,
        )


def model_block(func):
    """
    Decorator for model block creation functions.

    This decorator wraps a function to automatically create and return a ModelBlock
    instance. The decorated function should modify the block's attributes directly
    (e.g., `block.rules['intermediate'] += [...]`) and does not need to create or
    return the block explicitly.

    Parameters
    ----------
    func : callable
        A function that takes keyword arguments and modifies a ModelBlock instance.
        The function receives the block as its first positional argument followed
        by any keyword arguments passed to the decorated function.

    Returns
    -------
    callable
        A wrapped function that creates a ModelBlock, passes it to the original
        function, and returns the modified block.

    Examples
    --------
    Before using the decorator:

    >>> def my_block(*, param1=True):
    ...     block = ModelBlock()
    ...     block.rules['intermediate'] += [('x', 'param1 * 2')]
    ...     return block

    After using the decorator:

    >>> @model_block
    ... def my_block(block, *, param1=True):
    ...     block.rules['intermediate'] += [('x', 'param1 * 2')]

    Both approaches produce the same ModelBlock instance, but the decorated
    version is more concise and reduces boilerplate code.

    Notes
    -----
    The decorated function must accept `block` as its first positional argument.
    All other arguments should be keyword-only for clarity and consistency with
    existing block creation patterns.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(**kwargs):
        block = ModelBlock()
        func(block, **kwargs)
        return block

    return wrapper


class Model:

    RULE_KEYS = (
        "intermediate",
        "read_expectations",
        "transition",
        "expectations",
        "optimality",
        "calibration",
        "analytical_steady",
    )

    # Rule keys that must have unique keys across categories
    # analytical_steady and calibration are excluded:
    # - analytical_steady can override keys for steady-state computation
    # - calibration rules are merged into optimality during steady-state solving
    #   and intentionally duplicate keys to calibrate parameters
    UNIQUE_RULE_KEYS = (
        "intermediate",
        "read_expectations",
        "transition",
        "expectations",
        "optimality",
    )

    def _validate_unique_rule_keys(self):
        """
        Validate that there are no duplicate keys across rule categories.

        The following rule categories must have unique keys:
        - transition
        - expectations
        - optimality
        - intermediate
        - read_expectations

        analytical_steady and calibration are excluded from this check:
        - analytical_steady can duplicate keys from other categories for
          steady-state computation.
        - calibration rules are merged into optimality during steady-state
          solving and intentionally duplicate keys to calibrate parameters.

        Raises
        ------
        ValueError
            If duplicate keys are found across the specified rule categories.
        """
        seen_keys: dict[str, str] = {}  # Maps key -> first category where it appeared
        duplicates: list[tuple[str, str, str]] = []

        for category in self.UNIQUE_RULE_KEYS:
            rules_dict = self.rules.get(category, {})
            for key in rules_dict.keys():
                if key in seen_keys:
                    duplicates.append((key, seen_keys[key], category))
                else:
                    seen_keys[key] = category

        if duplicates:
            msg_parts = ["Duplicate keys found across rule categories:"]
            for key, first_cat, second_cat in duplicates:
                msg_parts.append(
                    f"  '{key}' appears in both '{first_cat}' and '{second_cat}'"
                )
            raise ValueError("\n".join(msg_parts))

    def __init__(
        self,
        flags=None,
        params=None,
        steady_guess=None,
        rules=None,
        exog_list=None,
        steady_flag=False,
        inner_functions=None,
        shared_function_bundles=None,
        label: str = "_default",
        fallback_labels: list[str] | None = None,
    ):
        """
        Initial setup of the model rules and parameters

        Parameters
        ----------
        flags : dict, optional
            Flags for the model. The default is None.
        params : dict, optional
            Parameters for the model. The default is None.
        steady_guess : dict, optional
            Initial guesses for the steady state. The default is None.
        rules : dict, optional
            Rules for the model. The default is None.
        exog_list : list, optional
            List of exogenous variables. The default is None.
        steady_flag : bool, optional
            Flag for whether this is a steady state version of the model.
        shared_function_bundles : dict, optional
            Pre-compiled function bundles to reuse across models.
        label : str, optional
            Identifier attached to this model. Defaults to "_default".
        """

        self.core_var_types = ["u", "x", "z", "E", "params"]

        # Initialize other inputs
        self.steady_flag = steady_flag
        self.label = label if label is not None else "_default"
        self.fallback_labels = list(fallback_labels or [])

        self.rule_keys = tuple(Model.RULE_KEYS)

        self.core = ModelBlock(
            flags=flags,
            params=params,
            steady_guess=steady_guess,
            rules=rules,
            exog_list=exog_list,
        )

        # The object for processing rules
        self.rp = RuleProcessor()

        # The module of generated code to be called
        self.inner_functions = inner_functions

        # Shared function bundles for reuse across model instances
        self._shared_function_bundles = shared_function_bundles

        # Initialize attributes to be set later
        self.var_lists = None
        self.steady_state = None
        # Initialize steady_dict to prevent AttributeError when accessing after failed solve attempts or before first successful solve
        self.steady_dict = {}
        # self.jac = None

        self.mod_steady = None
        self.mod_steady_cal = None

        self.linear_mod = None

        # Flags to track solution status
        self._steady_solved = False
        self._linearized = False

        # Initialize transformation registry
        # List of tuples: (var_list, transform_fn_str, inverse_fn_str, prefix)
        self._transformations = []

    def _normalize_rules(self, value):
        value = initialize_if_none(value, {})
        normalized = {}
        for key in self.rule_keys:
            raw = value.get(key)
            if isinstance(raw, MyOrderedDict):
                normalized[key] = raw
            else:
                normalized[key] = MyOrderedDict(raw or {})
        return normalized

    @property
    def flags(self):
        return self.core.flags

    @flags.setter
    def flags(self, value):
        self.core.flags = initialize_if_none(value, {})

    @property
    def params(self):
        return self.core.params

    @params.setter
    def params(self, value):
        value = initialize_if_none(value, {})
        self.core.params = value if isinstance(value, PresetDict) else PresetDict(value)

    @property
    def steady_guess(self):
        return self.core.steady_guess

    @steady_guess.setter
    def steady_guess(self, value):
        value = initialize_if_none(value, {})
        self.core.steady_guess = (
            value if isinstance(value, PresetDict) else PresetDict(value)
        )

    @property
    def exog_list(self):
        return self.core.exog_list

    @exog_list.setter
    def exog_list(self, value):
        value = [] if value is None else list(value)
        self.core.exog_list = value

    @property
    def rules(self):
        return self.core.rules

    @rules.setter
    def rules(self, value):
        self.core.rules = self._normalize_rules(value)

    def add_block(
        self,
        block: "BaseModelBlock | ModelBlock | None" = None,
        *,
        flags=None,
        params=None,
        steady_guess=None,
        rules=None,
        exog_list=None,
        overwrite: bool = False,
        rename: dict[str, str] | None = None,
    ):
        """Merge another block of configuration into the core block.

        Parameters
        ----------
        block : BaseModelBlock or ModelBlock, optional
            Pre-built block to merge. Provide either this or the keyword
            components below.
        flags, params, steady_guess, rules, exog_list : optional
            Components used to construct a temporary block if ``block`` is not
            supplied.
        overwrite : bool, default False
            When True, new values replace existing entries. When False,
            existing values win and only missing keys are appended.
        rename : dict, optional
            Mapping of substring replacements to apply prior to merging.
        """

        if block is not None and any(
            value is not None
            for value in (flags, params, steady_guess, rules, exog_list)
        ):
            raise ValueError(
                "Provide either a ModelBlock or component keywords, not both."
            )

        if block is None:
            block = ModelBlock(
                flags=flags,
                params=params,
                steady_guess=steady_guess,
                rules=rules,
                exog_list=exog_list,
            )
        else:
            if not isinstance(block, BaseModelBlock):
                raise TypeError("block must be a ModelBlock or BaseModelBlock instance")

        # Delegate to the core block's add_block method
        self.core.add_block(block, overwrite=overwrite, rename=rename)

        return self

    def add_exog(self, var, pers=None, vol=None):
        """
        Adds an exogenous variable to the model.

        Parameters
        ----------
        var : str
            Name of the exogenous variable.
        pers : float, optional
            Persistence of the exogenous variable. Defaults to None, in which case PERS_{var} must be set separately.
        vol : float, optional
            Volatility of the exogenous variable. Defaults to None, in which case VOL_{var} must be set separately.
        """

        self.exog_list.append(var)

        if pers is not None:
            self.params[f"PERS_{var}"] = pers
        if vol is not None:
            self.params[f"VOL_{var}"] = vol

    def _generate_prefix_from_function(self, fn_str):
        """
        Generate a suitable prefix based on the function string.

        Parameters
        ----------
        fn_str : str
            The function string, e.g., "np.log", "jnp.sqrt"

        Returns
        -------
        str
            A suitable prefix, e.g., "log", "sqrt"
        """
        import re

        # Remove "np." or "jnp." prefix
        prefix = re.sub(r"^(np|jnp)\.", "", fn_str)
        # Replace consecutive non-alphanumeric characters with underscores
        prefix = re.sub(r"[^a-zA-Z0-9]+", "_", prefix)
        # Remove leading/trailing underscores
        prefix = prefix.strip("_")
        return prefix

    def transform_variables(self, var_list, transform_fn, inverse_fn, prefix=None):
        """
        Register a transformation to be applied to specified variables.

        The transformation will be applied when finalize() is called. Variables
        will be replaced with their transformed versions:
        - Variable names on LHS get the prefix
        - Expressions on RHS get the transformation function applied
        - Uses of the variable in other rules get the inverse transformation applied

        Parameters
        ----------
        var_list : list of str
            List of variable names to transform.
        transform_fn : str
            The transformation function as a string, e.g., "np.log".
        inverse_fn : str
            The inverse transformation function as a string, e.g., "np.exp".
        prefix : str, optional
            Prefix to add to transformed variable names. If None, a suitable
            prefix will be generated based on the transform_fn.

        Examples
        --------
        >>> mod = Model()
        >>> mod.transform_variables(['x'], 'np.log', 'np.exp', 'log')
        >>> # After finalize(), 'x' in rules becomes 'log_x' on LHS
        >>> # and uses of 'x' in RHS become 'np.exp(log_x)'
        """
        if prefix is None:
            prefix = self._generate_prefix_from_function(transform_fn)

        # Store the transformation request
        self._transformations.append((var_list, transform_fn, inverse_fn, prefix))

    def log_transform(self, var_list, prefix="log"):
        """
        Convenience method to apply logarithmic transformation to variables.

        This is a wrapper for transform_variables() with transform_fn="np.log"
        and inverse_fn="np.exp".

        Parameters
        ----------
        var_list : list of str
            List of variable names to transform with logarithm.
        prefix : str, optional
            Prefix to add to transformed variable names. Defaults to "log".

        Examples
        --------
        >>> mod = Model()
        >>> mod.log_transform(['K', 'C'])
        >>> # After finalize(), 'K' becomes 'log_K', 'C' becomes 'log_C'
        >>> # Uses of 'K' in rules become 'np.exp(log_K)'
        """
        self.transform_variables(var_list, "np.log", "np.exp", prefix)

    def finalize(self):
        """
        Finalizes the model by converting the rules to code.
        """

        # Validate that there are no duplicate keys across rule categories
        self._validate_unique_rule_keys()

        # Set up initial guess for steady state
        self.init_dict = self.params.copy()
        self.init_dict.update(self.steady_guess)

        # Exogenous process parameters
        phi = np.array([self.params["PERS_" + exog] for exog in self.exog_list])
        # sig = np.array([self.params["VOL_" + exog] for exog in self.exog_list])

        Phi = np.diag(phi)
        # impact_matrix = np.diag(sig)
        impact_matrix = np.eye(len(self.exog_list))

        self.linear_mod = LinearModel(self, Phi=Phi, impact_matrix=impact_matrix)

        # Convert rules to code
        self._update_rules()
        if self.inner_functions is None:
            self._compile_rules()

        # processor = RuleProcessor()

        # If this is not a steady state version of the model, add one
        # Check if we should copy from parent model to avoid recompilation
        if not self.steady_flag:
            # Prepare params for steady models by excluding _STEADY parameters
            # since they are replaced with base variables in steady state rules
            steady_params = {
                k: v for k, v in self.params.items() if not k.endswith("_STEADY")
            }

            if hasattr(self, "_copy_steady_from"):
                # Copy steady models from parent with updated params
                parent = self._copy_steady_from
                if hasattr(parent, "mod_steady") and parent.mod_steady is not None:
                    self.mod_steady = parent.mod_steady.update_copy(
                        params=steady_params,
                        steady_flag=True,
                        label=f"{self.label}_steady",
                    )
                else:
                    rules_steady = self.rp.get_steady_rules(self.rules, calibrate=False)
                    self.mod_steady = self.update_copy(
                        rules_steady,
                        steady_flag=True,
                        label=f"{self.label}_steady",
                    )

                if (
                    hasattr(parent, "mod_steady_cal")
                    and parent.mod_steady_cal is not None
                ):
                    self.mod_steady_cal = parent.mod_steady_cal.update_copy(
                        params=steady_params,
                        steady_flag=True,
                        label=f"{self.label}_steady_cal",
                    )
                else:
                    rules_steady_cal = self.rp.get_steady_rules(
                        self.rules, calibrate=True
                    )
                    self.mod_steady_cal = self.update_copy(
                        rules_steady_cal,
                        steady_flag=True,
                        label=f"{self.label}_steady_cal",
                    )

                # Clean up the marker
                delattr(self, "_copy_steady_from")
            else:
                # Create new steady models
                rules_steady = self.rp.get_steady_rules(self.rules, calibrate=False)
                self.mod_steady = self.update_copy(
                    rules_steady,
                    steady_flag=True,
                    label=f"{self.label}_steady",
                )

                rules_steady_cal = self.rp.get_steady_rules(self.rules, calibrate=True)
                self.mod_steady_cal = self.update_copy(
                    rules_steady_cal,
                    steady_flag=True,
                    label=f"{self.label}_steady_cal",
                )

        # Set sizes
        self.N = {key: len(val) for key, val in self.var_lists.items()}

        # Set jacobian of objective function using FunctionBundle
        objfcn_steady_bundle = FunctionBundle(
            self.objfcn_steady,
            argnums=0,
            has_aux=False,
        )
        self.jac_objfcn_steady = objfcn_steady_bundle.jacobian_jit[0]

        # Useful for mapping letters to numbers
        self.arg_lists = {
            "transition": ["u", "x", "z", "params"],
            "expectations": ["u", "x", "z", "u_new", "x_new", "z_new", "params"],
            "optimality": ["u", "x", "z", "E", "params"],
            "intermediates": ["u", "x", "z", "params"],
            "expectations_variables": ["u", "x", "z", "E", "params"],
        }

        # Create FunctionBundle instances for each function
        # If we have shared bundles (from update_copy), reuse them
        if self._shared_function_bundles is not None:
            # Reuse the shared function bundles (no changes needed)
            pass
        else:
            # Create new function bundles (one per function, not per argnum)
            self._shared_function_bundles = {}

            for key, var_list in self.arg_lists.items():
                fn = getattr(self, key)  # fetch the bound method ONCE

                # Create a single FunctionBundle for all argnums for this function
                # Pass list of argnums (indices) to create all jacobians at once
                argnums_list = list(range(len(var_list)))
                bundle = FunctionBundle(
                    fn,
                    argnums=argnums_list,
                    has_aux=False,
                )

                # Store the bundle with a mapping from var name to argnum index
                self._shared_function_bundles[key] = {
                    "bundle": bundle,
                    "var_to_argnum": {var: ii for ii, var in enumerate(var_list)},
                }

        # from py_tools.utilities import tic, toc
        # Warm up the jacobians
        # for name, arg_list in self.arg_lists.items():
        #     these_args = [jnp.array(np.ones(len(self.var_lists[arg.replace('_new', '')])), dtype=jnp.float64) for arg in arg_list]
        #     for arg in arg_list:
        #         # start = tic()
        #         self.jacobians[name][arg](*these_args)
        #         # toc(start)
        #         # start = tic()
        #         # for ii in range(100):
        #         #     self.jacobians[name][arg](*these_args)
        #         # toc(start)

    def fcn(self, name, *args):

        std_args = standardize_args(*args)
        # Get the bundle (same primal for all argnums)
        bundle = self._shared_function_bundles[name]["bundle"]
        return bundle.f_jit(*std_args)

    def d(self, name, wrt, *args):
        """Jacobian of `name` w.r.t. argument `wrt`."""

        # trace_args(f"d[{name}][{wrt}]", *args)
        std_args = standardize_args(*args)
        for i, a in enumerate(std_args):
            assert isinstance(a, jax.Array), f"Arg {i} is not jax.Array: {type(a)}"

        bundle_info = self._shared_function_bundles[name]
        bundle = bundle_info["bundle"]
        argnum = bundle_info["var_to_argnum"][wrt]
        return bundle.jacobian_fwd_jit[argnum](*std_args)

    def d_wrt_multi(self, name, wrt_list, args):

        bundle_info = self._shared_function_bundles[name]
        bundle = bundle_info["bundle"]
        var_to_argnum = bundle_info["var_to_argnum"]

        return jnp.hstack(
            [bundle.jacobian_fwd_jit[var_to_argnum[wrt]](*args) for wrt in wrt_list]
        )

    def update_copy(
        self,
        rules=None,
        params=None,
        steady_flag=False,
        force_recompile=False,
        *,
        label: str | None = None,
    ):
        """
        Make a copy of the model with new rules

        Parameters
        ----------
        rules : dict
            New rules to add to the model. The default is None.
        params : dict
            New parameters to add to the model. The default is None.
        steady_flag : bool, optional
            Whether the model is a steady state version. The default is False.

        label : str, optional
            Override label for the copied model. Defaults to the current model's label when not provided.

        Returns
        -------
        mod : Model
            New model with updated rules.
        """

        if (
            (rules is None)
            and (steady_flag == self.steady_flag)
            and (not force_recompile)
        ):
            inner_functions = self.inner_functions
            shared_function_bundles = self._shared_function_bundles
        else:
            inner_functions = None
            shared_function_bundles = None

        these_rules = self.rules.copy()
        these_rules.update(initialize_if_none(rules, {}))

        these_params = self.params.copy()
        these_params.update(initialize_if_none(params, {}))

        mod = Model(
            flags=self.flags,
            params=these_params,
            steady_guess=self.steady_guess,
            rules=these_rules,
            exog_list=self.exog_list,
            steady_flag=steady_flag,
            inner_functions=inner_functions,
            shared_function_bundles=shared_function_bundles,
            label=self.label if label is None else label,
        )

        # If we're copying a non-steady model with only param changes,
        # set the steady sub-models before finalize to avoid recompilation
        if (not steady_flag) and (rules is None) and (not force_recompile):
            if hasattr(self, "mod_steady") and self.mod_steady is not None:
                # Mark that we want to copy the steady model
                mod._copy_steady_from = self

        mod.finalize()

        return mod

    def create_core_state_array(self, u=None, x=None, z=None, E=None, params=None):

        arrays = {"u": u, "x": x, "z": z, "E": E, "params": params}

        for key in self.core_var_types:
            if arrays[key] is None:
                arrays[key] = jnp.zeros(len(self.var_lists[key]))

        return jnp.hstack([arrays[name] for name in self.core_var_types])

    def array_to_state(self, **kwargs):
        """
        Converts jax.numpy DeviceArrays to a state array
        """

        core_vals = self.create_core_state_array(**kwargs)
        st = self.inner_functions.array_to_state(core_vals)

        return st

    def array_to_state_plus_intermediates(self, **kwargs):
        """
        Converts jax.numpy DeviceArrays to a state array
        """

        st0 = self.array_to_state(**kwargs)
        st1 = self.inner_functions.intermediate_variables(st0)

        return st1

    def _format_expression_with_values(self, rule_str, state_dict):
        """
        Format an expression with variable values substituted in brackets.

        Parameters
        ----------
        rule_str : str
            Expression to format.
        state_dict : dict
            Dictionary mapping variable names to their current values.

        Returns
        -------
        str
            Formatted expression with values, e.g., 'K [6.000] ** alp [0.360]'
        """
        import re

        # Pattern to match variable names in the expression
        # This matches the same pattern as RuleProcessor.p_word
        var_pattern = re.compile(
            r"\b(?!\w*\.\w*)(?!\w*_NEXT)(?!st|st_new)(?!np|jnp)(?<!\.)([a-zA-Z]\w*)(?!')(?!\s*\()\b"
        )

        def replace_with_value(match):
            var = match.group(1)
            # Get value from state_dict, or show 'N/A' if not available
            value = state_dict.get(var, np.nan)
            return f"{var} [{value:.6g}]"

        return var_pattern.sub(replace_with_value, rule_str)

    def format_equation_with_values(self, var_name, rule_str, state_dict):
        """
        Format an equation with variable values substituted in brackets.

        Parameters
        ----------
        var_name : str
            Name of the variable being defined (left-hand side).
        rule_str : str
            Expression defining the variable (right-hand side).
        state_dict : dict
            Dictionary mapping variable names to their current values.

        Returns
        -------
        str
            Formatted equation string with values, e.g.,
            'y [2.150] = K [6.000] ** alp [0.360]'

        Examples
        --------
        >>> model.format_equation_with_values('y', 'K ** alp',
        ...     {'y': 2.15, 'K': 6.0, 'alp': 0.36})
        'y [2.150] = K [6.000] ** alp [0.360]'
        """
        # Get the value of the left-hand side variable
        lhs_value = state_dict.get(var_name, np.nan)
        lhs_str = f"{var_name} [{lhs_value:.6g}]"

        # Format the right-hand side
        rhs_str = self._format_expression_with_values(rule_str, state_dict)

        return f"{lhs_str} = {rhs_str}"

    def expectations(self, u, x, z, u_new, x_new, z_new, params):

        # u, x, z, u_new, x_new, z_new, params = standardize_args(u, x, z, u_new, x_new, z_new, params)
        st = self.array_to_state_plus_intermediates(u=u, x=x, z=z, params=params)
        st_new = self.array_to_state_plus_intermediates(
            u=u_new, x=x_new, z=z_new, params=params
        )
        return self.inner_functions.expectations_inner(st, st_new)

    def transition(self, u, x, z, params):

        # u, x, z, params = standardize_args(u, x, z, params)
        st = self.array_to_state_plus_intermediates(u=u, x=x, z=z, params=params)
        return self.inner_functions.transition_inner(st)

    def optimality(self, u, x, z, E, params):

        # u, x, z, E, params = standardize_args(u, x, z, E, params)
        st = self.array_to_state_plus_intermediates(u=u, x=x, z=z, E=E, params=params)
        st = self.inner_functions.read_expectations_variables(st)
        return self.inner_functions.optimality_inner(st)

    def intermediates(self, u, x, z, params):

        # u, x, z, params = standardize_args(u, x, z, params)
        st = self.array_to_state(u=u, x=x, z=z, params=params)
        return self.inner_functions.intermediate_variables_array(st)

    def expectations_variables(self, u, x, z, E, params):

        # u, x, z, E, params = standardize_args(u, x, z, E, params)
        st = self.array_to_state(u=u, x=x, z=z, E=E, params=params)
        return self.inner_functions.read_expectations_variables_array(st)

    def dict_to_components(self, st):

        # components = {}
        for name in ["u", "x", "z", "params"]:
            self.steady_components[name] = jnp.array(
                [self.steady_dict[key] for key in self.var_lists[name]]
            )

    def get_steady_err(self, u, x, z, params):

        x_new = self.transition(u, x, z, params)
        E = self.expectations(u, x, z, u, x, z, params)

        err_pol = self.optimality(u, x, z, E, params)
        err_trans = x_new - x

        err_all = jnp.hstack((err_trans, err_pol))

        # self.d_get_steady_err(u, x, z, params)

        return err_all

    def objfcn_steady(self, ux, z, params):

        # ux, z, params = standardize_args(ux, z, params)
        u, x = jnp.split(ux, [self.N["u"]])
        return self.get_steady_err(u, x, z, params)

    def d_objfcn_steady(self, ux, z, params):
        """Manual version -- not needed due to jax automated version"""

        u, x = jnp.split(ux, [self.N["u"]])

        E = self.fcn("expectations", u, x, z, u, x, z, params)

        x_new_ux = self.d_wrt_multi("transition", ["u", "x"], (u, x, z, params))
        d_trans = x_new_ux

        d_E = self.d_wrt_multi("expectations", ["u", "x"], (u, x, z, u, x, z, params))
        f_ux = self.d_wrt_multi("optimality", ["u", "x"], (u, x, z, E, params))

        # Get jacobian for 'E' argument in optimality function
        bundle_info = self._shared_function_bundles["optimality"]
        bundle = bundle_info["bundle"]
        argnum_E = bundle_info["var_to_argnum"]["E"]
        f_E = bundle.jacobian_fwd_jit[argnum_E](u, x, z, E, params)

        d_optimality = f_ux + f_E @ d_E

        return jnp.vstack((d_trans, d_optimality))

    def initialize_values(self, init_dict):

        init_vals = {}
        for name in ["u", "x", "params"]:
            values_list = []
            for key in self.var_lists[name]:
                # For params, fall back to self.params if not in init_dict
                # This handles _STEADY parameters that may not be in steady_dict
                if name == "params" and key not in init_dict:
                    values_list.append(jnp.asarray(self.params[key], dtype=jnp.float64))
                else:
                    values_list.append(jnp.asarray(init_dict[key], dtype=jnp.float64))
            # Handle empty lists gracefully
            if values_list:
                init_vals[name] = jnp.stack(values_list)
            else:
                init_vals[name] = jnp.array([], dtype=jnp.float64)

        init_vals["z"] = jnp.zeros(len(self.exog_list))

        # Exogenous variables are zero in steady state
        init_vals["z"] = jnp.zeros(len(self.var_lists["z"]))

        return init_vals

    def _solve_steady_attempt(
        self,
        init_vals=None,
        init_dict=None,
        calibrate=False,
        save: bool = False,
        load_initial_guess: bool = True,
        backup_to_use: int | None = None,
        verbose_iterations: bool = False,
        solver: str = "newton",
    ) -> bool:
        """Internal helper that performs one steady-state solve attempt.

        Parameters
        ----------
        verbose_iterations : bool, optional
            If True, saves detailed iteration log with variable values and
            equations to a file in the debug directory. Default is False.
        solver : {"auto", "scipy", "newton"}, optional
            Solver selection. "auto" uses "newton" when verbose_iterations is
            enabled, otherwise "scipy". If "newton" fails, falls back to "scipy".
        """

        if init_dict is None:
            init_dict = self.init_dict

        if self.steady_flag:

            if init_vals is None:
                init_vals = self.initialize_values(init_dict)

            # This is the steady state model, solve accordingly
            # Handle empty u or x arrays gracefully
            if self.N["u"] > 0 and self.N["x"] > 0:
                x0 = jnp.hstack((init_vals["u"], init_vals["x"]))
            elif self.N["u"] > 0:
                x0 = init_vals["u"]
            elif self.N["x"] > 0:
                x0 = init_vals["x"]
            else:
                # Both u and x are empty - nothing to solve
                # All variables have analytical solutions
                x0 = None

            if x0 is not None:
                # Need to solve for some variables
                args = (init_vals["z"], init_vals["params"])
                # Use the jitted function from FunctionBundle created in finalize()
                objfcn_steady_bundle = FunctionBundle(
                    self.objfcn_steady,
                    argnums=0,
                    has_aux=False,
                )
                self.objfcn_steady_jit = objfcn_steady_bundle.f_jit

                # Set up callback for verbose iteration logging
                callback = None
                log_file = None
                log_path = None
                debug_dir = None

                if verbose_iterations:
                    from datetime import datetime

                    settings = get_settings()
                    debug_dir = Path(settings.paths.debug_dir)
                    debug_dir.mkdir(parents=True, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_filename = f"{self.label}_steady_iterations_{timestamp}.txt"
                    log_path = debug_dir / log_filename

                    print(f"Saving detailed iteration log to: {log_path}")

                    # Open file and write header
                    log_file = open(log_path, "w")
                    log_file.write("=" * 70 + "\n")
                    log_file.write(
                        "Steady State Solve Attempt - Detailed Iteration Log\n"
                    )
                    log_file.write("=" * 70 + "\n")
                    log_file.write(
                        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    log_file.write(f"Model Label: {self.label}\n")
                    log_file.write("=" * 70 + "\n\n")

                    def iteration_callback(iteration, ux, f_val, dist):
                        """Callback to log detailed iteration information."""
                        log_file.write(f"\nIteration {iteration}: |f| = {dist:.6e}\n")
                        log_file.write("-" * 70 + "\n")

                        # Split ux into u and x
                        if self.N["u"] > 0 and self.N["x"] > 0:
                            u, x = jnp.split(ux, [self.N["u"]])
                        elif self.N["u"] > 0:
                            u = ux
                            x = jnp.array([], dtype=jnp.float64)
                        else:  # self.N["x"] > 0
                            u = jnp.array([], dtype=jnp.float64)
                            x = ux

                        # Get full state with intermediates
                        state = self.array_to_state_plus_intermediates(
                            u=u, x=x, z=init_vals["z"], params=init_vals["params"]
                        )

                        # Compute expectations (E variables) for steady state
                        E = self.expectations(
                            u,
                            x,
                            init_vals["z"],
                            u,
                            x,
                            init_vals["z"],
                            init_vals["params"],
                        )

                        # Add E variables to state
                        state = self.inner_functions.read_expectations_variables(state)

                        # Convert State NamedTuple to dict for easier access
                        state_dict = state._asdict()

                        # Add computed E array values to state_dict
                        # E is an array, map it to expectation variable names
                        if hasattr(self, "var_lists") and "E" in self.var_lists:
                            for i, e_var_name in enumerate(self.var_lists["E"]):
                                if i < len(E):
                                    state_dict[e_var_name] = E[i]

                        # Write equations for intermediate variables
                        log_file.write("\n  Intermediate Variables:\n")
                        for var_name, rule_str in self.rules["intermediate"].items():
                            equation = self.format_equation_with_values(
                                var_name, rule_str, state_dict
                            )
                            log_file.write(f"    {equation}\n")

                        # Write equations for expectation variables
                        if self.rules.get("expectations"):
                            log_file.write("\n  Expectations:\n")
                            for var_name, rule_str in self.rules[
                                "expectations"
                            ].items():
                                # For steady state, drop _NEXT suffixes since next period = current period
                                rule_str_steady = rule_str.replace("_NEXT", "")
                                equation = self.format_equation_with_values(
                                    var_name, rule_str_steady, state_dict
                                )
                                log_file.write(f"    {equation}\n")

                        # Write equations for expectation variables read from state
                        if self.rules.get("read_expectations"):
                            log_file.write("\n  Expectation Variables (from state):\n")
                            for var_name, rule_str in self.rules[
                                "read_expectations"
                            ].items():
                                equation = self.format_equation_with_values(
                                    var_name, rule_str, state_dict
                                )
                                log_file.write(f"    {equation}\n")

                        # Write solved variables (u and x) with their values
                        log_file.write("\n  State Variables (x):\n")
                        for i, var_name in enumerate(self.var_lists["x"]):
                            value = state_dict.get(var_name, np.nan)
                            log_file.write(f"    {var_name}: {value:.6g}\n")

                        log_file.write("\n  Control Variables (u):\n")
                        for i, var_name in enumerate(self.var_lists["u"]):
                            value = state_dict.get(var_name, np.nan)
                            log_file.write(f"    {var_name}: {value:.6g}\n")

                        # Write residuals with equilibrium conditions
                        log_file.write("\n  Residuals:\n")

                        # Compute x_new for transition residuals
                        x_new = self.transition(
                            u, x, z=init_vals["z"], params=init_vals["params"]
                        )

                        # First N["x"] residuals are transition errors: x_new - x = 0
                        for i, var_name in enumerate(self.var_lists["x"]):
                            if i < len(f_val):
                                residual_val = f_val[i]
                                x_val = state_dict.get(var_name, np.nan)
                                x_new_val = x_new[i] if i < len(x_new) else np.nan

                                # Format the equilibrium condition
                                residual_eq = (
                                    f"    {var_name} (transition) [{residual_val:.6e}] = "
                                    f"{var_name}_new [{x_new_val:.6g}] - {var_name} [{x_val:.6g}]\n"
                                )
                                log_file.write(residual_eq)

                        # Remaining residuals are optimality errors
                        for i, var_name in enumerate(self.var_lists["u"]):
                            idx = self.N["x"] + i
                            if idx < len(f_val):
                                residual_val = f_val[idx]

                                # Get the optimality equation
                                if var_name in self.rules.get("optimality", {}):
                                    rule_str = self.rules["optimality"][var_name]
                                    # Format the equation with values
                                    rhs_formatted = self._format_expression_with_values(
                                        rule_str, state_dict
                                    )
                                    residual_eq = (
                                        f"    {var_name} (optimality) [{residual_val:.6e}] = "
                                        f"{rhs_formatted}\n"
                                    )
                                else:
                                    residual_eq = f"    {var_name} (optimality): {residual_val:.6e}\n"

                                log_file.write(residual_eq)

                        log_file.flush()

                    callback = iteration_callback

                # res = root(objfcn_jit, x0, args=args, jac=jac_objfcn_jit)
                solver_choice = solver
                if solver_choice == "auto":
                    solver_choice = "newton" if verbose_iterations else "scipy"

                if solver_choice not in {"scipy", "newton"}:
                    raise ValueError(
                        "solver must be one of {'auto', 'scipy', 'newton'}"
                    )

                if verbose_iterations and solver_choice == "scipy":
                    raise ValueError(
                        "verbose_iterations requires solver='newton' for callbacks"
                    )

                try:
                    if solver_choice == "newton":
                        # Use custom newton solver that supports callback
                        res = newton_root(
                            self.objfcn_steady_jit,
                            x0,
                            args=args,
                            jac=self.jac_objfcn_steady,
                            callback=callback,
                        )
                        if not res.success:
                            # Fall back to scipy without verbose callback
                            res = scipy_root(
                                self.objfcn_steady_jit,
                                x0,
                                args=args,
                                jac=self.jac_objfcn_steady,
                            )
                    else:
                        # Use scipy's root solver for backward compatibility
                        res = scipy_root(
                            self.objfcn_steady_jit,
                            x0,
                            args=args,
                            jac=self.jac_objfcn_steady,
                        )
                finally:
                    # Close log file if it was opened
                    if verbose_iterations and callback is not None:
                        log_file.write("\n" + "=" * 70 + "\n")
                        log_file.write("End of iteration log\n")
                        log_file.write("=" * 70 + "\n")
                        log_file.close()
                        print(f"Iteration log saved to: {log_path}")
                        if debug_dir is not None:
                            io.prune_files_by_stem(
                                debug_dir,
                                f"{self.label}_steady_iterations",
                                keep=settings.debug.keep_iteration_logs,
                                suffix=".txt",
                            )

                self.res_steady = res

                if self.res_steady.success:
                    # Handle splitting the result based on N["u"] and N["x"]
                    if self.N["u"] > 0 and self.N["x"] > 0:
                        u_hat, x_hat = jnp.split(self.res_steady.x, [self.N["u"]])
                    elif self.N["u"] > 0:
                        u_hat = self.res_steady.x
                        x_hat = jnp.array([], dtype=jnp.float64)
                    else:  # self.N["x"] > 0
                        u_hat = jnp.array([], dtype=jnp.float64)
                        x_hat = self.res_steady.x

                    self.steady_dict = self.array_to_state_plus_intermediates(
                        u=u_hat, x=x_hat, z=init_vals["z"], params=init_vals["params"]
                    )
                    self.steady_dict = self.inner_functions.read_expectations_variables(
                        self.steady_dict
                    )
            else:
                # Both u and x are empty - all variables have analytical solutions
                # No solver needed, just compute steady state directly
                u_hat = jnp.array([], dtype=jnp.float64)
                x_hat = jnp.array([], dtype=jnp.float64)

                self.steady_dict = self.array_to_state_plus_intermediates(
                    u=u_hat, x=x_hat, z=init_vals["z"], params=init_vals["params"]
                )
                self.steady_dict = self.inner_functions.read_expectations_variables(
                    self.steady_dict
                )

                # Create a successful result object for consistency
                from types import SimpleNamespace

                self.res_steady = SimpleNamespace(
                    success=True,
                    x=jnp.array([], dtype=jnp.float64),
                    message="All variables have analytical solutions",
                )

        else:

            if load_initial_guess:
                # Always try to load from calibrated version first to get both
                # steady state values and calibrated parameters
                loaded = self.load_steady(
                    load_calibration=True,
                    backup_to_use=backup_to_use,
                )

                # If calibrated version not found, try non-calibrated version
                if not loaded:
                    loaded = self.load_steady(
                        load_calibration=False,
                        backup_to_use=backup_to_use,
                    )

                if loaded:
                    init_guess = dict(getattr(self, "init_dict", {}))
                    # Get the current calibration parameter names
                    calibration_params = set(self.rules.get("calibration", {}).keys())

                    # Get list of all parameter names (from params dict if var_lists not available)
                    if self.var_lists is not None:
                        param_names = set(self.var_lists.get("params", []))
                    else:
                        param_names = set()

                    # If var_lists not available or empty, fall back to params dict
                    if not param_names:
                        param_names = set(self.params.keys())

                    for key, value in loaded.items():
                        # Only load calibrated parameters if they're still in calibration rules
                        # For non-parameter variables, always load them
                        if key in param_names:
                            # This is a parameter - only load if in calibration rules
                            if key in calibration_params:
                                init_guess[key] = jnp.asarray(value, dtype=jnp.float64)
                            # Otherwise skip this parameter (keep preset value)
                        else:
                            # This is a variable (not a parameter) - always load it
                            init_guess[key] = jnp.asarray(value, dtype=jnp.float64)

                    self.init_dict = init_guess

            if calibrate:
                this_mod = self.mod_steady_cal
            else:
                this_mod = self.mod_steady

            this_mod.solve_steady(
                init_vals=init_vals,
                init_dict=self.init_dict,  # Use updated init_dict after loading
                calibrate=False,
                save=save,
                backup_to_use=backup_to_use,
                display=False,  # Don't display in sub-model; parent will display
                verbose_iterations=verbose_iterations,
            )
            # Guard against AttributeError when this_mod doesn't have steady_dict
            # (happens when solve fails before setting it)
            if hasattr(this_mod, "steady_dict") and this_mod.steady_dict is not None:
                self.steady_dict = this_mod.steady_dict
            self.res_steady = this_mod.res_steady

        if getattr(self, "steady_dict", None) is not None:
            success = getattr(self.res_steady, "success", False)
            if success:
                # Verify that the steady state satisfies all equations,
                # including those replaced by analytical_steady rules
                if not self.steady_flag:
                    self._verify_analytical_steady()
                self.update_steady(calibrate=calibrate)
                if save:
                    self._save_steady_snapshot()
            else:
                self.steady_dict = {}
        else:
            success = getattr(self.res_steady, "success", False)

        return success

    def _steady_dict_has_key(self, key: str) -> bool:
        """Check if a key exists in steady_dict (works for both dict and NamedTuple)."""
        if hasattr(self.steady_dict, "_fields"):
            # It's a NamedTuple - check if key is in field names
            return key in self.steady_dict._fields
        else:
            # It's a dict-like object - use standard `in` check
            return key in self.steady_dict

    def _verify_analytical_steady(self, tol: float = 1e-8) -> bool:
        """Verify that the steady state satisfies all transition and optimality equations.

        This check is particularly important when analytical_steady rules are used,
        as incorrect analytical formulas can lead to steady states that don't satisfy
        the original model equations.

        Parameters
        ----------
        tol : float, optional
            Tolerance for the residual check. Default is 1e-8.

        Returns
        -------
        bool
            True if all equations are satisfied within tolerance, False otherwise.

        Raises
        ------
        ValueError
            If the steady state does not satisfy the transition or optimality
            equations within the specified tolerance. The error message includes
            details about which equations failed.
        """
        if not getattr(self, "steady_dict", None):
            return True

        # Build the arrays manually, using init_dict as fallback for parameters
        # that may not be in steady_dict (like _STEADY parameters)
        u_vals = []
        for key in self.var_lists["u"]:
            if self._steady_dict_has_key(key):
                u_vals.append(float(self.steady_dict[key]))
            else:
                u_vals.append(float(self.init_dict.get(key, 0.0)))
        u = jnp.array(u_vals, dtype=jnp.float64)

        x_vals = []
        for key in self.var_lists["x"]:
            if self._steady_dict_has_key(key):
                x_vals.append(float(self.steady_dict[key]))
            else:
                x_vals.append(float(self.init_dict.get(key, 0.0)))
        x = jnp.array(x_vals, dtype=jnp.float64)

        params_vals = []
        for key in self.var_lists["params"]:
            if self._steady_dict_has_key(key):
                params_vals.append(float(self.steady_dict[key]))
            else:
                params_vals.append(float(self.init_dict.get(key, 0.0)))
        params = jnp.array(params_vals, dtype=jnp.float64)

        z = jnp.zeros(len(self.var_lists["z"]), dtype=jnp.float64)

        # Compute residuals for transition and optimality equations
        err_all = self.get_steady_err(u, x, z, params)

        # Split into transition and optimality residuals
        n_x = self.N["x"]
        err_trans = err_all[:n_x]
        err_opt = err_all[n_x:]

        # Check for violations
        trans_violations = []
        opt_violations = []

        trans_var_names = self.var_lists["x"]
        for i, (var_name, err) in enumerate(zip(trans_var_names, err_trans)):
            if jnp.abs(err) > tol:
                trans_violations.append((var_name, float(err)))

        opt_var_names = self.var_lists["u"]
        for i, (var_name, err) in enumerate(zip(opt_var_names, err_opt)):
            if jnp.abs(err) > tol:
                opt_violations.append((var_name, float(err)))

        if trans_violations or opt_violations:
            msg_parts = ["Steady state verification failed."]
            if trans_violations:
                msg_parts.append(
                    "Transition equations not satisfied: "
                    + ", ".join(f"{var}={err:.2e}" for var, err in trans_violations)
                )
            if opt_violations:
                msg_parts.append(
                    "Optimality conditions not satisfied: "
                    + ", ".join(f"{var}={err:.2e}" for var, err in opt_violations)
                )

            # Check if this is likely due to analytical_steady rules
            if self.rules.get("analytical_steady"):
                analytical_vars = set(self.rules["analytical_steady"].keys())
                failed_trans_vars = {var for var, _ in trans_violations}
                if failed_trans_vars & analytical_vars:
                    msg_parts.append(
                        "Note: Some failed transition equations correspond to "
                        "variables with analytical_steady rules. Please verify "
                        "that the analytical formulas are correct."
                    )

            raise ValueError(" ".join(msg_parts))

        return True

    def update_steady(self, *, calibrate: bool = False):
        """Refresh steady-state components and optionally calibrate parameters."""

        self.steady_components = {}
        for name in ["u", "x", "params"]:
            self.steady_components[name] = jnp.array(
                [
                    (
                        self.steady_dict[key]
                        if self._steady_dict_has_key(key)
                        else self.init_dict.get(key, 0.0)
                    )
                    for key in self.var_lists[name]
                ]
            )

        # Exogenous variables are zero in steady state
        self.steady_components["z"] = jnp.zeros(len(self.var_lists["z"]))

        if calibrate:
            # Get calibration parameter names to filter updates
            calibration_params = set(self.rules.get("calibration", {}).keys())

            # Only update parameters that are in the calibration rules
            # Other parameters keep their preset values
            for key in self.var_lists["params"]:
                if key in calibration_params and self._steady_dict_has_key(key):
                    # This is a calibrated parameter - update it from steady_dict
                    # Use overwrite_item since PresetDict won't update existing keys
                    self.params.overwrite_item(key, self.steady_dict[key])
                # For non-calibrated parameters, keep their current value in self.params

        # Update _STEADY parameters with steady state values
        steady_vars = getattr(self, "_steady_vars", set())
        if steady_vars and not self.steady_flag:
            for var in steady_vars:
                param_name = f"{var}_STEADY"
                if self._steady_dict_has_key(var):
                    steady_value = float(self.steady_dict[var])
                    # Use overwrite_item since PresetDict won't update existing keys
                    self.params.overwrite_item(param_name, steady_value)
                    # Also update init_dict for consistency
                    self.init_dict[param_name] = steady_value

        # Mark steady state as solved
        self._steady_solved = True

    def _load_steady_snapshot(self, backup_to_use: int | None = None):
        """Load steady-state values previously saved with :meth:`solve_steady`."""

        path = self._steady_snapshot_path()

        if backup_to_use is not None:
            if backup_to_use < 0:
                logger.warning("Backup index must be non-negative; skipping load.")
                return {}

            stem = f"{self.label}_steady_state"
            backups = io.list_steady_backups(path, stem=stem)
            if not backups:
                logger.info("No steady-state backups available; skipping load.")
                return {}

            if backup_to_use >= len(backups):
                logger.warning(
                    "Requested backup index %d but only %d available; skipping load.",
                    backup_to_use,
                    len(backups),
                )
                return {}

            path = backups[-(backup_to_use + 1)]

        data = io.load_json(path)
        restored = {
            key: jnp.asarray(value, dtype=jnp.float64) for key, value in data.items()
        }
        self.steady_dict = restored
        return restored

    def solve_steady(
        self,
        init_vals=None,
        init_dict=None,
        calibrate=False,
        save: bool = False,
        load_initial_guess: bool = True,
        backup_to_use: int | None = None,
        *,
        max_backup_attempts: int | None = None,
        display: bool = True,
        verbose_iterations: bool = False,
        solver: str = "newton",
    ):
        """
        Solve for the steady state with retries using fallback initial guesses.

        Parameters
        ----------
        verbose_iterations : bool, optional
            If True, saves detailed iteration log with variable values and
            equations to a file in the debug directory. Default is False.
            The log includes intermediate variables with their defining equations,
            all variable values, and residuals at each iteration.
            Only the first attempt produces a log; fallback attempts suppress logs.
        solver : {"auto", "scipy", "newton"}, optional
            Solver selection. "auto" uses "newton" when verbose_iterations is
            enabled, otherwise "scipy". If "newton" fails, falls back to "scipy".
        """

        attempt_count = 0

        def attempt(description: str, **kwargs) -> bool:
            nonlocal attempt_count
            if self.steady_flag:
                logger.info("Attempting steady-state solve with %s...", description)
            use_verbose = verbose_iterations and attempt_count == 0
            attempt_count += 1
            success = self._solve_steady_attempt(
                init_vals=init_vals,
                init_dict=init_dict,
                calibrate=calibrate,
                save=save,
                load_initial_guess=kwargs.get("load_initial_guess", load_initial_guess),
                backup_to_use=kwargs.get("backup_to_use", backup_to_use),
                verbose_iterations=use_verbose,
                solver=solver,
            )
            if self.steady_flag:
                if success:
                    logger.info("Steady-state solve succeeded.")
                else:
                    logger.warning("Steady-state solve failed.")
            return success

        if attempt("load_initial_guess=True", load_initial_guess=True):
            if self.res_steady.success and display:
                self.print_steady()
            return self.res_steady

        if attempt("load_initial_guess=False", load_initial_guess=False):
            if self.res_steady.success and display:
                self.print_steady()
            return self.res_steady

        def backup_attempts_for_label(label: str) -> bool:
            stem = f"{label}_steady_state"
            backups = io.list_steady_backups(self._steady_snapshot_path(), stem=stem)
            if not backups:
                logger.info("No backups found for label '%s'.", label)
                return False

            available = len(backups)
            limit = (
                available
                if max_backup_attempts is None
                else min(max_backup_attempts, available)
            )

            for idx in range(limit):
                if attempt(
                    f"backup #{idx} for label '{label}'",
                    load_initial_guess=True,
                    backup_to_use=idx,
                ):
                    return True
            return False

        if backup_attempts_for_label(self.label):
            if self.res_steady.success and display:
                self.print_steady()
            return self.res_steady

        for alt_label in self.fallback_labels:
            if backup_attempts_for_label(alt_label):
                if self.res_steady.success and display:
                    self.print_steady()
                return self.res_steady

        if self.res_steady.success and display:
            self.print_steady()

        return self.res_steady

    def load_steady(
        self,
        *,
        load_calibration: bool = True,
        backup_to_use: int | None = None,
    ):
        """Load steady-state solution from the cached steady or calibration model."""

        target = self.mod_steady_cal if load_calibration else self.mod_steady
        if target is None:
            raise RuntimeError(
                "Steady model has not been initialized; call finalize() first."
            )

        try:
            # Load from parent model's path (not sub-model's path)
            # because _save_steady_snapshot is called on the parent
            self.steady_dict = self._load_steady_snapshot(backup_to_use=backup_to_use)
        except FileNotFoundError:
            logger.warning(
                "No saved steady-state snapshot found; run solve_steady(save=True) first."
            )
            return {}

        if not self.steady_dict:
            return {}

        self.update_steady(calibrate=load_calibration)

        return self.steady_dict

    def print_steady(self):
        """Display steady-state parameters and variables in tabular form."""

        if not getattr(self, "steady_dict", None):
            raise RuntimeError(
                "Steady-state values are unavailable; run solve_steady() first."
            )

        if self.inner_functions is None:
            raise RuntimeError(
                "Inner functions are not compiled; call finalize() before printing."
            )

        if self.var_lists is None:
            self._get_var_lists()

        def _steady_to_mapping(data):
            if hasattr(data, "items"):
                return dict(data.items())
            if hasattr(data, "_asdict"):
                return dict(data._asdict())
            keys = list(
                itertools.chain.from_iterable(
                    self.var_lists.get(name, [])
                    for name in (
                        "u",
                        "x",
                        "z",
                        "E",
                        "params",
                        "intermediate",
                        "read_expectations",
                    )
                )
            )
            mapping = {}
            for key in keys:
                try:
                    mapping[key] = data[key]
                except (KeyError, TypeError):
                    continue
            return mapping

        steady_mapping = _steady_to_mapping(self.steady_dict)

        init_vals = self.initialize_values(self.steady_dict)
        core_kwargs = {
            "u": init_vals["u"],
            "x": init_vals["x"],
            "z": init_vals["z"],
            "params": init_vals["params"],
        }

        state = self.array_to_state_plus_intermediates(**core_kwargs)
        state = self.inner_functions.read_expectations_variables(state)

        if self.var_lists.get("E"):
            expectations_array = self.inner_functions.expectations_inner(state, state)
            update = {
                name: expectations_array[idx]
                for idx, name in enumerate(self.var_lists["E"])
            }
            state = state._replace(**update)

        state_mapping = dict(state._asdict())
        full_values = {**steady_mapping, **state_mapping}

        def _lookup(name):
            if name in full_values:
                return full_values[name]
            if hasattr(self.steady_dict, "__getitem__"):
                try:
                    return self.steady_dict[name]
                except (KeyError, TypeError):
                    pass
            if name in self.params:
                return self.params[name]
            logger.warning("No steady-state value found for '%s'; using NaN.", name)
            return np.nan

        def _prepare(value):
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return str(value)
            if array.size == 1:
                return float(array.reshape(()))
            return np.array2string(array, precision=6, separator=", ")

        param_names = sorted(self.var_lists.get("params", []))
        param_rows = [(name, _prepare(_lookup(name))) for name in param_names]

        variable_names = sorted(
            {
                name
                for category in (
                    "u",
                    "x",
                    "z",
                    "E",
                    "intermediate",
                    "read_expectations",
                )
                for name in self.var_lists.get(category, [])
            }
        )
        variable_rows = [(name, _prepare(_lookup(name))) for name in variable_names]

        print("Steady-state parameters:")
        print(
            tabulate(
                param_rows,
                headers=["Parameter", "Value"],
                tablefmt="github",
                floatfmt="4.3f",
            )
        )
        print("\nSteady-state variables:")
        print(
            tabulate(
                variable_rows,
                headers=["Variable", "Value"],
                tablefmt="github",
                floatfmt="4.3f",
            )
        )

    def _steady_snapshot_path(self):
        settings = get_settings()
        save_dir = settings.paths.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"{self.label}_steady_state.json"

    def _save_steady_snapshot(self):
        path = self._steady_snapshot_path()
        steady = self.steady_dict
        if hasattr(steady, "_asdict"):
            items = steady._asdict().items()
        else:
            items = steady.items()

        serializable = {key: float(value) for key, value in items}
        io.save_json_with_backups(
            serializable,
            path,
            stem=f"{self.label}_steady_state",
        )
        self.steady_dict = {
            key: jnp.asarray(value, dtype=jnp.float64)
            for key, value in serializable.items()
        }

    def compute_derivatives(self, u, x, z, u_new, x_new, z_new, params, E=None):

        if E is None:
            E = self.fcn("expectations", u, x, z, u_new, x_new, z_new, params)

        arg_dict = {
            "transition": (u, x, z, params),
            "expectations": (u, x, z, u_new, x_new, z_new, params),
            "optimality": (u, x, z, E, params),
            "intermediates": (u, x, z, params),
            "expectations_variables": (u, x, z, E, params),
        }

        self.derivatives = {}
        for key, arg_list in self.arg_lists.items():
            self.derivatives[key] = {}
            these_args = arg_dict[key]
            for var in arg_list:
                self.derivatives[key][var] = self.d(key, var, *these_args)

        return None

    def steady_state_derivatives(self):

        init_vals = self.initialize_values(self.steady_dict)
        u, x, z, params = (
            init_vals["u"],
            init_vals["x"],
            init_vals["z"],
            init_vals["params"],
        )
        self.compute_derivatives(u, x, z, u, x, z, params)

    def get_s_steady(self):

        init_vals = self.initialize_values(self.steady_dict)
        u, x, z, params = (
            init_vals["u"],
            init_vals["x"],
            init_vals["z"],
            init_vals["params"],
        )
        E = self.fcn("expectations", u, x, z, u, x, z, params)
        inter = self.fcn("intermediates", u, x, z, params)
        exp_vars = self.fcn("expectations_variables", u, x, z, E, params)

        return np.hstack((u, x, z, inter, E, exp_vars))

    def simulate_linear(self, Nt, s_init=None, shocks=None):

        return self.linear_mod.simulate(Nt, s_init=s_init, shocks=shocks)

    def compute_linear_irfs(
        self,
        Nt_irf,
        *,
        save_irfs: bool = True,
        overwrite: bool = True,
    ):
        """
        Compute impulse response functions for all shocks.

        This method computes IRFs using the linearized model and returns them
        as a dictionary of IrfResult objects, one for each shock. Each IrfResult
        includes the responses of all state variables (UX), exogenous variables (Z),
        and intermediate variables (Y).

        Parameters
        ----------
        Nt_irf : int
            Number of periods for the IRF horizon.
        save_irfs : bool, default True
            If True, save the computed IRFs using ``save_linear_irfs``.
        overwrite : bool, default True
            If True and ``save_irfs`` is enabled, overwrite any existing file.

        Returns
        -------
        dict[str, IrfResult]
            Dictionary mapping shock names to IrfResult objects containing the
            impulse responses for all variables to that shock.

        Notes
        -----
        The full IRF tensor is also stored in ``self.linear_mod.irfs`` for
        backward compatibility.
        """
        irf_dict = self.linear_mod.compute_irfs(Nt_irf)
        if save_irfs:
            self.save_linear_irfs(overwrite=overwrite)
        return irf_dict

    def plot_linear_irfs(
        self,
        include_list=None,
        *,
        plot_type="png",
        title_template="Impulse response to {shock}",
        x_label="Periods",
        n_periods=None,
        shock_sizes=None,
        **plot_kwargs,
    ):
        """
        Render linear impulse response plots for each exogenous shock.

        This is a convenience wrapper around :func:`plot_model_irfs` for plotting
        a single model's IRFs across all shocks.

        Parameters
        ----------
        include_list : Sequence[str], optional
            Variables to display. Defaults to the full state vector ordering.
        plot_type : str, optional
            File extension passed to :func:`plot_paths`. Defaults to ``"png"``.
        title_template : str, optional
            Template used for page titles. Receives ``shock`` keyword.
        x_label : str, optional
            Label for the x-axis. Defaults to ``"Periods"``.
        n_periods : int, optional
            Optional horizon override. If provided and IRFs are already cached,
            the existing responses must have at least ``n_periods`` entries; the
            plots will truncate to the requested length.
        shock_sizes : dict, optional
            Optional per-shock scaling factors applied before plotting.
        **plot_kwargs
            Additional keyword arguments forwarded to :func:`plot_model_irfs`.

        Returns
        -------
        dict[str, list[Path]]
            Mapping from shock name to list of generated plot file paths.

        Raises
        ------
        RuntimeError
            If the model has not been linearized or finalized.
        """
        from ..plot import plot_model_irfs

        if self.linear_mod is None:
            raise RuntimeError("Call linearize() before plotting linear IRFs.")

        # Auto-compute IRFs if missing
        irfs = getattr(self.linear_mod, "irfs", None)
        if irfs is None:
            target = n_periods if n_periods is not None else 50
            logger.info("IRFs missing; computing horizon %d before plotting.", target)
            self.compute_linear_irfs(Nt_irf=target)
        elif n_periods is not None and n_periods > irfs.shape[1]:
            raise ValueError(
                f"Requested n_periods={n_periods} exceeds stored IRF "
                f"horizon={irfs.shape[1]}"
            )

        # Validate include_list
        full_list = getattr(self, "all_vars", None)
        if full_list is None:
            raise RuntimeError("Model variable metadata unavailable; call finalize().")

        if include_list is not None:
            include_list = list(include_list)
            missing = set(include_list) - set(full_list)
            if missing:
                logger.info(
                    "Skipping unknown IRF variables: %s",
                    sorted(missing),
                )
                include_list = [var for var in include_list if var in full_list]
                if not include_list:
                    logger.info("No valid variables remain for IRF plotting; skipping.")
                    return {}

        shock_sizes = shock_sizes or {}

        # Respect caller-provided plot_dir; otherwise use label subdirectory.
        plot_kwargs = dict(plot_kwargs)
        plot_dir = plot_kwargs.pop("plot_dir", None)
        if plot_dir is None:
            from ..settings import get_settings

            settings = get_settings()
            plot_dir = settings.paths.plot_dir / "irfs" / self.label

        # Plot each shock using plot_model_irfs
        results: dict[str, list[Path]] = {}
        for shock in self.exog_list:
            title_str = title_template.format(shock=shock)
            shock_size = float(shock_sizes.get(shock, 1.0))
            prefix = f"{self.label}_irf_to_{shock}"

            paths = plot_model_irfs(
                models=[self],
                shock=shock,
                include_list=include_list,
                plot_dir=plot_dir,
                model_names=None,  # Single model, no legend needed
                prefix=prefix,
                plot_type=plot_type,
                title_str=title_str,
                x_str=x_label,
                n_periods=n_periods,
                shock_size=shock_size,
                **plot_kwargs,
            )
            results[shock] = paths

        return results

    def linearize(self, Phi=None, impact_matrix=None, method="klein"):
        """
        Linearize the model around the steady state.

        Parameters
        ----------
        Phi : array_like, optional
            The steady state transition matrix. The default is None.
        impact_matrix : array_like, optional
            The steady state impact matrix. The default is None.
        method : str, optional
            The method to use for linearization. The default is "klein".
        """

        self.linear_mod.linearize(Phi=Phi, impact_matrix=impact_matrix, method=method)

        # Mark model as linearized
        self._linearized = True

    def save_linear_irfs(
        self,
        filepath=None,
        *,
        format: str = "npz",
        include_matrices: bool = False,
        overwrite: bool = True,
        timestamp: bool = False,
    ):
        """
        Save linear impulse response functions to a file.

        Convenience method that delegates to ``linear_mod.save_irfs()``.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save the IRFs. If None, uses defaults based on model label
            and settings.
        format : str, default "npz"
            Output format. Supported: 'npz', 'csv', 'json'.
        include_matrices : bool, default False
            If True, also save the linearization matrices (A, B, G_x, G_z, H_x, H_z).
        overwrite : bool, default True
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.

        Returns
        -------
        Path
            The path to the saved file.

        Raises
        ------
        RuntimeError
            If the model has not been linearized or no IRFs have been computed.
        """
        if self.linear_mod is None:
            raise RuntimeError("Model has not been linearized. Call linearize() first.")

        return self.linear_mod.save_irfs(
            filepath,
            format=format,
            include_matrices=include_matrices,
            overwrite=overwrite,
            timestamp=timestamp,
        )

    def save_simulation(
        self,
        simulation_data=None,
        filepath=None,
        *,
        format: str = "npz",
        overwrite: bool = False,
        timestamp: bool = False,
    ):
        """
        Save linear simulation results to a file.

        Parameters
        ----------
        simulation_data : array-like, optional
            Simulation data to save. If None, uses cached simulation if available.
        filepath : str or Path, optional
            Path to save the results. If None, uses defaults based on model label
            and settings.
        format : str, default "npz"
            Output format. Supported: 'npz', 'csv', 'json'.
        overwrite : bool, default False
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.

        Returns
        -------
        Path
            The path to the saved file.

        Raises
        ------
        ValueError
            If no simulation data is provided and no cached simulation exists.
        """
        from ..io import resolve_output_path, save_results

        if simulation_data is None:
            raise ValueError(
                "No simulation data provided. Pass simulation_data or run simulate_linear() first."
            )

        # Determine file extension based on format
        suffix_map = {"npz": ".npz", "json": ".json", "csv": ".csv"}
        suffix = suffix_map.get(format, ".npz")

        # Resolve output path
        path = resolve_output_path(
            filepath,
            result_type="simulations",
            model_label=self.label,
            timestamp=timestamp,
            suffix=suffix,
        )

        # Prepare data
        data = {"simulation": np.asarray(simulation_data)}

        # Prepare metadata
        metadata = {
            "model_label": self.label,
            "var_names": getattr(self, "all_vars", []),
        }

        # Save
        return save_results(
            data,
            path,
            format=format,
            metadata=metadata,
            overwrite=overwrite,
        )

    def _apply_transformations(self):
        """
        Apply registered variable transformations to rules and steady_guess.

        This method processes all transformations registered via transform_variables()
        and log_transform(). For each transformation:
        1. Renames variables on LHS to include the prefix
        2. Applies the transformation function to RHS expressions
        3. Replaces variable uses in other rules with inverse transformations
        4. Transforms initial guesses in steady_guess
        """
        if not self._transformations:
            return

        import re

        # Helper functions for regex replacement
        def make_replace_next(inverse_fn, new_var):
            """Create a replacement function for _NEXT variables."""
            return lambda match: f"{inverse_fn}({new_var}_NEXT)"

        def make_replace_current(inverse_fn, new_var):
            """Create a replacement function for current variables."""
            return lambda match: f"{inverse_fn}({new_var})"

        # Process each transformation
        for var_list, transform_fn, inverse_fn, prefix in self._transformations:
            # Build mapping from old variable name to new variable name
            var_mapping = {var: f"{prefix}_{var}" for var in var_list}

            # Update rules for each category
            for category in self.rule_keys:
                if category not in self.rules:
                    continue

                new_rules = MyOrderedDict()
                for rule_name, rule_expr in self.rules[category].items():
                    # Check if this rule defines one of the variables we're transforming
                    if rule_name in var_list:
                        # This is a variable being transformed on the LHS
                        new_var_name = var_mapping[rule_name]

                        # First, replace any references to the transformed variable in its own RHS
                        # This handles cases like ('x', 'x - y') where x appears on both sides
                        new_expr = rule_expr
                        for old_var, new_var in var_mapping.items():
                            # Create patterns for variable matching
                            pattern = (
                                r"\b"
                                + re.escape(old_var)
                                + r"(?!_)(?!\w)"  # Not followed by underscore or word char
                            )
                            pattern_next = r"\b" + re.escape(old_var) + r"_NEXT\b"

                            # Replace var_NEXT with inverse_fn(new_var_NEXT)
                            new_expr = re.sub(
                                pattern_next,
                                make_replace_next(inverse_fn, new_var),
                                new_expr,
                            )

                            # Replace var with inverse_fn(new_var)
                            new_expr = re.sub(
                                pattern,
                                make_replace_current(inverse_fn, new_var),
                                new_expr,
                            )

                        # Then apply transformation to the entire RHS
                        new_expr = f"{transform_fn}({new_expr})"
                        new_rules[new_var_name] = new_expr
                    else:
                        # This rule is not being transformed, but may reference transformed vars
                        # Replace references to old variables with inverse transformation
                        new_expr = rule_expr
                        for old_var, new_var in var_mapping.items():
                            # Create a regex pattern that matches the variable name as a whole word
                            # but not as part of another identifier
                            # We need to be careful about _NEXT suffix and function calls
                            pattern = (
                                r"\b"
                                + re.escape(old_var)
                                + r"(?!_)(?!\w)"  # Not followed by underscore or word char
                            )
                            # Check if the variable appears with _NEXT suffix
                            pattern_next = r"\b" + re.escape(old_var) + r"_NEXT\b"

                            # Replace var_NEXT with inverse_fn(new_var_NEXT)
                            new_expr = re.sub(
                                pattern_next,
                                make_replace_next(inverse_fn, new_var),
                                new_expr,
                            )

                            # Replace var with inverse_fn(new_var), but only if not part of var_NEXT
                            # which we already handled
                            new_expr = re.sub(
                                pattern,
                                make_replace_current(inverse_fn, new_var),
                                new_expr,
                            )

                        new_rules[rule_name] = new_expr

                self.rules[category] = new_rules

            # Update steady_guess: if we have a guess for 'x', transform it to 'log_x'
            for old_var, new_var in var_mapping.items():
                if old_var in self.steady_guess:
                    old_value = self.steady_guess[old_var]
                    # Remove the old variable
                    del self.steady_guess[old_var]
                    # Add the transformed variable with transformed value
                    # Transform the value using a safe mapping of known transformation functions
                    import numpy as np

                    # Map of known transformation function strings to actual functions
                    transform_map = {
                        "np.log": np.log,
                        "np.exp": np.exp,
                        "np.sqrt": np.sqrt,
                        "np.square": np.square,
                        "jnp.log": np.log,
                        "jnp.exp": np.exp,
                        "jnp.sqrt": np.sqrt,
                        "jnp.square": np.square,
                    }

                    if transform_fn in transform_map:
                        try:
                            transformed_value = transform_map[transform_fn](old_value)
                            self.steady_guess[new_var] = transformed_value
                        except Exception as e:
                            logger.warning(
                                f"Could not transform steady_guess for {old_var}: {e}. "
                                f"Please set steady_guess['{new_var}'] manually."
                            )
                    else:
                        logger.warning(
                            f"Unknown transformation function '{transform_fn}' for {old_var}. "
                            f"Please set steady_guess['{new_var}'] manually."
                        )

    def _update_rules(self):

        # Apply variable transformations first, before any other rule processing
        self._apply_transformations()

        # Find variables with _STEADY suffix and handle them
        steady_vars = self.rp.find_steady_vars(self.rules)
        if steady_vars:
            # Store the list of steady variables for later use
            self._steady_vars = steady_vars

            if self.steady_flag:
                # In steady state model, replace x_STEADY with x in all rules at once
                self.core.rules = self.rp.replace_steady_vars(
                    self.rules, self.steady_flag
                )
                # Remove _STEADY parameters since they are replaced in rules
                for var in steady_vars:
                    param_name = f"{var}_STEADY"
                    if param_name in self.params:
                        del self.params[param_name]
            else:
                # In dynamic model, add x_STEADY as parameters with initial value 0.0
                for var in steady_vars:
                    param_name = f"{var}_STEADY"
                    if param_name not in self.params:
                        self.params[param_name] = 0.0

        ignore_vars = (
            list(
                itertools.chain.from_iterable(
                    [
                        self.rules[name].keys()
                        for name in ["transition", "expectations", "optimality"]
                    ]
                )
            )
            + list(self.params.keys())
            + self.exog_list
        )

        for name in ["intermediate", "read_expectations"]:
            self.rules[name] = self.rp.sort_dependencies(self.rules[name], ignore_vars)

        self._get_var_lists()

    def _compile_rules(self):
        """
        Convert rules into a compiled Python module, resolving dependencies.
        """

        functions = []

        for key, fname in [
            ("transition", "transition_inner"),
            ("expectations", "expectations_inner"),
            ("optimality", "optimality_inner"),
        ]:

            args = ["st", "st_new"] if key == "expectations" else ["st"]

            this_fcn = {"name": fname}
            this_fcn["args"] = args
            this_fcn["body"] = []
            this_fcn["returns"] = [
                self.rp.process_rule(rule) for _, rule in self.rules[key].items()
            ]
            this_fcn["return_array"] = True
            this_fcn["return_mutate"] = False

            functions.append(this_fcn)

        for key, name in [
            ("intermediate", "intermediate_variables"),
            ("read_expectations", "read_expectations_variables"),
        ]:

            ignore_vars = self.var_lists[key]

            this_fcn = {"name": f"{name}"}
            this_fcn["args"] = ["st"]
            this_fcn["body"] = [
                f"{var} = {self.rp.process_rule(rule, ignore_vars=ignore_vars)}"
                for var, rule in self.rules[key].items()
            ]
            this_fcn["returns"] = list(self.rules[key].keys())
            this_fcn["return_array"] = False
            this_fcn["return_mutate"] = True

            functions.append(this_fcn)

            # array version
            this_fcn = {"name": f"{name}_array"}
            this_fcn["args"] = ["st"]
            this_fcn["body"] = [f"st = {name}(st)"]
            this_fcn["returns"] = [f"st.{var}" for var in self.rules[key].keys()]
            this_fcn["return_array"] = True
            this_fcn["return_mutate"] = False

            functions.append(this_fcn)

        cg = CodeGenerator(
            jit=False,
            resolve_debug_dir=not self.steady_flag,
        )

        module_name = f"inner_functions_{self.label}"
        self.inner_functions = cg.compile_module(
            functions,
            module_name,
            core_vars=self.core_vars,
            derived_vars=self.derived_vars,
            display_source=False,
        )

        return None

    def _get_var_lists(self):
        """
        Construct lists of variables for different model roles based on rules.
        """

        mapping = {
            "u": "optimality",
            "x": "transition",
            "E": "expectations",
            "intermediate": "intermediate",
            "read_expectations": "read_expectations",
        }

        self.var_lists = {
            key: list(self.rules[val].keys()) for key, val in mapping.items()
        }

        self.var_lists["err"] = ["err_" + var for var in self.var_lists["u"]]
        self.var_lists["z"] = self.exog_list
        self.var_lists["params"] = sorted(
            list(set(self.params.keys()) - set(self.var_lists["u"]))
        )

        # Get list of all variables
        self.all_vars = (
            self.var_lists["u"]
            + self.var_lists["x"]
            + self.var_lists["z"]
            + self.var_lists["intermediate"]
            + self.var_lists["E"]
            + self.var_lists["read_expectations"]
        )

        self.core_vars = list(
            itertools.chain.from_iterable(
                [self.var_lists[name] for name in self.core_var_types]
            )
        )

        # TODO: should probably have this be all_vars
        self.derived_vars = (
            self.var_lists["intermediate"] + self.var_lists["read_expectations"]
        )

    def to_dynare(
        self, output_path: str | Path | None = None, steady: bool = False
    ) -> str:
        """
        Export model to Dynare .mod file format.

        This is a convenience wrapper around io.dynare.export_to_dynare().
        For detailed documentation, see that function.

        Parameters
        ----------
        output_path : str, Path, or None
            Path to write the .mod file. If None, writes to debug directory
            as <Model.label>.mod (e.g., "_default.mod").
        steady : bool, optional
            If True, add a "steady;" command after the initval block to compute
            the steady state in Dynare. Default is False.

        Returns
        -------
        str
            The generated Dynare .mod file content.

        Raises
        ------
        RuntimeError
            If the model has not been finalized.

        Examples
        --------
        >>> model.finalize()
        >>> # Write to default location (debug_dir/<label>.mod)
        >>> dynare_code = model.to_dynare()
        >>> # Write to specific location
        >>> dynare_code = model.to_dynare(output_path="my_model.mod")
        >>> # Include steady state computation
        >>> dynare_code = model.to_dynare(steady=True)
        """
        from ..io.dynare import export_to_dynare

        return export_to_dynare(self, output_path=output_path, steady=steady)
