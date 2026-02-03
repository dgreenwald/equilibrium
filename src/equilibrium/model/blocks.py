"""Model block classes for modular model construction.

This module provides base classes and decorators for creating reusable model
components (blocks) that can be combined to build economic models.
"""

import re
from functools import wraps

from ..utils.containers import MyOrderedDict, PresetDict
from ..utils.utilities import initialize_if_none
from .constants import RULE_KEYS


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

    def with_suffix(
        self,
        suffix: str,
        variables: set[str],
        suffix_before: list[str] | str | None = None,
    ):
        """Apply suffix to specific variables using word-boundary matching.

        This method appends a suffix to variables in the set, using regex word
        boundaries to match complete identifiers. The special _NEXT suffix is
        preserved (VAR_NEXT becomes VAR<suffix>_NEXT).

        Parameters
        ----------
        suffix : str
            The suffix to append to variable names.
        variables : set[str]
            Set of variable names to suffix (typically LHS variables from rules).
        suffix_before : list[str] or str, optional
            Additional terms (besides _NEXT) to insert suffix before. For example,
            if suffix_before=['_AGENT'] and suffix='_firm', then 'C_AGENT' becomes
            'C_firm_AGENT' rather than 'C_AGENT_firm'. Useful with placeholder
            variables that will be renamed later. Accepts a single string or list.

        Returns
        -------
        BaseModelBlock
            New block with suffixed variables.

        Examples
        --------
        If variables = {'K', 'Y'} and suffix = '_firm':
        - 'K' -> 'K_firm'
        - 'K_NEXT' -> 'K_firm_NEXT'
        - 'Y' -> 'Y_firm'
        - 'log_K' -> 'log_K' (K not a complete identifier)
        - 'K_new' -> 'K_new' (different variable, not in the set)

        If variables = {'C', 'I'}, suffix = '_firm', suffix_before = ['_AGENT']:
        - 'C_AGENT' -> 'C_firm_AGENT'
        - 'I' -> 'I_firm'
        """

        if not suffix or not variables:
            return self

        # Normalize suffix_before to a list
        if suffix_before is None:
            suffix_before_list = []
        elif isinstance(suffix_before, str):
            suffix_before_list = [suffix_before]
        elif isinstance(suffix_before, list):
            suffix_before_list = suffix_before
        else:
            raise TypeError(
                f"suffix_before must be a list, str, or None, got {type(suffix_before)}"
            )

        # Sort by length (longest first) to handle cases like 'K' and 'log_K' correctly
        sorted_vars = sorted(variables, key=len, reverse=True)

        # Build list of special terms to insert suffix before
        # Always include _NEXT, plus any user-specified terms
        special_terms = suffix_before_list + ["_NEXT"]

        # Create regex pattern for matching special term sequences at end of variable
        if special_terms:
            terms_pattern = "(?:" + "|".join(re.escape(t) for t in special_terms) + ")*"
        else:
            terms_pattern = ""

        # Pre-compute suffixed forms for each variable
        var_to_suffixed = {}
        for var in sorted_vars:
            # Analyze variable to determine where suffix goes
            # Pattern: variable = (prefix)(special_terms_sequence)
            pattern = r"^(\w*?)(" + terms_pattern + r")$"
            match = re.match(pattern, var)
            if match:
                # Insert suffix before special terms (if any)
                suffixed_var = match.group(1) + suffix + match.group(2)
            else:
                # Shouldn't happen with the pattern above, but fallback to append
                suffixed_var = var + suffix
            var_to_suffixed[var] = suffixed_var

        def apply_suffix(text: str) -> str:
            """Apply suffix to variables in text using word boundaries."""
            result = text
            for var in sorted_vars:
                suffixed_var = var_to_suffixed[var]

                # Handle temporal _NEXT: VAR_NEXT -> suffixed_VAR_NEXT
                # This catches cases where _NEXT appears after the variable in expressions
                result = re.sub(
                    r"\b" + re.escape(var) + r"_NEXT\b", suffixed_var + "_NEXT", result
                )

                # Handle variable itself (but not when followed by _NEXT)
                result = re.sub(
                    r"\b" + re.escape(var) + r"\b(?!_NEXT)", suffixed_var, result
                )

            return result

        # Apply suffix to flags
        new_flags = {}
        for key, value in self.flags.items():
            new_key = apply_suffix(key)
            if new_key in new_flags:
                raise ValueError(
                    f"Suffix results in duplicate keys for flags: '{new_key}'"
                )
            new_flags[new_key] = value

        # Apply suffix to params and steady_guess
        new_params = {}
        for key, value in self.params.items():
            new_key = apply_suffix(key)
            if new_key in new_params:
                raise ValueError(
                    f"Suffix results in duplicate keys for params: '{new_key}'"
                )
            new_params[new_key] = value

        new_steady = {}
        for key, value in self.steady_guess.items():
            new_key = apply_suffix(key)
            if new_key in new_steady:
                raise ValueError(
                    f"Suffix results in duplicate keys for steady_guess: '{new_key}'"
                )
            new_steady[new_key] = value

        # Apply suffix to exog_list
        new_exog = []
        for exog in self.exog_list:
            new_exog_name = apply_suffix(exog)
            if new_exog_name in new_exog:
                raise ValueError(
                    f"Suffix results in duplicate exogenous variables: '{new_exog_name}'"
                )
            new_exog.append(new_exog_name)

        # Apply suffix to rules (both LHS and RHS)
        new_rules = {}
        for key in self.rule_keys:
            od = MyOrderedDict()
            for rule_name, expression in self.rules.get(key, {}).items():
                new_name = apply_suffix(rule_name)
                if new_name in od:
                    raise ValueError(
                        f"Suffix results in duplicate rule names in '{key}': '{new_name}'"
                    )
                new_expr = apply_suffix(expression)
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

    @staticmethod
    def _get_lhs_variables(block: "BaseModelBlock") -> set[str]:
        """Extract all LHS variable names from block rules and exog_list.

        Parameters
        ----------
        block : BaseModelBlock
            Block to extract variable names from.

        Returns
        -------
        set[str]
            Set of all variable names that appear on the left-hand side of
            rule assignments or in the exog_list.
        """
        lhs_vars = set()
        for rule_category in block.rules.values():
            for var_name, _expr in rule_category.items():
                lhs_vars.add(var_name)
        lhs_vars.update(block.exog_list)
        return lhs_vars

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
        suffix: str | None = None,
        suffix_before: list[str] | str | None = None,
        exclude_vars: set[str] | list[str] | None = None,
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
            Mapping of substring replacements to apply after suffix (if provided).
        suffix : str, optional
            Suffix to append to all variables created in the block (LHS variables).
            Uses word-boundary matching so 'K' and 'log_K' are independent.
            Special handling: VAR_NEXT becomes VAR<suffix>_NEXT. Applied before
            ``rename`` if both are provided.
        suffix_before : list[str] or str, optional
            Additional terms (besides _NEXT) to insert suffix before. For example,
            if suffix_before=['_AGENT'] and suffix='_firm', then 'C_AGENT' becomes
            'C_firm_AGENT' rather than 'C_AGENT_firm'. Useful with placeholder
            variables that will be renamed later. Applied during suffix phase,
            before ``rename``. Default is None (no additional terms).
        exclude_vars : set[str] or list[str], optional
            Set or list of variable names (LHS variables) to exclude from the
            incoming block. Excluded variables will not be added to rules or
            exog_list. Does not affect params, steady_guess, or flags.
            Applied after suffix and rename transformations, so exclusions
            should use the final transformed variable names.
            Default is None (no exclusion).

        Returns
        -------
        BaseModelBlock
            Returns self to allow method chaining.

        Notes
        -----
        When both ``exclude_vars`` and ``overwrite`` are specified, exclusion
        takes precedence. Variables in ``exclude_vars`` will never be added,
        regardless of the ``overwrite`` setting.

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

        # Apply suffix first (if provided)
        if suffix:
            lhs_vars = self._get_lhs_variables(block)
            block = block.with_suffix(suffix, lhs_vars, suffix_before=suffix_before)

        # Then apply rename (if provided)
        if rename:
            block = block.with_replacements(rename)

        # Normalize exclude_vars to a set for O(1) lookup
        if exclude_vars is None:
            exclude_vars_set = set()
        elif isinstance(exclude_vars, set):
            exclude_vars_set = exclude_vars
        elif isinstance(exclude_vars, list):
            exclude_vars_set = set(exclude_vars)
        else:
            raise TypeError(
                f"exclude_vars must be a set, list, or None, got {type(exclude_vars)}"
            )

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
                # Skip excluded variables
                if exog in exclude_vars_set:
                    continue
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
                # Skip excluded variables
                if rule_name in exclude_vars_set:
                    continue
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
    :const:`RULE_KEYS`. For custom rule keys, use :class:`BaseModelBlock`.
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
            rule_keys=RULE_KEYS,
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

    @wraps(func)
    def wrapper(**kwargs):
        block = ModelBlock()
        func(block, **kwargs)
        return block

    return wrapper
