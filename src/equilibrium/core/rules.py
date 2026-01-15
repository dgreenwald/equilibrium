import logging
import re

import networkx as nx

from ..utils.containers import MyOrderedDict

logger = logging.getLogger(__name__)


class RuleProcessor:
    """
    Handles rule transformation and dependency sorting for DGE model code generation.
    """

    def __init__(self):
        self.p_next = re.compile(
            r"\b(?!\w*\.\w*)(?!st|st_new)(?<!\.)([a-zA-Z]\w*)_NEXT(?!')(?!\s*\()\b"
        )
        self.p_word = re.compile(
            r"\b(?!\w*\.\w*)(?!\w*_NEXT)(?!st|st_new)(?<!\.)([a-zA-Z]\w*)(?!')(?!\s*\()\b"
        )
        # Pattern to match variables with _STEADY suffix and capture the base variable name
        # Group 1 captures the base variable name (e.g., "c" from "c_STEADY")
        self.p_steady = re.compile(r"\b([a-zA-Z]\w*)_STEADY\b")

    def find_steady_vars(self, rules):
        """
        Find all variables with _STEADY suffix in the given rules.

        Parameters
        ----------
        rules : dict
            Dictionary of rule categories containing rule expressions.

        Returns
        -------
        set[str]
            Set of base variable names (without _STEADY suffix) that have
            _STEADY references in the rules.
        """
        steady_vars = set()
        for category_rules in rules.values():
            if isinstance(category_rules, dict):
                for rule in category_rules.values():
                    if isinstance(rule, str):
                        matches = self.p_steady.findall(rule)
                        steady_vars.update(matches)
        return steady_vars

    def replace_steady_vars(self, rules, steady_flag):
        """
        Replace _STEADY suffix variables in rules.

        When steady_flag is True, replaces "x_STEADY" with "x".
        When steady_flag is False, keeps "x_STEADY" as is (it will be a parameter).

        Parameters
        ----------
        rules : dict
            Dictionary of rule categories.
        steady_flag : bool
            Whether this is a steady state version of the model.

        Returns
        -------
        dict
            Updated rules dictionary with _STEADY variables replaced if steady_flag is True.
        """
        if not steady_flag:
            # In the dynamic model, keep _STEADY variables as parameters
            return rules

        # In the steady state model, replace x_STEADY with x
        new_rules = {}
        for category, category_rules in rules.items():
            if isinstance(category_rules, dict):
                new_category = type(category_rules)()
                for var_name, rule in category_rules.items():
                    if isinstance(rule, str):
                        new_rule = self.p_steady.sub(r"\1", rule)
                    else:
                        new_rule = rule
                    new_category[var_name] = new_rule
                new_rules[category] = new_category
            else:
                new_rules[category] = category_rules
        return new_rules

    def process_rule(self, rule, ignore_vars=None):
        """
        Replace variable names in a rule string with dictionary lookups for st and st_new.
        """
        ignore = set(ignore_vars or [])

        def replace_next(match):
            var = match.group(1)
            return var if var in ignore else f"st_new.{var}"

        # Replace normal variables unless ignored
        def replace_word(match):
            var = match.group(1)
            return var if var in ignore else f"st.{var}"

        rule = self.p_next.sub(replace_next, rule)
        rule = self.p_word.sub(replace_word, rule)
        return rule

    def update_graph(self, G, var, rule, ignore_vars):
        """
        Add a variable and its dependencies to the graph G for topological sorting.
        """
        G.add_node(var)
        all_dependencies = re.findall(self.p_word, rule)
        dependencies_to_graph = set(all_dependencies) - set(ignore_vars)
        for dep in dependencies_to_graph:
            G.add_edge(dep, var)
        return G

    def sort_dependencies(self, rules, ignore_vars=None):
        """
        Sort the rules topologically to respect dependencies. Raises if cycles are found.

        Parameters
        ----------
        rules : dict
            Dictionary of variable rules to sort.
        ignore_vars : list, optional
            List of variable names to exclude from dependency edges.

        Returns
        -------
        sorted_rules : MyOrderedDict
            Ordered dictionary of rules sorted by dependency.
        """
        ignore_vars = set(ignore_vars or [])
        G = nx.DiGraph()
        for var, rule in rules.items():
            G = self.update_graph(G, var, rule, ignore_vars)

        cycles = list(nx.simple_cycles(G))
        if cycles:
            logger.error("Cycles detected in rule dependencies.")
            for ii, cycle in enumerate(cycles):
                logger.error("Cycle %d: %s", ii, cycle)
            raise Exception("Cyclic dependencies found in rules")

        sorted_vars = list(nx.topological_sort(G))
        return MyOrderedDict([(key, rules[key]) for key in sorted_vars])

    def get_steady_rules(self, rules, calibrate=True):
        """
        Update rules to their steady state versions.
        Includes replacing policy variables with analytical steady state solutions
        and optionally adding calibrated parameters as policy variables.

        Parameters
        ----------
        rules : dict
            Rules to be updated.
        calibrate : bool
            Whether to include calibrated parameters. Default is True.

        Returns
        -------
        rules_steady : dict
            Updated rules dictionary.
        """
        rules_steady = rules.copy()
        if calibrate:
            rules_steady["optimality"] += rules_steady["calibration"]
        if rules_steady["analytical_steady"]:
            for name in ["transition", "optimality"]:
                rules_steady[name] = MyOrderedDict(
                    [
                        (key, val)
                        for key, val in rules_steady[name].items()
                        if key not in rules_steady["analytical_steady"]
                    ]
                )
            rules_steady["intermediate"] += rules_steady["analytical_steady"]
        return rules_steady
