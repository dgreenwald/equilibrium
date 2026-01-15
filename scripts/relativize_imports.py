#!/usr/bin/env python
"""
Rewrite  'from solvers.klein import solve'
into     'from .solvers.klein import solve'

and     'import solvers.klein as kl'
into     'from .solvers import klein as kl'
"""

import pathlib

import libcst as cst

INTERNAL = {"core", "model", "solvers", "utils"}


class Relativizer(cst.CSTTransformer):
    def leave_ImportFrom(self, node, updated):
        if node.relative:  # already relative
            return updated
        if node.module and node.module.value.split(".")[0] in INTERNAL:
            return updated.with_changes(relative=cst.Dot())
        return updated

    def leave_Import(self, node, updated):
        # single-name 'import solvers.klein as kl' → 'from .solvers import klein as kl'
        if len(node.names) == 1:
            name = node.names[0].name
            if isinstance(name, cst.Attribute):
                head = name.value.value  # first component
                if head in INTERNAL:
                    new_from = cst.ImportFrom(
                        module=cst.Name(
                            name.attr.value if isinstance(name.attr, cst.Name) else ""
                        ),
                        names=[cst.ImportAlias(name=name.attr)],
                        relative=cst.Dot(),
                    )
                    # preserve any alias  "as xyz"
                    return new_from.with_changes(
                        names=[
                            cst.ImportAlias(name=name.attr, asname=node.names[0].asname)
                        ]
                    )
        return updated


def process_file(path: pathlib.Path):
    code = path.read_text()
    mod = cst.parse_module(code)
    new = mod.visit(Relativizer())
    path.write_text(new.code)


if __name__ == "__main__":
    root = pathlib.Path("src/equilibrium")
    for py in root.rglob("*.py"):
        process_file(py)
    print("✅ Done.  Review with  git diff  then commit.")
