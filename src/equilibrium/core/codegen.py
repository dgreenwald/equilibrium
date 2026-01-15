import logging
import types
from importlib import resources
from pathlib import Path

from black import FileMode, format_str
from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)


def get_env() -> Environment:
    """
    Create and configure a Jinja2 environment for template rendering.

    Returns
    -------
    Environment
        Configured Jinja2 environment with template loader.
    """
    pkg_path = resources.files("equilibrium.templates")
    loader = FileSystemLoader(str(pkg_path))
    return Environment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,  # fail on missing vars
    )


def _string_to_module(code, module_name):
    """
    Dynamically create a Python module from source code string.

    Parameters
    ----------
    code : str
        Python source code to execute.
    module_name : str
        Name for the dynamically created module.

    Returns
    -------
    module
        Python module object with code executed in its namespace.
    """
    module = types.ModuleType(module_name)
    exec(code, module.__dict__)
    return module


def _render_code(functions, core_vars, derived_vars, jit=True):
    """
    Render Python code from templates using function definitions.

    Parameters
    ----------
    functions : list[dict]
        List of function specifications with name, args, body, returns.
    core_vars : list[str]
        Core model variables.
    derived_vars : list[str]
        Derived/intermediate model variables.
    jit : bool, optional
        Whether to apply @jax.jit decorators. Default is True.

    Returns
    -------
    str
        Formatted Python source code.
    """
    env = get_env()
    tmpl = env.get_template("functions.py.jinja")
    code = tmpl.render(
        funcs=functions, core_vars=core_vars, derived_vars=derived_vars, jit=jit
    )
    return format_str(code, mode=FileMode())


class CodeGenerator:
    """
    Generates Python source code for DGE model functions from symbolic rules.
    """

    def __init__(
        self,
        jit: bool = True,
        debug_dir: str | Path | None = None,
        *,
        resolve_debug_dir: bool = True,
    ):
        self.jit = jit

        if debug_dir is None and resolve_debug_dir:
            # Avoid importing settings unless necessary to keep module load light.
            from ..settings import get_settings

            debug_dir = get_settings().paths.debug_dir

        self.debug_dir = Path(debug_dir).expanduser() if debug_dir else None

    def compile_module(
        self,
        functions: list[dict],
        module_name: str,
        core_vars: list[str] = None,
        derived_vars: list[str] = None,
        display_source: bool = False,
    ):
        """
        Compile the generated code into an executable Python module.

        Parameters
        ----------
        functions: list
            dictionaries with function information in jinja format
        module_name : str
            Name of the Python module to create.
        display_source: bool
            Set to True to display the generated code on screen

        Returns
        -------
        module : module
            Compiled Python module.
        """

        if core_vars is None:
            core_vars = []
        if derived_vars is None:
            derived_vars = []

        code = _render_code(functions, core_vars, derived_vars, jit=self.jit)
        if display_source:
            logger.info("Generated code for module %s:\n%s", module_name, code)

        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / f"{module_name}.py").write_text(code)

        return _string_to_module(code, module_name)
