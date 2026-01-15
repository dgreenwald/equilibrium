"""Main CLI entry point for equilibrium."""

import argparse
import sys

from .scaffold import scaffold_project


def main():
    """Main entry point for the equilibrium CLI."""
    parser = argparse.ArgumentParser(
        prog="equilibrium",
        description="Equilibrium: Dynamic general-equilibrium solver in JAX",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new equilibrium project",
        description="Create a new project with a working RBC example",
    )
    init_parser.add_argument(
        "project_name",
        help="Name of the project directory to create",
    )

    args = parser.parse_args()

    if args.command == "init":
        scaffold_project(args.project_name)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
