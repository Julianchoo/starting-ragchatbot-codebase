#!/usr/bin/env python3
"""
Development script for formatting Python code.
Runs black and isort on the codebase.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main() -> int:
    """Main formatting function."""
    project_root = Path(__file__).parent.parent

    commands = [
        (["uv", "run", "black", ".", "--check", "--diff"], "black formatting check"),
        (
            ["uv", "run", "isort", ".", "--check-only", "--diff"],
            "isort import sorting check",
        ),
    ]

    if "--fix" in sys.argv:
        commands = [
            (["uv", "run", "black", "."], "black code formatting"),
            (["uv", "run", "isort", "."], "isort import sorting"),
        ]

    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False

    if success:
        print("All formatting checks passed!")
        return 0
    else:
        print("Formatting issues found. Run with --fix to auto-fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
