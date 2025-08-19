#!/usr/bin/env python3
"""
Development script for linting Python code.
Runs flake8 and mypy on the codebase.
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
        print(f"Error in {description}:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main() -> int:
    """Main linting function."""
    project_root = Path(__file__).parent.parent

    commands = [
        (["uv", "run", "flake8", "backend/", "main.py", "scripts/"], "flake8 linting"),
        (
            ["uv", "run", "mypy", "backend/", "main.py", "scripts/"],
            "mypy type checking",
        ),
    ]

    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False

    if success:
        print("All linting checks passed!")
        return 0
    else:
        print("Linting issues found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
