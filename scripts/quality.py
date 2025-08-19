#!/usr/bin/env python3
"""
Development script for running all code quality checks.
Combines formatting, linting, and testing.
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list[str] = None) -> bool:
    """Run a Python script and return success status."""
    if args is None:
        args = []

    script_path = Path(__file__).parent / script_name
    command = [sys.executable, str(script_path)] + args

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_tests() -> bool:
    """Run pytest tests."""
    print("Running tests...")
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "backend/tests/"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error running tests:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main() -> int:
    """Main quality check function."""
    print("Running comprehensive code quality checks...\n")

    # Check if --fix flag is provided
    fix_mode = "--fix" in sys.argv
    format_args = ["--fix"] if fix_mode else []

    checks = [
        ("format.py", format_args, "Code formatting"),
        ("lint.py", [], "Code linting"),
        (None, [], "Tests"),  # Special case for tests
    ]

    all_passed = True

    for script, args, description in checks:
        print(f"\n{description}:")
        print("-" * 40)

        if script is None:  # Tests
            success = run_tests()
        else:
            success = run_script(script, args)

        if not success:
            all_passed = False
            print(f"{description} failed!")
        else:
            print(f"{description} passed!")

    print("\n" + "=" * 50)
    if all_passed:
        print("All quality checks passed! Code is ready.")
        return 0
    else:
        print("Some quality checks failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
