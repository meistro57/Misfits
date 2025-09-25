#!/usr/bin/env python3
"""
Test Runner for Misfits! Game

This script runs the test suite with various options and configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_pytest(args):
    """Run pytest with given arguments."""
    cmd = ["python", "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_unit_tests():
    """Run unit tests."""
    print("🧪 Running Unit Tests")
    print("=" * 50)
    return run_pytest([
        "tests/unit",
        "-v",
        "--tb=short"
    ])


def run_integration_tests():
    """Run integration tests."""
    print("🔗 Running Integration Tests")
    print("=" * 50)
    return run_pytest([
        "tests/integration",
        "-v",
        "--tb=short",
        "-m", "not slow"
    ])


def run_ai_tests():
    """Run AI-related tests (requires LLM providers)."""
    print("🤖 Running AI Tests")
    print("=" * 50)
    return run_pytest([
        "tests",
        "-v",
        "--tb=short",
        "-m", "ai"
    ])


def run_all_tests():
    """Run all tests."""
    print("🎯 Running All Tests")
    print("=" * 50)
    return run_pytest([
        "tests",
        "-v",
        "--tb=short"
    ])


def run_fast_tests():
    """Run only fast tests."""
    print("⚡ Running Fast Tests")
    print("=" * 50)
    return run_pytest([
        "tests",
        "-v",
        "--tb=short",
        "-m", "not slow and not ai"
    ])


def run_coverage():
    """Run tests with coverage reporting."""
    print("📊 Running Tests with Coverage")
    print("=" * 50)
    return run_pytest([
        "tests",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])


def run_performance_tests():
    """Run performance/benchmark tests."""
    print("🏃 Running Performance Tests")
    print("=" * 50)
    return run_pytest([
        "tests/performance",
        "-v",
        "--tb=short"
    ])


def run_linting():
    """Run code linting."""
    print("🔍 Running Linting")
    print("=" * 50)

    # Run flake8
    print("Running flake8...")
    result = subprocess.run(["flake8", "src", "tests"], cwd=project_root)
    if result.returncode != 0:
        return result.returncode

    # Run black check
    print("Running black check...")
    result = subprocess.run(["black", "--check", "src", "tests"], cwd=project_root)
    if result.returncode != 0:
        print("Code formatting issues found. Run 'black src tests' to fix.")
        return result.returncode

    print("✓ All linting checks passed!")
    return 0


def run_type_checking():
    """Run type checking with mypy."""
    print("🔬 Running Type Checking")
    print("=" * 50)

    result = subprocess.run(["mypy", "src"], cwd=project_root)
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Misfits! Test Runner")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--ai", action="store_true",
                       help="Run AI tests (requires LLM providers)")
    parser.add_argument("--fast", action="store_true",
                       help="Run only fast tests")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage reporting")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests")
    parser.add_argument("--lint", action="store_true",
                       help="Run linting checks")
    parser.add_argument("--types", action="store_true",
                       help="Run type checking")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests and checks")
    parser.add_argument("--pytest-args", type=str,
                       help="Additional arguments to pass to pytest")

    args = parser.parse_args()

    # If no specific option is chosen, run fast tests by default
    if not any([args.unit, args.integration, args.ai, args.fast,
                args.coverage, args.performance, args.lint, args.types, args.all]):
        args.fast = True

    exit_code = 0

    try:
        if args.all:
            # Run everything
            exit_code |= run_linting()
            exit_code |= run_type_checking()
            exit_code |= run_coverage()
            exit_code |= run_performance_tests()

        else:
            if args.lint:
                exit_code |= run_linting()

            if args.types:
                exit_code |= run_type_checking()

            if args.unit:
                exit_code |= run_unit_tests()

            if args.integration:
                exit_code |= run_integration_tests()

            if args.ai:
                exit_code |= run_ai_tests()

            if args.fast:
                exit_code |= run_fast_tests()

            if args.coverage:
                exit_code |= run_coverage()

            if args.performance:
                exit_code |= run_performance_tests()

            # Handle custom pytest arguments
            if args.pytest_args:
                custom_args = args.pytest_args.split()
                exit_code |= run_pytest(custom_args)

        if exit_code == 0:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")

        return exit_code

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())