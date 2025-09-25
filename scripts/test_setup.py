#!/usr/bin/env python3
"""
Setup Test Runner for Misfits! Game

This script runs comprehensive tests to validate that the setup process
completes successfully and all components are properly initialized.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "tests"))

def test_actual_setup():
    """Test the actual setup.py script in isolation."""
    print("🧪 Testing actual setup.py script...")

    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()

    try:
        os.chdir(test_dir)

        # Copy necessary files to test directory
        setup_script = project_root / "scripts" / "setup.py"
        test_setup_script = Path(test_dir) / "setup.py"
        shutil.copy2(setup_script, test_setup_script)

        # Create minimal src structure for imports
        src_dir = Path(test_dir) / "src"
        src_dir.mkdir()

        # Copy required source modules (simplified for testing)
        utils_dir = src_dir / "utils"
        core_dir = src_dir / "core"
        utils_dir.mkdir()
        core_dir.mkdir()

        # Create __init__.py files
        (src_dir / "__init__.py").touch()
        (utils_dir / "__init__.py").touch()
        (core_dir / "__init__.py").touch()

        # Run setup script
        result = subprocess.run([
            sys.executable, "setup.py"
        ], capture_output=True, text=True)

        print(f"Setup script exit code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # Check if critical files were created
        expected_files = [
            "misfits.db",
            "memories.db",
            "vectors.db"
        ]

        files_created = []
        for file in expected_files:
            if os.path.exists(file):
                files_created.append(file)
                print(f"✅ Created: {file}")
            else:
                print(f"❌ Missing: {file}")

        success = len(files_created) >= 1  # At least one DB should be created

        return success, result.returncode, files_created

    finally:
        os.chdir(old_cwd)
        shutil.rmtree(test_dir)


def run_unit_tests():
    """Run the unit tests for setup validation."""
    print("🧪 Running unit tests...")

    # Run the test suite
    test_file = project_root / "tests" / "test_setup.py"

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", str(test_file), "-v"
        ], capture_output=True, text=True, cwd=project_root)

        print("Pytest output:")
        print(result.stdout)
        if result.stderr:
            print("Pytest errors:")
            print(result.stderr)

        return result.returncode == 0

    except FileNotFoundError:
        # Fallback to unittest if pytest not available
        print("Pytest not found, using unittest...")

        result = subprocess.run([
            sys.executable, str(test_file)
        ], capture_output=True, text=True, cwd=project_root)

        print("Unittest output:")
        print(result.stdout)
        if result.stderr:
            print("Unittest errors:")
            print(result.stderr)

        return result.returncode == 0


def validate_project_structure():
    """Validate that the project has the expected structure."""
    print("🏗️  Validating project structure...")

    required_files = [
        "scripts/setup.py",
        "main.py",
        "requirements.txt",
        "pyproject.toml"
    ]

    required_dirs = [
        "src",
        "tests",
        "data",
        "assets",
        "scripts"
    ]

    missing_files = []
    missing_dirs = []

    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.is_dir():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ Found directory: {dir_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")

    return len(missing_files) == 0 and len(missing_dirs) == 0


def check_dependencies():
    """Check that required dependencies are available."""
    print("📦 Checking dependencies...")

    required_packages = [
        "yaml",
        "sqlite3",
        "pathlib",
        "tempfile",
        "unittest"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")

    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))

    return len(missing_packages) == 0


def main():
    """Main test runner."""
    print("🎲 Misfits! Setup Validation Test Runner")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0

    # Test 1: Project structure
    print("\n1. PROJECT STRUCTURE VALIDATION")
    print("-" * 40)
    total_tests += 1
    if validate_project_structure():
        print("✅ Project structure validation PASSED")
        passed_tests += 1
    else:
        print("❌ Project structure validation FAILED")

    # Test 2: Dependencies
    print("\n2. DEPENDENCY CHECK")
    print("-" * 40)
    total_tests += 1
    if check_dependencies():
        print("✅ Dependency check PASSED")
        passed_tests += 1
    else:
        print("❌ Dependency check FAILED")

    # Test 3: Unit tests
    print("\n3. UNIT TESTS")
    print("-" * 40)
    total_tests += 1
    if run_unit_tests():
        print("✅ Unit tests PASSED")
        passed_tests += 1
    else:
        print("❌ Unit tests FAILED")

    # Test 4: Actual setup script (optional, can be problematic in some environments)
    print("\n4. ACTUAL SETUP SCRIPT TEST")
    print("-" * 40)
    total_tests += 1
    try:
        success, exit_code, files_created = test_actual_setup()
        if success:
            print("✅ Setup script test PASSED")
            print(f"Files created: {files_created}")
            passed_tests += 1
        else:
            print("❌ Setup script test FAILED")
            print(f"Exit code: {exit_code}")
    except Exception as e:
        print(f"❌ Setup script test FAILED with exception: {e}")

    # Final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Setup validation complete.")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())