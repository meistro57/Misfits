#!/usr/bin/env python3
"""
Simple Setup Validation Script for Misfits! Game

This script performs quick validation that setup completed successfully.
"""

import os
import sqlite3
import sys
from pathlib import Path

def validate_setup():
    """Validate that setup completed successfully."""
    print("🔍 Validating Misfits! setup...")

    validation_results = []

    # Check database files
    required_databases = ["misfits.db", "memories.db", "vectors.db"]
    for db_name in required_databases:
        if os.path.exists(db_name):
            print(f"✅ Database found: {db_name}")

            # Quick database validation
            if db_name == "misfits.db":
                try:
                    conn = sqlite3.connect(db_name)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]

                    required_tables = ['game_state', 'character_profiles', 'world_events', 'settings']
                    missing_tables = [t for t in required_tables if t not in tables]

                    if not missing_tables:
                        print(f"✅ All required tables present in {db_name}")
                        validation_results.append(True)
                    else:
                        print(f"❌ Missing tables in {db_name}: {missing_tables}")
                        validation_results.append(False)

                    conn.close()
                except Exception as e:
                    print(f"❌ Error validating {db_name}: {e}")
                    validation_results.append(False)
            else:
                validation_results.append(True)
        else:
            print(f"❌ Database missing: {db_name}")
            validation_results.append(False)

    # Check configuration directories
    config_dirs = [
        "data/personalities",
        "data/simulation_modes",
        "data/events",
        "logs"
    ]

    for dir_path in config_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ Directory found: {dir_path}")
            validation_results.append(True)
        else:
            print(f"❌ Directory missing: {dir_path}")
            validation_results.append(False)

    # Check simulation mode configs were created
    mode_configs = [
        "data/simulation_modes/learning_growth.yaml",
        "data/simulation_modes/sandbox.yaml"
    ]

    for config_file in mode_configs:
        if os.path.exists(config_file):
            print(f"✅ Configuration file: {config_file}")
            validation_results.append(True)
        else:
            print(f"❌ Configuration missing: {config_file}")
            validation_results.append(False)

    # Summary
    passed = sum(validation_results)
    total = len(validation_results)

    print(f"\n📊 Validation Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 Setup validation SUCCESSFUL! Ready to run Misfits!")
        return True
    else:
        print("⚠️  Setup validation FAILED. Some components are missing.")
        return False

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)