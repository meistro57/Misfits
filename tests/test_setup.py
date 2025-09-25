#!/usr/bin/env python3
"""
Test Suite for Misfits! Game Setup Validation

This module contains comprehensive tests to ensure that the setup.py script
completes successfully and initializes all required components properly.
"""

import os
import sys
import sqlite3
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import asyncio
import yaml
import json
from contextlib import asynccontextmanager

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

# Import setup functions
from setup import (
    setup_database,
    setup_memory_database,
    setup_vector_database,
    setup_config_files,
    setup_directories,
    check_dependencies,
    check_llm_providers,
    main
)


class TestSetupDatabase(unittest.TestCase):
    """Test database initialization functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    def test_main_database_creation(self):
        """Test that main game database is created with correct tables."""
        db_path = "test_game.db"
        setup_database(db_path)

        # Verify database file exists
        self.assertTrue(os.path.exists(db_path))

        # Verify tables are created
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check all required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'game_state',
            'character_profiles',
            'world_events',
            'settings'
        ]

        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not created")

        # Test table schemas
        cursor.execute("PRAGMA table_info(game_state);")
        game_state_cols = [row[1] for row in cursor.fetchall()]
        expected_cols = ['id', 'save_name', 'game_data', 'created_at', 'updated_at']
        for col in expected_cols:
            self.assertIn(col, game_state_cols)

        conn.close()

    def test_database_constraint_unique_save_name(self):
        """Test that save_name constraint works properly."""
        db_path = "test_constraint.db"
        setup_database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Insert first record
        cursor.execute(
            "INSERT INTO game_state (save_name, game_data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("test_save", "test_data", 123456789, 123456789)
        )

        # Try to insert duplicate save_name - should fail
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO game_state (save_name, game_data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("test_save", "different_data", 987654321, 987654321)
            )

        conn.close()


class TestSetupMemoryDatabase(unittest.TestCase):
    """Test memory database setup."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch('setup.SQLiteMemoryStore')
    def test_memory_database_initialization(self, mock_memory_store):
        """Test that memory database initializes correctly."""
        mock_store_instance = MagicMock()
        mock_memory_store.return_value = mock_store_instance

        db_path = "test_memories.db"
        setup_memory_database(db_path)

        # Verify SQLiteMemoryStore was called with correct path
        mock_memory_store.assert_called_once_with(db_path)


class TestSetupVectorDatabase(unittest.TestCase):
    """Test vector database setup."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch('setup.create_vector_db')
    def test_vector_database_creation(self, mock_create_vector_db):
        """Test vector database creation with mock embedding."""
        mock_db_instance = MagicMock()
        mock_create_vector_db.return_value = mock_db_instance

        db_path = "test_vectors.db"
        setup_vector_database(db_path)

        # Verify create_vector_db was called with correct config
        mock_create_vector_db.assert_called_once()
        call_args = mock_create_vector_db.call_args[0][0]

        self.assertEqual(call_args["embedding"]["provider"], "mock")
        self.assertEqual(call_args["embedding"]["dimension"], 384)
        self.assertEqual(call_args["storage"]["type"], "sqlite")
        self.assertEqual(call_args["storage"]["path"], db_path)


class TestSetupConfigFiles(unittest.TestCase):
    """Test configuration file setup."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch('setup.create_config_loader')
    def test_config_files_creation(self, mock_create_config_loader):
        """Test that configuration files are created properly."""
        # Mock config loader
        mock_loader = MagicMock()
        mock_loader.config_dir = self.test_dir
        mock_create_config_loader.return_value = mock_loader

        setup_config_files()

        # Verify config loader methods were called
        mock_create_config_loader.assert_called_once()
        mock_loader.save_default_config.assert_called_once_with("game_config.yaml")

        # Verify simulation mode configs were created
        mode_configs_dir = Path(self.test_dir) / "simulation_modes"
        self.assertTrue(mode_configs_dir.exists())

        expected_files = ["learning_growth.yaml", "sandbox.yaml"]
        for config_file in expected_files:
            config_path = mode_configs_dir / config_file
            self.assertTrue(config_path.exists(), f"Config file {config_file} not created")

            # Verify content is valid YAML
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.assertIn("name", config_data)
                self.assertIn("description", config_data)
                self.assertIn("parameters", config_data)

    @patch('setup.create_config_loader')
    def test_existing_config_not_overwritten(self, mock_create_config_loader):
        """Test that existing config files are not overwritten."""
        # Create existing config file
        config_dir = Path(self.test_dir)
        game_config_path = config_dir / "game_config.yaml"
        game_config_path.parent.mkdir(parents=True, exist_ok=True)
        game_config_path.write_text("existing_config: true")

        # Mock config loader
        mock_loader = MagicMock()
        mock_loader.config_dir = str(config_dir)
        mock_create_config_loader.return_value = mock_loader

        setup_config_files()

        # Verify save_default_config was not called since file exists
        mock_loader.save_default_config.assert_not_called()

        # Verify existing content is preserved
        with open(game_config_path, 'r') as f:
            content = f.read()
            self.assertIn("existing_config: true", content)


class TestSetupDirectories(unittest.TestCase):
    """Test directory structure setup."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch('setup.project_root', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_all_directories_created(self, mock_project_root):
        """Test that all required directories are created."""
        setup_directories()

        expected_dirs = [
            "data/personalities",
            "data/simulation_modes",
            "data/events",
            "assets/models",
            "assets/textures",
            "assets/audio",
            "assets/ui",
            "tests",
            "docs/api",
            "docs/modding",
            "docs/deployment",
            "logs"
        ]

        for dir_path in expected_dirs:
            full_path = mock_project_root / dir_path
            self.assertTrue(full_path.exists(), f"Directory {dir_path} not created")
            self.assertTrue(full_path.is_dir(), f"{dir_path} is not a directory")

        # Clean up mock project root
        shutil.rmtree(mock_project_root)


class TestCheckDependencies(unittest.TestCase):
    """Test dependency checking functionality."""

    def test_all_dependencies_available(self):
        """Test when all required dependencies are available."""
        with patch('builtins.__import__') as mock_import:
            # Mock successful imports
            mock_import.return_value = MagicMock()

            result = check_dependencies()
            self.assertTrue(result)

    def test_missing_dependencies_detected(self):
        """Test when some dependencies are missing."""
        def mock_import_side_effect(name):
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return MagicMock()

        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            result = check_dependencies()
            self.assertFalse(result)


class TestCheckLLMProviders(unittest.TestCase):
    """Test LLM provider availability checks."""

    @patch('aiohttp.ClientSession')
    def test_ollama_available(self, mock_session):
        """Test when Ollama is available."""
        # Mock successful HTTP response for Ollama
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response

        mock_get = AsyncMock()
        mock_get.return_value = mock_response

        mock_session_instance = AsyncMock()
        mock_session_instance.get = mock_get
        mock_session_instance.__aenter__.return_value = mock_session_instance

        mock_session.return_value = mock_session_instance

        # This test mainly ensures the function doesn't crash
        # The actual async checking is complex to test in detail
        check_llm_providers()

    @patch('aiohttp.ClientSession')
    def test_llm_providers_unavailable(self, mock_session):
        """Test when LLM providers are not available."""
        # Mock failed HTTP requests
        mock_session_instance = AsyncMock()
        mock_session_instance.get.side_effect = Exception("Connection failed")
        mock_session_instance.__aenter__.return_value = mock_session_instance

        mock_session.return_value = mock_session_instance

        # This test mainly ensures the function handles errors gracefully
        check_llm_providers()


class TestSetupIntegration(unittest.TestCase):
    """Integration tests for complete setup process."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch('setup.project_root')
    @patch('setup.check_llm_providers')
    @patch('setup.setup_vector_database')
    @patch('setup.setup_memory_database')
    @patch('setup.setup_config_files')
    @patch('setup.check_dependencies')
    def test_successful_setup_integration(self, mock_check_deps, mock_setup_config,
                                         mock_setup_memory, mock_setup_vector,
                                         mock_check_llm, mock_project_root):
        """Test complete successful setup process."""
        # Setup mocks
        mock_project_root.return_value = Path(self.test_dir)
        mock_check_deps.return_value = True

        # Run main setup
        result = main()

        # Verify successful completion
        self.assertEqual(result, 0)

        # Verify all setup functions were called
        mock_check_deps.assert_called_once()
        mock_setup_config.assert_called_once()
        mock_setup_memory.assert_called_once()
        mock_setup_vector.assert_called_once()
        mock_check_llm.assert_called_once()

    @patch('setup.check_dependencies')
    def test_setup_fails_on_missing_dependencies(self, mock_check_deps):
        """Test that setup fails when dependencies are missing."""
        mock_check_deps.return_value = False

        result = main()

        # Verify setup failed
        self.assertEqual(result, 1)

    @patch('setup.check_dependencies')
    @patch('setup.setup_database')
    def test_setup_handles_exceptions(self, mock_setup_db, mock_check_deps):
        """Test that setup handles exceptions gracefully."""
        mock_check_deps.return_value = True
        mock_setup_db.side_effect = Exception("Database setup failed")

        result = main()

        # Verify setup failed gracefully
        self.assertEqual(result, 1)


class TestSetupValidation(unittest.TestCase):
    """Test validation of setup completion."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    def test_validate_complete_setup(self):
        """Test validation that setup completed successfully."""
        # Create expected files and directories
        setup_database("misfits.db")

        # Validate database exists and has correct structure
        self.assertTrue(os.path.exists("misfits.db"))

        conn = sqlite3.connect("misfits.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        required_tables = ['game_state', 'character_profiles', 'world_events', 'settings']
        for table in required_tables:
            self.assertIn(table, tables)

        conn.close()

    def test_database_write_permissions(self):
        """Test that databases can be written to after setup."""
        setup_database("test_write.db")

        conn = sqlite3.connect("test_write.db")
        cursor = conn.cursor()

        # Test writing to game_state table
        cursor.execute(
            "INSERT INTO game_state (save_name, game_data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("test_save", '{"test": "data"}', 123456789, 123456789)
        )

        # Test reading back
        cursor.execute("SELECT * FROM game_state WHERE save_name = ?", ("test_save",))
        result = cursor.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result[1], "test_save")

        conn.close()


def run_setup_validation_tests():
    """Run all setup validation tests and return results."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestSetupDatabase,
        TestSetupMemoryDatabase,
        TestSetupVectorDatabase,
        TestSetupConfigFiles,
        TestSetupDirectories,
        TestCheckDependencies,
        TestCheckLLMProviders,
        TestSetupIntegration,
        TestSetupValidation
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("🧪 Running Misfits! Setup Validation Tests")
    print("=" * 50)

    result = run_setup_validation_tests()

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All setup validation tests passed!")
        exit_code = 0
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        exit_code = 1

    print("=" * 50)
    sys.exit(exit_code)