"""
Config Loader - Configuration management and validation system.

This module handles loading, validating, and managing configuration
files for different aspects of the Misfits game system.
"""

import json
import yaml
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


@dataclass
class LLMProviderConfig:
    """Configuration for LLM providers."""
    provider: str
    host: str = "localhost"
    port: int = 11434
    model: str = "llama2:7b-chat"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    api_key: Optional[str] = None


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    type: str = "sqlite"
    path: str = "misfits.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    embedding_provider: str = "mock"
    embedding_model: str = "all-MiniLM-L6-v2"
    storage_type: str = "sqlite"
    storage_path: str = "vectors.db"
    dimension: int = 384


@dataclass
class SimulationConfig:
    """Configuration for simulation settings."""
    default_mode: str = "comedy_chaos"
    tick_interval: float = 5.0
    max_characters: int = 10
    auto_save_interval: int = 300  # seconds
    enable_chaos_button: bool = True
    enable_interventions: bool = True


@dataclass
class UIConfig:
    """Configuration for UI settings."""
    theme: str = "default"
    show_debug_info: bool = False
    auto_follow_drama: bool = True
    max_timeline_events: int = 100
    update_frequency: float = 2.0


@dataclass
class GameConfig:
    """Main game configuration."""
    llm: Dict[str, LLMProviderConfig]
    database: DatabaseConfig
    vector_db: VectorDBConfig
    simulation: SimulationConfig
    ui: UIConfig
    debug_mode: bool = False
    log_level: str = "INFO"


class ConfigValidator:
    """Validates configuration settings."""

    def __init__(self):
        self.errors: List[str] = []

    def validate_game_config(self, config: GameConfig) -> bool:
        """Validate complete game configuration."""
        self.errors.clear()

        # Validate LLM configuration
        if not config.llm:
            self.errors.append("At least one LLM provider must be configured")
        else:
            for provider_name, provider_config in config.llm.items():
                self._validate_llm_provider(provider_name, provider_config)

        # Validate database configuration
        self._validate_database_config(config.database)

        # Validate vector database configuration
        self._validate_vector_db_config(config.vector_db)

        # Validate simulation configuration
        self._validate_simulation_config(config.simulation)

        # Validate UI configuration
        self._validate_ui_config(config.ui)

        return len(self.errors) == 0

    def _validate_llm_provider(self, name: str, config: LLMProviderConfig):
        """Validate LLM provider configuration."""
        valid_providers = ["ollama", "lm_studio", "gpt4all", "openai_compatible", "mock"]

        if config.provider not in valid_providers:
            self.errors.append(f"Invalid LLM provider '{config.provider}' for {name}")

        if config.temperature < 0.0 or config.temperature > 2.0:
            self.errors.append(f"Temperature must be between 0.0 and 2.0 for {name}")

        if config.max_tokens <= 0:
            self.errors.append(f"Max tokens must be positive for {name}")

        if config.port <= 0 or config.port > 65535:
            self.errors.append(f"Invalid port number for {name}")

    def _validate_database_config(self, config: DatabaseConfig):
        """Validate database configuration."""
        valid_types = ["sqlite", "postgresql", "mysql"]

        if config.type not in valid_types:
            self.errors.append(f"Invalid database type '{config.type}'")

        if config.type == "sqlite":
            if not config.path:
                self.errors.append("SQLite database path is required")
        else:
            if not config.host:
                self.errors.append(f"Host is required for {config.type} database")
            if not config.port:
                self.errors.append(f"Port is required for {config.type} database")

    def _validate_vector_db_config(self, config: VectorDBConfig):
        """Validate vector database configuration."""
        valid_embedding_providers = ["mock", "sentence_transformers", "huggingface", "openai"]
        valid_storage_types = ["memory", "sqlite", "faiss", "chroma"]

        if config.embedding_provider not in valid_embedding_providers:
            self.errors.append(f"Invalid embedding provider '{config.embedding_provider}'")

        if config.storage_type not in valid_storage_types:
            self.errors.append(f"Invalid vector storage type '{config.storage_type}'")

        if config.dimension <= 0:
            self.errors.append("Vector dimension must be positive")

    def _validate_simulation_config(self, config: SimulationConfig):
        """Validate simulation configuration."""
        valid_modes = ["comedy_chaos", "psychological_deep", "learning_growth", "sandbox"]

        if config.default_mode not in valid_modes:
            self.errors.append(f"Invalid default simulation mode '{config.default_mode}'")

        if config.tick_interval <= 0:
            self.errors.append("Tick interval must be positive")

        if config.max_characters <= 0:
            self.errors.append("Max characters must be positive")

        if config.auto_save_interval <= 0:
            self.errors.append("Auto save interval must be positive")

    def _validate_ui_config(self, config: UIConfig):
        """Validate UI configuration."""
        valid_themes = ["default", "dark", "light", "colorful"]

        if config.theme not in valid_themes:
            self.errors.append(f"Invalid UI theme '{config.theme}'")

        if config.update_frequency <= 0:
            self.errors.append("UI update frequency must be positive")

        if config.max_timeline_events <= 0:
            self.errors.append("Max timeline events must be positive")

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors.copy()


class ConfigLoader:
    """Loads and manages configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.validator = ConfigValidator()

    def load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load a configuration file (JSON or YAML)."""
        file_path = self.config_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        # Determine file format
        suffix = file_path.suffix.lower()
        if suffix == '.json':
            return self._load_json(file_path)
        elif suffix in ['.yaml', '.yml']:
            return self._load_yaml(file_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML library required to load YAML configuration files")

    def load_game_config(self, config_file: str = "game_config.yaml") -> GameConfig:
        """Load main game configuration."""
        config_dict = self.load_config_file(config_file)
        return self._parse_game_config(config_dict)

    def _parse_game_config(self, config_dict: Dict[str, Any]) -> GameConfig:
        """Parse configuration dictionary into GameConfig object."""

        # Parse LLM providers
        llm_configs = {}
        for provider_name, provider_dict in config_dict.get("llm", {}).items():
            llm_configs[provider_name] = LLMProviderConfig(**provider_dict)

        # Parse database config
        database_config = DatabaseConfig(**config_dict.get("database", {}))

        # Parse vector database config
        vector_db_dict = config_dict.get("vector_db", {})
        vector_db_config = VectorDBConfig(
            embedding_provider=vector_db_dict.get("embedding_provider", "mock"),
            embedding_model=vector_db_dict.get("embedding_model", "all-MiniLM-L6-v2"),
            storage_type=vector_db_dict.get("storage_type", "sqlite"),
            storage_path=vector_db_dict.get("storage_path", "vectors.db"),
            dimension=vector_db_dict.get("dimension", 384)
        )

        # Parse simulation config
        simulation_config = SimulationConfig(**config_dict.get("simulation", {}))

        # Parse UI config
        ui_config = UIConfig(**config_dict.get("ui", {}))

        return GameConfig(
            llm=llm_configs,
            database=database_config,
            vector_db=vector_db_config,
            simulation=simulation_config,
            ui=ui_config,
            debug_mode=config_dict.get("debug_mode", False),
            log_level=config_dict.get("log_level", "INFO")
        )

    def save_config_file(self, filename: str, config_data: Dict[str, Any],
                        format: ConfigFormat = ConfigFormat.YAML):
        """Save configuration to file."""
        file_path = self.config_dir / filename

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if format == ConfigFormat.JSON:
            self._save_json(file_path.with_suffix('.json'), config_data)
        else:  # YAML
            self._save_yaml(file_path.with_suffix('.yaml'), config_data)

    def _save_json(self, file_path: Path, config_data: Dict[str, Any]):
        """Save configuration as JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def _save_yaml(self, file_path: Path, config_data: Dict[str, Any]):
        """Save configuration as YAML."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except ImportError:
            raise ImportError("PyYAML library required to save YAML configuration files")

    def validate_config(self, config: GameConfig) -> bool:
        """Validate configuration and return True if valid."""
        return self.validator.validate_game_config(config)

    def get_validation_errors(self) -> List[str]:
        """Get configuration validation errors."""
        return self.validator.get_errors()

    def create_default_config(self) -> GameConfig:
        """Create default configuration."""
        return GameConfig(
            llm={
                "primary": LLMProviderConfig(
                    provider="ollama",
                    host="localhost",
                    port=11434,
                    model="llama2:7b-chat",
                    temperature=0.7
                ),
                "fallback": LLMProviderConfig(
                    provider="mock",
                    host="localhost",
                    port=0,
                    model="mock",
                    temperature=0.7
                )
            },
            database=DatabaseConfig(
                type="sqlite",
                path="misfits.db"
            ),
            vector_db=VectorDBConfig(
                embedding_provider="mock",
                storage_type="sqlite",
                storage_path="vectors.db"
            ),
            simulation=SimulationConfig(
                default_mode="comedy_chaos",
                tick_interval=5.0,
                max_characters=10
            ),
            ui=UIConfig(
                theme="default",
                auto_follow_drama=True
            )
        )

    def save_default_config(self, filename: str = "game_config.yaml"):
        """Save default configuration to file."""
        default_config = self.create_default_config()
        config_dict = self._config_to_dict(default_config)
        self.save_config_file(filename, config_dict)

    def _config_to_dict(self, config: GameConfig) -> Dict[str, Any]:
        """Convert GameConfig to dictionary for serialization."""
        return {
            "llm": {name: asdict(llm_config) for name, llm_config in config.llm.items()},
            "database": asdict(config.database),
            "vector_db": asdict(config.vector_db),
            "simulation": asdict(config.simulation),
            "ui": asdict(config.ui),
            "debug_mode": config.debug_mode,
            "log_level": config.log_level
        }

    def load_personality_config(self, filename: str) -> Dict[str, Any]:
        """Load personality configuration file."""
        return self.load_config_file(filename)

    def load_simulation_mode_config(self, filename: str) -> Dict[str, Any]:
        """Load simulation mode configuration file."""
        return self.load_config_file(filename)

    def load_events_config(self, filename: str) -> Dict[str, Any]:
        """Load events configuration file."""
        return self.load_config_file(filename)

    def get_config_files(self) -> List[str]:
        """Get list of available configuration files."""
        if not self.config_dir.exists():
            return []

        config_files = []
        for file_path in self.config_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.json', '.yaml', '.yml']:
                config_files.append(file_path.name)

        return sorted(config_files)

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries, with override taking precedence."""

        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()

            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value

            return result

        return deep_merge(base_config, override_config)

    def load_config_with_overrides(self, base_config_file: str,
                                  override_files: List[str] = None) -> GameConfig:
        """Load configuration with optional override files."""
        base_config = self.load_config_file(base_config_file)

        if override_files:
            for override_file in override_files:
                if Path(self.config_dir / override_file).exists():
                    override_config = self.load_config_file(override_file)
                    base_config = self.merge_configs(base_config, override_config)

        return self._parse_game_config(base_config)

    def load_environment_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # LLM configuration from environment
        if os.getenv("MISFITS_LLM_HOST"):
            env_config["llm"] = {
                "primary": {
                    "provider": os.getenv("MISFITS_LLM_PROVIDER", "ollama"),
                    "host": os.getenv("MISFITS_LLM_HOST"),
                    "port": int(os.getenv("MISFITS_LLM_PORT", "11434")),
                    "model": os.getenv("MISFITS_LLM_MODEL", "llama2:7b-chat"),
                    "temperature": float(os.getenv("MISFITS_LLM_TEMPERATURE", "0.7"))
                }
            }

        # Database configuration from environment
        if os.getenv("MISFITS_DB_PATH"):
            env_config["database"] = {
                "type": os.getenv("MISFITS_DB_TYPE", "sqlite"),
                "path": os.getenv("MISFITS_DB_PATH")
            }

        # Debug mode from environment
        if os.getenv("MISFITS_DEBUG"):
            env_config["debug_mode"] = os.getenv("MISFITS_DEBUG").lower() in ["true", "1", "yes"]

        return env_config


def create_config_loader(config_dir: str = None) -> ConfigLoader:
    """Create configuration loader with optional custom directory."""
    if config_dir is None:
        # Default to 'data' directory in project
        project_root = Path(__file__).parent.parent.parent
        config_dir = str(project_root / "data")

    return ConfigLoader(config_dir)