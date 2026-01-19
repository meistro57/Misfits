# config.py
"""Configuration for the Misfits web GUI."""

from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebGUISettings(BaseSettings):
    """Settings for the web GUI, configurable via environment variables."""

    app_name: str = Field(default="Misfits Web GUI")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    demo_mode: bool = Field(
        default=True,
        description=(
            "Run the GUI with a demo data generator by default for development. "
            "Override via MISFITS_WEB_DEMO=false in production."
        ),
    )
    update_interval_seconds: float = Field(default=2.0, ge=0.5)

    model_config = SettingsConfigDict(env_prefix="MISFITS_WEB_", case_sensitive=False)


@lru_cache(maxsize=1)
def get_settings() -> WebGUISettings:
    """Return cached settings instance."""

    return WebGUISettings()
