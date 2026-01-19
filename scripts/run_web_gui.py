# run_web_gui.py
"""Launch the Misfits web GUI using Uvicorn."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web_gui.config import get_settings  # noqa: E402


def main() -> NoReturn:
    """Run the web GUI server."""

    settings = get_settings()
    uvicorn.run(
        "src.web_gui.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.demo_mode,
        log_level="info",
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
