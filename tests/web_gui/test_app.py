# test_app.py
"""Tests for the Misfits web GUI."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web_gui.app import create_app  # noqa: E402
from src.web_gui.config import WebGUISettings  # noqa: E402


def _create_test_client() -> TestClient:
    settings = WebGUISettings(demo_mode=False)
    app = create_app(settings)
    return TestClient(app)


def test_index_page_returns_html() -> None:
    client = _create_test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "Misfits Web GUI" in response.text


def test_state_endpoint_returns_payload() -> None:
    client = _create_test_client()

    response = client.get("/api/state")

    assert response.status_code == 200
    payload = response.json()
    assert "chaos_status" in payload
    assert "world_summary" in payload
    assert "characters" in payload


def test_chaos_endpoint_records_event() -> None:
    client = _create_test_client()

    response = client.post("/api/chaos")

    assert response.status_code == 200
    payload = response.json()
    assert payload["event"]
