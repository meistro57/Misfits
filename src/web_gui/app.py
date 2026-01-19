# app.py
"""FastAPI application for the Misfits web GUI."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.web_gui.config import WebGUISettings, get_settings
from src.web_gui.state import GameStateStore


LOGGER = logging.getLogger("misfits.web_gui")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Misfits Web GUI</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f1115;
      --card: #1a1d24;
      --text: #f3f4f6;
      --muted: #9ca3af;
      --accent: #f59e0b;
      --success: #22c55e;
    }
    body {
      margin: 0;
      font-family: "Inter", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 24px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #2b2f3a;
    }
    main {
      padding: 24px 32px 48px;
      display: grid;
      gap: 24px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }
    .card {
      background: var(--card);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .title {
      font-size: 18px;
      margin-bottom: 12px;
    }
    .muted {
      color: var(--muted);
    }
    button {
      border: none;
      background: var(--accent);
      color: #111827;
      padding: 10px 16px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary {
      background: transparent;
      border: 1px solid var(--muted);
      color: var(--text);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(245, 158, 11, 0.15);
      color: var(--accent);
    }
    ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    li {
      padding: 8px 0;
      border-bottom: 1px dashed rgba(255,255,255,0.08);
    }
    li:last-child {
      border-bottom: none;
    }
    .character {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .stat {
      font-size: 14px;
    }
    .tag {
      background: rgba(34, 197, 94, 0.15);
      color: var(--success);
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
    }
    .error {
      color: #f87171;
      margin-top: 8px;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Misfits Web GUI</h1>
      <p class="muted">Live view of the neighbourhood chaos.</p>
    </div>
    <div>
      <button id="chaos-btn">Trigger Chaos</button>
      <button class="secondary" id="refresh-btn">Refresh</button>
    </div>
  </header>
  <main>
    <section class="card">
      <div class="title">World Status</div>
      <div id="world-status" class="muted">Loading...</div>
      <div id="chaos-status" class="pill" style="margin-top: 12px;">Stable</div>
    </section>
    <section class="card">
      <div class="title">Characters</div>
      <ul id="character-list"></ul>
    </section>
    <section class="card">
      <div class="title">Recent Events</div>
      <ul id="event-list"></ul>
    </section>
  </main>
  <script>
    const stateUrl = "/api/state";
    const chaosUrl = "/api/chaos";

    const worldStatus = document.getElementById("world-status");
    const chaosStatus = document.getElementById("chaos-status");
    const characterList = document.getElementById("character-list");
    const eventList = document.getElementById("event-list");
    const chaosBtn = document.getElementById("chaos-btn");
    const refreshBtn = document.getElementById("refresh-btn");

    async function fetchState() {
      try {
        const response = await fetch(stateUrl);
        if (!response.ok) {
          throw new Error("Unable to fetch state");
        }
        const data = await response.json();
        renderState(data);
      } catch (error) {
        worldStatus.textContent = "Error loading state.";
        worldStatus.classList.add("error");
      }
    }

    function renderState(data) {
      worldStatus.textContent = `Weather: ${data.world_summary.weather} · Time: ${data.world_summary.time}`;
      chaosStatus.textContent = `Chaos: ${data.chaos_status}`;

      characterList.innerHTML = "";
      data.characters.forEach(character => {
        const item = document.createElement("li");
        item.innerHTML = `
          <div class="character">
            <div>
              <strong>${character.name}</strong>
              <div class="stat muted">${character.activity} in ${character.location}</div>
            </div>
            <span class="tag">${character.mood}</span>
          </div>
        `;
        characterList.appendChild(item);
      });

      eventList.innerHTML = "";
      if (data.recent_events.length === 0) {
        eventList.innerHTML = "<li class=\"muted\">No events yet.</li>";
        return;
      }
      data.recent_events.forEach(event => {
        const item = document.createElement("li");
        item.textContent = event;
        eventList.appendChild(item);
      });
    }

    chaosBtn.addEventListener("click", async () => {
      try {
        const response = await fetch(chaosUrl, { method: "POST" });
        if (!response.ok) {
          throw new Error("Chaos event failed");
        }
        await fetchState();
      } catch (error) {
        chaosStatus.textContent = "Chaos: error";
        chaosStatus.classList.add("error");
      }
    });

    refreshBtn.addEventListener("click", fetchState);

    fetchState();
    setInterval(fetchState, 4000);
  </script>
</body>
</html>
"""


def create_app(settings: WebGUISettings | None = None) -> FastAPI:
    """Create the FastAPI app instance."""

    app_settings = settings or get_settings()
    store = GameStateStore()

    async def demo_loop() -> None:
        """Run demo updates in the background."""

        while True:
            try:
                store.advance_demo_state()
            except Exception as exc:
                LOGGER.exception("Failed to update demo state", exc_info=exc)
            await asyncio.sleep(app_settings.update_interval_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = None
        if app_settings.demo_mode:
            task = asyncio.create_task(demo_loop())
        yield
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title=app_settings.app_name, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML_TEMPLATE

    @app.get("/api/state")
    async def get_state() -> Dict[str, Any]:
        snapshot = store.get_snapshot()
        return {
            "updated_at": snapshot.updated_at,
            "chaos_status": snapshot.chaos_status,
            "world_summary": snapshot.world_summary,
            "characters": [character.__dict__ for character in snapshot.characters],
            "recent_events": snapshot.recent_events,
        }

    @app.post("/api/chaos")
    async def trigger_chaos() -> JSONResponse:
        try:
            event = store.trigger_chaos()
        except Exception as exc:
            LOGGER.exception("Chaos trigger failed", exc_info=exc)
            raise HTTPException(status_code=500, detail="Chaos trigger failed") from exc
        return JSONResponse({"event": event})

    return app


app = create_app()
