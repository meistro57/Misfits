# state.py
"""In-memory state management for the Misfits web GUI."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CharacterSnapshot:
    """Snapshot of a character for UI consumption."""

    character_id: str
    name: str
    mood: str
    location: str
    activity: str
    energy: float


@dataclass
class GameSnapshot:
    """Snapshot of the overall game state."""

    updated_at: float
    chaos_status: str
    world_summary: Dict[str, Any]
    characters: List[CharacterSnapshot]
    recent_events: List[str]


@dataclass
class GameStateStore:
    """Mutable store for the web GUI, with a demo-state generator."""

    characters: Dict[str, CharacterSnapshot] = field(default_factory=dict)
    chaos_status: str = "stable"
    world_summary: Dict[str, Any] = field(default_factory=lambda: {"weather": "clear", "time": "day"})
    recent_events: List[str] = field(default_factory=list)

    def get_snapshot(self) -> GameSnapshot:
        """Return a copy of the current state."""

        return GameSnapshot(
            updated_at=time.time(),
            chaos_status=self.chaos_status,
            world_summary=self.world_summary.copy(),
            characters=list(self.characters.values()),
            recent_events=list(self.recent_events),
        )

    def record_event(self, message: str) -> None:
        """Record a new event for the UI."""

        self.recent_events.insert(0, message)
        if len(self.recent_events) > 10:
            self.recent_events.pop()

    def trigger_chaos(self) -> str:
        """Trigger a chaos event and return its description."""

        chaos_events = [
            "A spontaneous dance-off erupts in the kitchen.",
            "Someone discovers a mysterious locked diary.",
            "A rogue cat steals a sandwich and sparks an argument.",
            "The power flickers, and everyone blames the toaster.",
        ]
        event = random.choice(chaos_events)
        self.chaos_status = "building" if self.chaos_status == "stable" else "stable"
        self.record_event(event)
        return event

    def advance_demo_state(self) -> None:
        """Advance the demo state with small random changes."""

        moods = ["happy", "curious", "anxious", "excited", "bored"]
        activities = ["chatting", "reading", "cooking", "napping", "plotting"]
        locations = ["living_room", "kitchen", "garden", "bedroom", "study"]

        if not self.characters:
            self.characters = {
                "alice": CharacterSnapshot(
                    character_id="alice",
                    name="Alice",
                    mood="curious",
                    location="living_room",
                    activity="reading",
                    energy=0.8,
                ),
                "bob": CharacterSnapshot(
                    character_id="bob",
                    name="Bob",
                    mood="anxious",
                    location="kitchen",
                    activity="cooking",
                    energy=0.6,
                ),
                "carol": CharacterSnapshot(
                    character_id="carol",
                    name="Carol",
                    mood="happy",
                    location="garden",
                    activity="plotting",
                    energy=0.9,
                ),
            }

        for snapshot in self.characters.values():
            snapshot.mood = random.choice(moods)
            snapshot.activity = random.choice(activities)
            snapshot.location = random.choice(locations)
            snapshot.energy = round(max(0.1, min(1.0, snapshot.energy + random.uniform(-0.1, 0.1))), 2)

        self.world_summary["time"] = random.choice(["day", "night"])
        if random.random() < 0.2:
            self.record_event("A neighbourhood rumour gains traction.")
