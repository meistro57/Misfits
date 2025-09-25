"""
Chaos Button - UI component for triggering random world events.

This module provides the chaos button interface that allows players
to inject randomness and unexpected events into the game world.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ChaosButtonState(Enum):
    """States of the chaos button."""
    READY = "ready"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"
    CHARGING = "charging"


@dataclass
class ChaosButtonConfig:
    """Configuration for chaos button behavior."""
    cooldown_seconds: float = 300.0  # 5 minutes
    charge_time_seconds: float = 30.0
    max_uses_per_hour: int = 3
    requires_confirmation: bool = True
    dramatic_buildup: bool = True


class ChaosButtonUI:
    """UI component for the chaos button."""

    def __init__(self, config: ChaosButtonConfig = None):
        self.config = config or ChaosButtonConfig()
        self.state = ChaosButtonState.READY
        self.last_use_time = 0.0
        self.usage_history: List[float] = []
        self.charge_start_time: Optional[float] = None

        # Event callbacks
        self.on_button_pressed: Optional[Callable] = None
        self.on_chaos_triggered: Optional[Callable] = None
        self.on_cooldown_started: Optional[Callable] = None
        self.on_state_changed: Optional[Callable] = None

    def set_callbacks(self, **callbacks):
        """Set event callbacks."""
        for event_name, callback in callbacks.items():
            if hasattr(self, event_name):
                setattr(self, event_name, callback)

    def can_use_button(self) -> bool:
        """Check if the chaos button can currently be used."""
        current_time = time.time()

        # Check if in cooldown
        if self.state == ChaosButtonState.COOLDOWN:
            if current_time - self.last_use_time < self.config.cooldown_seconds:
                return False
            else:
                self._change_state(ChaosButtonState.READY)

        # Check usage limits
        hour_ago = current_time - 3600
        recent_uses = [t for t in self.usage_history if t > hour_ago]

        if len(recent_uses) >= self.config.max_uses_per_hour:
            return False

        # Check if disabled
        if self.state == ChaosButtonState.DISABLED:
            return False

        return True

    def initiate_chaos(self) -> Dict[str, Any]:
        """Initiate chaos event sequence."""
        if not self.can_use_button():
            return {
                "success": False,
                "reason": self._get_unavailable_reason(),
                "state": self.state.value
            }

        # Trigger button pressed callback
        if self.on_button_pressed:
            self.on_button_pressed()

        if self.config.requires_confirmation:
            return {
                "success": True,
                "requires_confirmation": True,
                "message": "Are you sure you want to unleash chaos?",
                "confirmation_timeout": 10.0
            }
        else:
            return self._execute_chaos()

    def confirm_chaos(self) -> Dict[str, Any]:
        """Confirm chaos execution after confirmation prompt."""
        if not self.can_use_button():
            return {
                "success": False,
                "reason": "Chaos button no longer available",
                "state": self.state.value
            }

        return self._execute_chaos()

    def _execute_chaos(self) -> Dict[str, Any]:
        """Execute the actual chaos event."""
        current_time = time.time()

        if self.config.dramatic_buildup:
            # Start charging sequence
            self._change_state(ChaosButtonState.CHARGING)
            self.charge_start_time = current_time

            return {
                "success": True,
                "charging": True,
                "charge_time": self.config.charge_time_seconds,
                "message": "Chaos is building..."
            }
        else:
            # Immediate execution
            return self._trigger_chaos_event()

    def update(self) -> Dict[str, Any]:
        """Update chaos button state (called regularly by game loop)."""
        current_time = time.time()
        status_changes = {}

        # Handle charging state
        if self.state == ChaosButtonState.CHARGING:
            if self.charge_start_time and current_time - self.charge_start_time >= self.config.charge_time_seconds:
                chaos_result = self._trigger_chaos_event()
                status_changes.update(chaos_result)

        # Handle cooldown expiration
        elif self.state == ChaosButtonState.COOLDOWN:
            if current_time - self.last_use_time >= self.config.cooldown_seconds:
                self._change_state(ChaosButtonState.READY)
                status_changes["cooldown_expired"] = True

        # Clean up old usage history
        hour_ago = current_time - 3600
        old_count = len(self.usage_history)
        self.usage_history = [t for t in self.usage_history if t > hour_ago]
        if len(self.usage_history) < old_count:
            status_changes["usage_history_cleaned"] = True

        return status_changes

    def _trigger_chaos_event(self) -> Dict[str, Any]:
        """Actually trigger the chaos event."""
        current_time = time.time()

        # Record usage
        self.last_use_time = current_time
        self.usage_history.append(current_time)
        self.charge_start_time = None

        # Enter cooldown
        self._change_state(ChaosButtonState.COOLDOWN)

        # Trigger chaos callback
        chaos_event = None
        if self.on_chaos_triggered:
            chaos_event = self.on_chaos_triggered()

        # Trigger cooldown callback
        if self.on_cooldown_started:
            self.on_cooldown_started(self.config.cooldown_seconds)

        return {
            "success": True,
            "chaos_triggered": True,
            "event": chaos_event,
            "cooldown_duration": self.config.cooldown_seconds,
            "message": self._get_chaos_message()
        }

    def _change_state(self, new_state: ChaosButtonState):
        """Change button state and notify callback."""
        old_state = self.state
        self.state = new_state

        if self.on_state_changed:
            self.on_state_changed(old_state, new_state)

    def _get_unavailable_reason(self) -> str:
        """Get reason why button is unavailable."""
        current_time = time.time()

        if self.state == ChaosButtonState.DISABLED:
            return "Chaos button is disabled in current simulation mode"

        if self.state == ChaosButtonState.COOLDOWN:
            remaining = self.config.cooldown_seconds - (current_time - self.last_use_time)
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            return f"Chaos button cooling down ({minutes}:{seconds:02d} remaining)"

        if self.state == ChaosButtonState.CHARGING:
            return "Chaos is already building up..."

        # Check usage limits
        hour_ago = current_time - 3600
        recent_uses = [t for t in self.usage_history if t > hour_ago]
        if len(recent_uses) >= self.config.max_uses_per_hour:
            return f"Maximum chaos uses per hour reached ({self.config.max_uses_per_hour})"

        return "Unknown reason"

    def _get_chaos_message(self) -> str:
        """Get a dramatic message for chaos activation."""
        messages = [
            "Reality tears at the seams...",
            "The universe hiccups...",
            "Chaos enters the chat...",
            "Murphy's Law activates...",
            "The butterfly effect intensifies...",
            "Plot twist incoming...",
            "The simulation glitches...",
            "Pandora's box creaks open...",
            "Chaos theory in action...",
            "The unexpected becomes expected..."
        ]

        import random
        return random.choice(messages)

    def get_button_display_info(self) -> Dict[str, Any]:
        """Get information for displaying the button in UI."""
        current_time = time.time()

        info = {
            "state": self.state.value,
            "available": self.can_use_button(),
            "text": self._get_button_text(),
            "tooltip": self._get_button_tooltip(),
            "style_class": self._get_button_style_class()
        }

        # Add progress information for various states
        if self.state == ChaosButtonState.COOLDOWN:
            remaining = self.config.cooldown_seconds - (current_time - self.last_use_time)
            info["cooldown_progress"] = max(0, 1 - (remaining / self.config.cooldown_seconds))
            info["time_remaining"] = max(0, remaining)

        elif self.state == ChaosButtonState.CHARGING and self.charge_start_time:
            elapsed = current_time - self.charge_start_time
            info["charge_progress"] = min(1.0, elapsed / self.config.charge_time_seconds)
            info["time_remaining"] = max(0, self.config.charge_time_seconds - elapsed)

        # Usage information
        hour_ago = current_time - 3600
        recent_uses = [t for t in self.usage_history if t > hour_ago]
        info["uses_this_hour"] = len(recent_uses)
        info["uses_remaining"] = max(0, self.config.max_uses_per_hour - len(recent_uses))

        return info

    def _get_button_text(self) -> str:
        """Get text to display on the button."""
        if self.state == ChaosButtonState.READY:
            return "🎲 CHAOS"
        elif self.state == ChaosButtonState.CHARGING:
            return "⚡ BUILDING..."
        elif self.state == ChaosButtonState.COOLDOWN:
            current_time = time.time()
            remaining = self.config.cooldown_seconds - (current_time - self.last_use_time)
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            return f"🕒 {minutes}:{seconds:02d}"
        elif self.state == ChaosButtonState.DISABLED:
            return "🚫 DISABLED"
        else:
            return "CHAOS"

    def _get_button_tooltip(self) -> str:
        """Get tooltip text for the button."""
        if self.state == ChaosButtonState.READY:
            return "Click to unleash chaos upon the world! Unpredictable events await..."
        elif self.state == ChaosButtonState.CHARGING:
            return "Chaos is building up energy. Prepare for the unexpected!"
        elif self.state == ChaosButtonState.COOLDOWN:
            return "Chaos button is cooling down. Even chaos needs a break."
        elif self.state == ChaosButtonState.DISABLED:
            return "Chaos button is disabled in the current simulation mode."
        else:
            return "The chaos button controls the unexpected events in your world."

    def _get_button_style_class(self) -> str:
        """Get CSS/style class for the button."""
        style_classes = {
            ChaosButtonState.READY: "chaos-button-ready",
            ChaosButtonState.CHARGING: "chaos-button-charging",
            ChaosButtonState.COOLDOWN: "chaos-button-cooldown",
            ChaosButtonState.DISABLED: "chaos-button-disabled"
        }
        return style_classes.get(self.state, "chaos-button-default")

    def enable_button(self):
        """Enable the chaos button."""
        if self.state == ChaosButtonState.DISABLED:
            self._change_state(ChaosButtonState.READY)

    def disable_button(self):
        """Disable the chaos button."""
        self._change_state(ChaosButtonState.DISABLED)

    def reset_cooldown(self):
        """Reset cooldown (admin/debug function)."""
        if self.state == ChaosButtonState.COOLDOWN:
            self._change_state(ChaosButtonState.READY)

    def get_statistics(self) -> Dict[str, Any]:
        """Get chaos button usage statistics."""
        current_time = time.time()

        # Calculate usage over different time periods
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        week_ago = current_time - 604800

        hour_uses = [t for t in self.usage_history if t > hour_ago]
        day_uses = [t for t in self.usage_history if t > day_ago]
        week_uses = [t for t in self.usage_history if t > week_ago]

        return {
            "total_uses": len(self.usage_history),
            "uses_last_hour": len(hour_uses),
            "uses_last_day": len(day_uses),
            "uses_last_week": len(week_uses),
            "current_state": self.state.value,
            "last_use_ago": current_time - self.last_use_time if self.last_use_time > 0 else None,
            "average_uses_per_day": len(week_uses) / 7 if week_uses else 0
        }