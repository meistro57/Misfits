"""
Observation Panel - UI for watching and monitoring character behavior.

This module provides the main observation interface where players
can watch characters live their lives and see the emergent stories unfold.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class ObservationMode(Enum):
    """Different observation modes for watching characters."""
    OVERVIEW = "overview"          # See all characters at once
    CHARACTER_FOCUS = "character_focus"  # Focus on one character
    LOCATION_FOCUS = "location_focus"    # Focus on one location
    RELATIONSHIP_VIEW = "relationship_view"  # Focus on relationships
    STORY_MODE = "story_mode"      # Narrative-focused view


class TimelineFilter(Enum):
    """Filters for the event timeline."""
    ALL = "all"
    SOCIAL = "social"
    DRAMATIC = "dramatic"
    FUNNY = "funny"
    SIGNIFICANT = "significant"


@dataclass
class ObservationConfig:
    """Configuration for the observation panel."""
    auto_follow_drama: bool = True
    show_internal_thoughts: bool = True
    highlight_conflicts: bool = True
    max_timeline_events: int = 100
    update_frequency: float = 2.0  # seconds between updates


class CharacterObserver:
    """Observes and tracks individual character states and actions."""

    def __init__(self):
        self.character_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.interesting_moments: List[Dict[str, Any]] = []

    def update_character_state(self, character_id: str, new_state: Dict[str, Any]):
        """Update character state and detect interesting moments."""
        if character_id not in self.character_histories:
            self.character_histories[character_id] = []

        history = self.character_histories[character_id]

        # Add timestamp to state
        timestamped_state = {
            **new_state,
            "timestamp": time.time()
        }

        # Detect interesting changes
        if history:
            interesting_changes = self._detect_interesting_changes(
                history[-1], timestamped_state, character_id
            )
            self.interesting_moments.extend(interesting_changes)

        # Add to history
        history.append(timestamped_state)

        # Keep history manageable
        if len(history) > 200:
            history[:] = history[-150:]

    def _detect_interesting_changes(self, old_state: Dict[str, Any],
                                  new_state: Dict[str, Any],
                                  character_id: str) -> List[Dict[str, Any]]:
        """Detect interesting changes in character state."""
        changes = []

        # Mood changes
        if old_state.get("mood") != new_state.get("mood"):
            mood_change = {
                "type": "mood_change",
                "character": character_id,
                "from": old_state.get("mood"),
                "to": new_state.get("mood"),
                "timestamp": new_state["timestamp"],
                "significance": self._calculate_mood_significance(
                    old_state.get("mood"), new_state.get("mood")
                )
            }
            changes.append(mood_change)

        # Location changes
        if old_state.get("location") != new_state.get("location"):
            changes.append({
                "type": "location_change",
                "character": character_id,
                "from": old_state.get("location"),
                "to": new_state.get("location"),
                "timestamp": new_state["timestamp"],
                "significance": 0.3
            })

        # Activity changes
        if old_state.get("current_action") != new_state.get("current_action"):
            changes.append({
                "type": "activity_change",
                "character": character_id,
                "from": old_state.get("current_action"),
                "to": new_state.get("current_action"),
                "timestamp": new_state["timestamp"],
                "significance": 0.2
            })

        # Energy level significant changes
        old_energy = old_state.get("energy", 0.5)
        new_energy = new_state.get("energy", 0.5)
        if abs(old_energy - new_energy) > 0.3:
            changes.append({
                "type": "energy_change",
                "character": character_id,
                "from": old_energy,
                "to": new_energy,
                "timestamp": new_state["timestamp"],
                "significance": abs(old_energy - new_energy)
            })

        return changes

    def _calculate_mood_significance(self, old_mood: str, new_mood: str) -> float:
        """Calculate how significant a mood change is."""
        mood_intensity = {
            "ecstatic": 1.0, "angry": 0.9, "depressed": 0.9, "love": 0.9,
            "excited": 0.8, "frustrated": 0.7, "anxious": 0.7,
            "happy": 0.6, "sad": 0.6, "worried": 0.5,
            "content": 0.3, "bored": 0.3, "neutral": 0.1
        }

        old_intensity = mood_intensity.get(old_mood, 0.3)
        new_intensity = mood_intensity.get(new_mood, 0.3)

        return abs(new_intensity - old_intensity)

    def get_character_summary(self, character_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a character's current state."""
        history = self.character_histories.get(character_id, [])

        if not history:
            return {"error": "No data available for character"}

        current_state = history[-1]
        recent_history = history[-10:] if len(history) >= 10 else history

        # Calculate patterns
        mood_pattern = self._analyze_mood_pattern(recent_history)
        activity_pattern = self._analyze_activity_pattern(recent_history)

        summary = {
            "current_state": current_state,
            "mood_pattern": mood_pattern,
            "activity_pattern": activity_pattern,
            "recent_significant_moments": self._get_recent_moments(character_id, 5),
            "character_trajectory": self._assess_character_trajectory(recent_history)
        }

        return summary

    def _analyze_mood_pattern(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in mood changes."""
        if len(history) < 2:
            return {"pattern": "insufficient_data"}

        moods = [state.get("mood", "neutral") for state in history]
        mood_changes = sum(1 for i in range(1, len(moods)) if moods[i] != moods[i-1])

        return {
            "volatility": mood_changes / max(1, len(moods) - 1),
            "current_mood": moods[-1],
            "dominant_moods": self._get_dominant_values(moods),
            "stability": "volatile" if mood_changes > len(moods) * 0.5 else "stable"
        }

    def _analyze_activity_pattern(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in activities."""
        if len(history) < 2:
            return {"pattern": "insufficient_data"}

        activities = [state.get("current_action", "idle") for state in history]

        return {
            "current_activity": activities[-1],
            "dominant_activities": self._get_dominant_values(activities),
            "variety_score": len(set(activities)) / len(activities) if activities else 0
        }

    def _get_dominant_values(self, values: List[str]) -> List[str]:
        """Get the most common values from a list."""
        from collections import Counter
        counter = Counter(values)
        return [item for item, count in counter.most_common(3)]

    def _get_recent_moments(self, character_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent interesting moments for a character."""
        character_moments = [
            moment for moment in self.interesting_moments
            if moment.get("character") == character_id
        ]

        return sorted(character_moments, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def _assess_character_trajectory(self, history: List[Dict[str, Any]]) -> str:
        """Assess the overall trajectory of a character."""
        if len(history) < 3:
            return "developing"

        # Simple trajectory based on mood and energy trends
        recent_moods = [h.get("mood", "neutral") for h in history[-5:]]
        recent_energy = [h.get("energy", 0.5) for h in history[-5:]]

        positive_moods = ["happy", "excited", "content", "ecstatic", "love"]
        negative_moods = ["sad", "angry", "depressed", "frustrated", "anxious"]

        positive_count = sum(1 for mood in recent_moods if mood in positive_moods)
        negative_count = sum(1 for mood in recent_moods if mood in negative_moods)

        avg_energy_recent = sum(recent_energy) / len(recent_energy)
        avg_energy_earlier = sum(h.get("energy", 0.5) for h in history[-10:-5]) / max(1, min(5, len(history) - 5))

        if positive_count > negative_count and avg_energy_recent > avg_energy_earlier:
            return "improving"
        elif negative_count > positive_count and avg_energy_recent < avg_energy_earlier:
            return "declining"
        elif positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "struggling"
        else:
            return "stable"


class LocationObserver:
    """Observes activities and interactions in specific locations."""

    def __init__(self):
        self.location_activities: Dict[str, List[Dict[str, Any]]] = {}
        self.social_interactions: List[Dict[str, Any]] = []

    def record_location_activity(self, location: str, activity: Dict[str, Any]):
        """Record an activity happening in a location."""
        if location not in self.location_activities:
            self.location_activities[location] = []

        activity_record = {
            **activity,
            "timestamp": time.time(),
            "location": location
        }

        self.location_activities[location].append(activity_record)

        # Keep location history manageable
        if len(self.location_activities[location]) > 100:
            self.location_activities[location] = self.location_activities[location][-75:]

        # Detect social interactions
        if activity.get("participants") and len(activity.get("participants", [])) > 1:
            self.social_interactions.append(activity_record)

    def get_location_summary(self, location: str) -> Dict[str, Any]:
        """Get summary of what's been happening in a location."""
        activities = self.location_activities.get(location, [])

        if not activities:
            return {"status": "quiet", "recent_activities": []}

        recent_activities = activities[-10:]
        current_occupants = self._get_current_occupants(activities)

        return {
            "status": "active" if current_occupants else "quiet",
            "current_occupants": current_occupants,
            "recent_activities": recent_activities,
            "activity_level": self._calculate_activity_level(activities),
            "dominant_activity_types": self._get_dominant_activity_types(activities)
        }

    def _get_current_occupants(self, activities: List[Dict[str, Any]]) -> List[str]:
        """Get characters currently in the location."""
        current_time = time.time()
        recent_threshold = current_time - 300  # 5 minutes

        recent_activities = [a for a in activities if a["timestamp"] > recent_threshold]
        occupants = set()

        for activity in recent_activities:
            participants = activity.get("participants", [])
            occupants.update(participants)

        return list(occupants)

    def _calculate_activity_level(self, activities: List[Dict[str, Any]]) -> str:
        """Calculate how active a location has been."""
        if not activities:
            return "quiet"

        current_time = time.time()
        hour_ago = current_time - 3600

        recent_activities = [a for a in activities if a["timestamp"] > hour_ago]
        activity_count = len(recent_activities)

        if activity_count > 20:
            return "very_active"
        elif activity_count > 10:
            return "active"
        elif activity_count > 3:
            return "moderate"
        else:
            return "quiet"

    def _get_dominant_activity_types(self, activities: List[Dict[str, Any]]) -> List[str]:
        """Get most common activity types in location."""
        activity_types = [a.get("type", "unknown") for a in activities[-20:]]

        from collections import Counter
        counter = Counter(activity_types)
        return [activity for activity, count in counter.most_common(3)]


class EventTimeline:
    """Manages the timeline of significant events in the game."""

    def __init__(self, max_events: int = 100):
        self.events: List[Dict[str, Any]] = []
        self.max_events = max_events

    def add_event(self, event: Dict[str, Any]):
        """Add an event to the timeline."""
        event_record = {
            **event,
            "timestamp": time.time(),
            "id": f"event_{int(time.time() * 1000)}"  # Unique ID
        }

        self.events.append(event_record)

        # Keep timeline manageable
        if len(self.events) > self.max_events:
            self.events = self.events[-int(self.max_events * 0.8):]

    def get_filtered_timeline(self, filter_type: TimelineFilter = TimelineFilter.ALL,
                            limit: int = 50) -> List[Dict[str, Any]]:
        """Get filtered timeline of events."""
        filtered_events = self.events

        if filter_type != TimelineFilter.ALL:
            filtered_events = self._apply_filter(filtered_events, filter_type)

        # Sort by timestamp (most recent first)
        filtered_events = sorted(filtered_events, key=lambda x: x["timestamp"], reverse=True)

        return filtered_events[:limit]

    def _apply_filter(self, events: List[Dict[str, Any]], filter_type: TimelineFilter) -> List[Dict[str, Any]]:
        """Apply filter to events."""
        if filter_type == TimelineFilter.SOCIAL:
            return [e for e in events if e.get("category") == "social" or
                   len(e.get("participants", [])) > 1]

        elif filter_type == TimelineFilter.DRAMATIC:
            return [e for e in events if e.get("drama_level", 0) > 0.6]

        elif filter_type == TimelineFilter.FUNNY:
            return [e for e in events if e.get("humor_level", 0) > 0.5]

        elif filter_type == TimelineFilter.SIGNIFICANT:
            return [e for e in events if e.get("significance", 0) > 0.7]

        return events


class ObservationPanel:
    """Main observation panel UI controller."""

    def __init__(self, config: ObservationConfig = None):
        self.config = config or ObservationConfig()
        self.current_mode = ObservationMode.OVERVIEW

        # Observation components
        self.character_observer = CharacterObserver()
        self.location_observer = LocationObserver()
        self.event_timeline = EventTimeline(self.config.max_timeline_events)

        # UI state
        self.focused_character = None
        self.focused_location = None
        self.timeline_filter = TimelineFilter.ALL

        # Auto-follow settings
        self.auto_follow_active = self.config.auto_follow_drama
        self.drama_threshold = 0.7

    def set_observation_mode(self, mode: ObservationMode):
        """Set the observation mode."""
        self.current_mode = mode

    def focus_character(self, character_id: str):
        """Focus observation on a specific character."""
        self.focused_character = character_id
        self.current_mode = ObservationMode.CHARACTER_FOCUS

    def focus_location(self, location: str):
        """Focus observation on a specific location."""
        self.focused_location = location
        self.current_mode = ObservationMode.LOCATION_FOCUS

    def update_game_state(self, game_state: Dict[str, Any]):
        """Update observers with new game state."""
        characters = game_state.get("characters", {})
        world_state = game_state.get("world", {})

        # Update character states
        for char_id, char_state in characters.items():
            self.character_observer.update_character_state(char_id, char_state)

        # Update location activities
        for location, location_info in world_state.get("locations", {}).items():
            if "recent_activity" in location_info:
                self.location_observer.record_location_activity(
                    location, location_info["recent_activity"]
                )

        # Add significant events to timeline
        if "recent_events" in game_state:
            for event in game_state["recent_events"]:
                if event.get("significance", 0) > 0.3:  # Only significant events
                    self.event_timeline.add_event(event)

        # Check for auto-follow triggers
        if self.auto_follow_active:
            self._check_auto_follow_triggers(game_state)

    def _check_auto_follow_triggers(self, game_state: Dict[str, Any]):
        """Check if we should automatically switch focus due to drama."""
        if "recent_events" in game_state:
            for event in game_state["recent_events"]:
                drama_level = event.get("drama_level", 0)
                if drama_level > self.drama_threshold:
                    # Auto-focus on dramatic events
                    participants = event.get("participants", [])
                    if participants:
                        self.focus_character(participants[0])
                    break

    def get_observation_interface(self) -> Dict[str, Any]:
        """Get the main observation interface based on current mode."""
        base_interface = {
            "current_mode": self.current_mode.value,
            "timeline": self.event_timeline.get_filtered_timeline(self.timeline_filter, 20),
            "timeline_filter": self.timeline_filter.value,
            "auto_follow_active": self.auto_follow_active
        }

        if self.current_mode == ObservationMode.OVERVIEW:
            base_interface["overview"] = self._get_overview_interface()

        elif self.current_mode == ObservationMode.CHARACTER_FOCUS:
            if self.focused_character:
                base_interface["character_focus"] = self._get_character_focus_interface()

        elif self.current_mode == ObservationMode.LOCATION_FOCUS:
            if self.focused_location:
                base_interface["location_focus"] = self._get_location_focus_interface()

        elif self.current_mode == ObservationMode.RELATIONSHIP_VIEW:
            base_interface["relationship_view"] = self._get_relationship_interface()

        elif self.current_mode == ObservationMode.STORY_MODE:
            base_interface["story_mode"] = self._get_story_interface()

        return base_interface

    def _get_overview_interface(self) -> Dict[str, Any]:
        """Get overview interface showing all characters and locations."""
        overview = {
            "characters": {},
            "locations": {},
            "current_dramas": [],
            "system_status": {}
        }

        # Character summaries
        for char_id in self.character_observer.character_histories.keys():
            char_summary = self.character_observer.get_character_summary(char_id)
            overview["characters"][char_id] = {
                "current_state": char_summary.get("current_state", {}),
                "trajectory": char_summary.get("character_trajectory", "stable"),
                "recent_activity": char_summary.get("activity_pattern", {}).get("current_activity", "idle")
            }

        # Location summaries
        for location in self.location_observer.location_activities.keys():
            overview["locations"][location] = self.location_observer.get_location_summary(location)

        # Current ongoing dramas
        recent_events = self.event_timeline.get_filtered_timeline(TimelineFilter.DRAMATIC, 5)
        overview["current_dramas"] = [
            event for event in recent_events
            if time.time() - event["timestamp"] < 1800  # Last 30 minutes
        ]

        return overview

    def _get_character_focus_interface(self) -> Dict[str, Any]:
        """Get character-focused interface."""
        char_summary = self.character_observer.get_character_summary(self.focused_character)

        return {
            "character_id": self.focused_character,
            "detailed_state": char_summary.get("current_state", {}),
            "mood_history": self._get_mood_history(self.focused_character),
            "recent_interactions": self._get_character_interactions(self.focused_character),
            "character_arc": char_summary.get("character_trajectory", "developing"),
            "significant_moments": char_summary.get("recent_significant_moments", []),
            "relationships": self._get_character_relationships(self.focused_character),
            "internal_thoughts": char_summary.get("current_state", {}).get("internal_monologue", "...")
        }

    def _get_location_focus_interface(self) -> Dict[str, Any]:
        """Get location-focused interface."""
        location_summary = self.location_observer.get_location_summary(self.focused_location)

        return {
            "location": self.focused_location,
            "current_occupants": location_summary.get("current_occupants", []),
            "activity_level": location_summary.get("activity_level", "quiet"),
            "recent_events": location_summary.get("recent_activities", []),
            "atmosphere": self._assess_location_atmosphere(self.focused_location),
            "interesting_objects": self._get_location_objects(self.focused_location)
        }

    def _get_relationship_interface(self) -> Dict[str, Any]:
        """Get relationship-focused interface."""
        return {
            "relationship_map": self._build_relationship_map(),
            "active_relationships": self._get_active_relationships(),
            "relationship_changes": self._get_recent_relationship_changes(),
            "social_clusters": self._identify_social_clusters()
        }

    def _get_story_interface(self) -> Dict[str, Any]:
        """Get story-mode interface focusing on narrative elements."""
        return {
            "current_storylines": self._identify_storylines(),
            "character_arcs": self._get_all_character_arcs(),
            "narrative_themes": self._identify_narrative_themes(),
            "story_suggestions": self._generate_story_suggestions()
        }

    # Helper methods for interface builders
    def _get_mood_history(self, character_id: str) -> List[Dict[str, Any]]:
        """Get mood history for a character."""
        history = self.character_observer.character_histories.get(character_id, [])
        return [{"timestamp": h["timestamp"], "mood": h.get("mood", "neutral")} for h in history[-20:]]

    def _get_character_interactions(self, character_id: str) -> List[Dict[str, Any]]:
        """Get recent interactions involving a character."""
        interactions = []
        for interaction in self.location_observer.social_interactions[-20:]:
            if character_id in interaction.get("participants", []):
                interactions.append(interaction)
        return interactions

    def _get_character_relationships(self, character_id: str) -> Dict[str, Any]:
        """Get character's relationships."""
        # This would interface with the memory system in a real implementation
        return {"placeholder": "relationship_data"}

    def _assess_location_atmosphere(self, location: str) -> str:
        """Assess the current atmosphere of a location."""
        summary = self.location_observer.get_location_summary(location)
        activity_level = summary.get("activity_level", "quiet")

        if activity_level == "very_active":
            return "bustling"
        elif activity_level == "active":
            return "lively"
        elif activity_level == "moderate":
            return "pleasant"
        else:
            return "peaceful"

    def _get_location_objects(self, location: str) -> List[str]:
        """Get interesting objects in a location."""
        # This would interface with the environment system
        return ["placeholder_object"]

    # Story mode helpers
    def _identify_storylines(self) -> List[Dict[str, Any]]:
        """Identify ongoing storylines."""
        return [{"title": "Placeholder Storyline", "participants": [], "status": "developing"}]

    def _get_all_character_arcs(self) -> Dict[str, str]:
        """Get character arc status for all characters."""
        arcs = {}
        for char_id in self.character_observer.character_histories.keys():
            summary = self.character_observer.get_character_summary(char_id)
            arcs[char_id] = summary.get("character_trajectory", "developing")
        return arcs

    def _identify_narrative_themes(self) -> List[str]:
        """Identify themes emerging in the narrative."""
        return ["friendship", "self_discovery", "conflict_resolution"]

    def _generate_story_suggestions(self) -> List[str]:
        """Generate suggestions for interesting story developments."""
        return ["Character A should confront their fears", "The group needs a celebration"]

    # Relationship analysis helpers
    def _build_relationship_map(self) -> Dict[str, Any]:
        """Build a map of character relationships."""
        return {"placeholder": "relationship_map"}

    def _get_active_relationships(self) -> List[Dict[str, Any]]:
        """Get currently active/developing relationships."""
        return []

    def _get_recent_relationship_changes(self) -> List[Dict[str, Any]]:
        """Get recent changes in relationships."""
        return []

    def _identify_social_clusters(self) -> List[List[str]]:
        """Identify groups of characters who interact frequently."""
        return []

    def set_timeline_filter(self, filter_type: TimelineFilter):
        """Set the timeline filter."""
        self.timeline_filter = filter_type

    def toggle_auto_follow(self):
        """Toggle automatic drama following."""
        self.auto_follow_active = not self.auto_follow_active