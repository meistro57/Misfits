"""
Environment - Game world management and location system.

This module handles the game world environment, locations, objects,
and environmental factors that affect character behavior.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
import time


class LocationType(Enum):
    """Types of locations in the game world."""
    LIVING_SPACE = "living_space"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    OUTDOOR = "outdoor"
    WORK_SPACE = "work_space"
    SOCIAL_SPACE = "social_space"
    STORAGE = "storage"
    ENTERTAINMENT = "entertainment"


class WeatherType(Enum):
    """Weather conditions that affect character moods and activities."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    FOGGY = "foggy"


@dataclass
class InteractableObject:
    """Objects in the world that characters can interact with."""
    object_id: str
    name: str
    location: str
    object_type: str
    interaction_types: List[str]  # e.g., ["use", "repair", "clean"]
    durability: float  # 0.0 to 1.0
    cleanliness: float  # 0.0 to 1.0
    functionality: float  # 0.0 to 1.0
    comfort_rating: float  # How comfortable/pleasant it is to use
    skill_requirements: Dict[str, float]  # Required skills to use effectively
    mood_effects: Dict[str, float]  # How using this affects character mood


@dataclass
class Location:
    """A location in the game world."""
    location_id: str
    name: str
    description: str
    location_type: LocationType
    capacity: int  # Maximum characters that can be here comfortably
    comfort_level: float  # Base comfort/happiness modifier
    privacy_level: float  # How private/secluded the location is
    noise_level: float  # How noisy the location is
    lighting: str  # bright, dim, dark, natural
    temperature: str  # hot, warm, comfortable, cool, cold
    connected_locations: List[str]  # Locations directly accessible from here
    objects: List[InteractableObject]
    environmental_effects: Dict[str, float]  # Effects on character needs/mood
    available_activities: List[str]  # What can be done here


class WorldEnvironment:
    """Main environment system managing the game world."""

    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, InteractableObject] = {}
        self.current_weather = WeatherType.SUNNY
        self.time_of_day = "morning"
        self.season = "spring"
        self.world_events: List[Dict[str, Any]] = []

        # Environmental factors
        self.global_temperature = 22.0  # Celsius
        self.air_quality = 0.8  # 0.0 to 1.0
        self.ambient_noise_level = 0.3  # 0.0 to 1.0

        self._initialize_default_locations()

    def _initialize_default_locations(self):
        """Set up the default house locations."""

        # Living Room
        living_room_objects = [
            InteractableObject(
                "tv_living", "Television", "living_room", "entertainment",
                ["watch", "turn_on", "turn_off"],
                durability=0.9, cleanliness=0.8, functionality=0.95,
                comfort_rating=0.7, skill_requirements={},
                mood_effects={"fun": 0.2, "social": 0.1}
            ),
            InteractableObject(
                "couch_living", "Comfortable Couch", "living_room", "furniture",
                ["sit", "lie_down", "nap"],
                durability=0.8, cleanliness=0.7, functionality=1.0,
                comfort_rating=0.9, skill_requirements={},
                mood_effects={"energy": 0.1, "comfort": 0.3}
            ),
            InteractableObject(
                "bookshelf", "Bookshelf", "living_room", "storage",
                ["read", "browse", "organize"],
                durability=0.95, cleanliness=0.8, functionality=1.0,
                comfort_rating=0.4, skill_requirements={"logic": 0.2},
                mood_effects={"intellectual": 0.3, "calm": 0.2}
            )
        ]

        self.locations["living_room"] = Location(
            location_id="living_room",
            name="Living Room",
            description="A cozy living room with comfortable seating and entertainment",
            location_type=LocationType.LIVING_SPACE,
            capacity=4,
            comfort_level=0.8,
            privacy_level=0.3,
            noise_level=0.4,
            lighting="natural",
            temperature="comfortable",
            connected_locations=["kitchen", "bedroom", "garden"],
            objects=living_room_objects,
            environmental_effects={"social": 0.1, "comfort": 0.2},
            available_activities=["socialize", "watch_tv", "read", "relax"]
        )

        # Kitchen
        kitchen_objects = [
            InteractableObject(
                "stove", "Gas Stove", "kitchen", "appliance",
                ["cook", "turn_on", "clean"],
                durability=0.9, cleanliness=0.6, functionality=0.95,
                comfort_rating=0.3, skill_requirements={"cooking": 0.3},
                mood_effects={"hunger": 0.4, "accomplishment": 0.2}
            ),
            InteractableObject(
                "fridge", "Refrigerator", "kitchen", "appliance",
                ["get_food", "store_food", "check_contents"],
                durability=0.95, cleanliness=0.8, functionality=1.0,
                comfort_rating=0.2, skill_requirements={},
                mood_effects={"hunger": 0.3}
            ),
            InteractableObject(
                "dining_table", "Dining Table", "kitchen", "furniture",
                ["eat", "sit", "work"],
                durability=0.9, cleanliness=0.7, functionality=1.0,
                comfort_rating=0.6, skill_requirements={},
                mood_effects={"social": 0.2, "hunger": 0.1}
            )
        ]

        self.locations["kitchen"] = Location(
            location_id="kitchen",
            name="Kitchen",
            description="A functional kitchen with modern appliances",
            location_type=LocationType.KITCHEN,
            capacity=3,
            comfort_level=0.6,
            privacy_level=0.4,
            noise_level=0.6,
            lighting="bright",
            temperature="warm",
            connected_locations=["living_room", "garden"],
            objects=kitchen_objects,
            environmental_effects={"hunger": 0.1},
            available_activities=["cook", "eat", "clean", "socialize"]
        )

        # Bedroom
        bedroom_objects = [
            InteractableObject(
                "bed", "Comfortable Bed", "bedroom", "furniture",
                ["sleep", "nap", "sit", "make_bed"],
                durability=0.9, cleanliness=0.7, functionality=1.0,
                comfort_rating=1.0, skill_requirements={},
                mood_effects={"energy": 0.5, "comfort": 0.4}
            ),
            InteractableObject(
                "wardrobe", "Wardrobe", "bedroom", "storage",
                ["change_clothes", "organize", "browse"],
                durability=0.95, cleanliness=0.8, functionality=1.0,
                comfort_rating=0.3, skill_requirements={},
                mood_effects={"confidence": 0.1, "hygiene": 0.1}
            ),
            InteractableObject(
                "desk", "Study Desk", "bedroom", "furniture",
                ["work", "study", "write", "use_computer"],
                durability=0.9, cleanliness=0.6, functionality=0.9,
                comfort_rating=0.4, skill_requirements={"logic": 0.1},
                mood_effects={"productivity": 0.3}
            )
        ]

        self.locations["bedroom"] = Location(
            location_id="bedroom",
            name="Bedroom",
            description="A private bedroom with a comfortable bed and personal space",
            location_type=LocationType.BEDROOM,
            capacity=1,
            comfort_level=0.9,
            privacy_level=0.9,
            noise_level=0.2,
            lighting="dim",
            temperature="comfortable",
            connected_locations=["living_room", "bathroom"],
            objects=bedroom_objects,
            environmental_effects={"privacy": 0.3, "energy": 0.2},
            available_activities=["sleep", "work", "study", "change_clothes", "relax"]
        )

        # Bathroom
        bathroom_objects = [
            InteractableObject(
                "shower", "Shower", "bathroom", "hygiene",
                ["shower", "clean", "maintenance"],
                durability=0.9, cleanliness=0.6, functionality=0.9,
                comfort_rating=0.7, skill_requirements={},
                mood_effects={"hygiene": 0.5, "energy": 0.2}
            ),
            InteractableObject(
                "toilet", "Toilet", "bathroom", "hygiene",
                ["use", "clean"],
                durability=0.9, cleanliness=0.5, functionality=0.95,
                comfort_rating=0.1, skill_requirements={},
                mood_effects={"relief": 0.3}
            ),
            InteractableObject(
                "mirror", "Bathroom Mirror", "bathroom", "hygiene",
                ["look", "groom", "practice_expressions"],
                durability=1.0, cleanliness=0.7, functionality=1.0,
                comfort_rating=0.2, skill_requirements={},
                mood_effects={"confidence": 0.1, "vanity": 0.1}
            )
        ]

        self.locations["bathroom"] = Location(
            location_id="bathroom",
            name="Bathroom",
            description="A clean bathroom with essential hygiene facilities",
            location_type=LocationType.BATHROOM,
            capacity=1,
            comfort_level=0.5,
            privacy_level=1.0,
            noise_level=0.3,
            lighting="bright",
            temperature="cool",
            connected_locations=["bedroom"],
            objects=bathroom_objects,
            environmental_effects={"hygiene": 0.3, "privacy": 0.4},
            available_activities=["shower", "groom", "use_toilet", "clean"]
        )

        # Garden
        garden_objects = [
            InteractableObject(
                "garden_bed", "Vegetable Garden", "garden", "nature",
                ["plant", "water", "harvest", "weed"],
                durability=0.7, cleanliness=0.9, functionality=0.8,
                comfort_rating=0.6, skill_requirements={"gardening": 0.4},
                mood_effects={"nature": 0.4, "accomplishment": 0.2}
            ),
            InteractableObject(
                "outdoor_table", "Patio Table", "garden", "furniture",
                ["sit", "eat_outside", "work_outside"],
                durability=0.8, cleanliness=0.7, functionality=1.0,
                comfort_rating=0.7, skill_requirements={},
                mood_effects={"nature": 0.2, "social": 0.1}
            )
        ]

        self.locations["garden"] = Location(
            location_id="garden",
            name="Garden",
            description="A small outdoor garden with plants and fresh air",
            location_type=LocationType.OUTDOOR,
            capacity=3,
            comfort_level=0.7,
            privacy_level=0.6,
            noise_level=0.2,
            lighting="natural",
            temperature="varies",
            connected_locations=["living_room", "kitchen"],
            objects=garden_objects,
            environmental_effects={"nature": 0.3, "air_quality": 0.2},
            available_activities=["garden", "relax_outside", "exercise", "socialize"]
        )

        # Register objects in global object dictionary
        for location in self.locations.values():
            for obj in location.objects:
                self.objects[obj.object_id] = obj

    def get_location(self, location_id: str) -> Optional[Location]:
        """Get a location by ID."""
        return self.locations.get(location_id)

    def get_object(self, object_id: str) -> Optional[InteractableObject]:
        """Get an object by ID."""
        return self.objects.get(object_id)

    def get_objects_in_location(self, location_id: str) -> List[InteractableObject]:
        """Get all objects in a specific location."""
        location = self.get_location(location_id)
        return location.objects if location else []

    def get_connected_locations(self, location_id: str) -> List[str]:
        """Get locations directly connected to the given location."""
        location = self.get_location(location_id)
        return location.connected_locations if location else []

    def can_move_between(self, from_location: str, to_location: str) -> bool:
        """Check if direct movement is possible between two locations."""
        from_loc = self.get_location(from_location)
        if not from_loc:
            return False
        return to_location in from_loc.connected_locations

    def get_available_activities(self, location_id: str) -> List[str]:
        """Get all available activities at a location."""
        location = self.get_location(location_id)
        if not location:
            return []

        activities = location.available_activities.copy()

        # Add object-specific activities
        for obj in location.objects:
            for interaction in obj.interaction_types:
                activity = f"{interaction}_{obj.object_type}"
                if activity not in activities:
                    activities.append(activity)

        return activities

    def update_weather(self):
        """Update weather conditions (called periodically)."""
        weather_transitions = {
            WeatherType.SUNNY: [WeatherType.CLOUDY, WeatherType.SUNNY, WeatherType.SUNNY],
            WeatherType.CLOUDY: [WeatherType.SUNNY, WeatherType.RAINY, WeatherType.CLOUDY],
            WeatherType.RAINY: [WeatherType.CLOUDY, WeatherType.STORMY, WeatherType.RAINY],
            WeatherType.STORMY: [WeatherType.RAINY, WeatherType.CLOUDY],
            WeatherType.SNOWY: [WeatherType.CLOUDY, WeatherType.SNOWY],
            WeatherType.FOGGY: [WeatherType.CLOUDY, WeatherType.SUNNY]
        }

        possible_weather = weather_transitions.get(self.current_weather, [WeatherType.SUNNY])
        self.current_weather = random.choice(possible_weather)

    def get_weather_effects(self) -> Dict[str, float]:
        """Get current weather effects on character mood."""
        weather_effects = {
            WeatherType.SUNNY: {"mood": 0.2, "energy": 0.1},
            WeatherType.CLOUDY: {"mood": -0.05},
            WeatherType.RAINY: {"mood": -0.1, "indoor_preference": 0.3},
            WeatherType.STORMY: {"mood": -0.2, "anxiety": 0.1, "indoor_preference": 0.5},
            WeatherType.SNOWY: {"mood": 0.1, "cozy_feeling": 0.2, "indoor_preference": 0.4},
            WeatherType.FOGGY: {"mood": -0.1, "mystery": 0.1}
        }

        return weather_effects.get(self.current_weather, {})

    def interact_with_object(self, object_id: str, interaction_type: str,
                           character_skills: Dict[str, float]) -> Dict[str, Any]:
        """Process character interaction with an object."""
        obj = self.get_object(object_id)
        if not obj:
            return {"success": False, "message": "Object not found"}

        if interaction_type not in obj.interaction_types:
            return {"success": False, "message": f"Cannot {interaction_type} this object"}

        # Check skill requirements
        skill_success = True
        for skill, required_level in obj.skill_requirements.items():
            character_level = character_skills.get(skill, 0.0)
            if character_level < required_level:
                skill_success = False
                break

        # Calculate success based on object condition and character skill
        base_success_rate = obj.functionality * obj.durability
        if skill_success:
            base_success_rate *= 1.2  # Bonus for having required skills

        success = random.random() < base_success_rate

        result = {
            "success": success,
            "object_id": object_id,
            "interaction": interaction_type,
            "mood_effects": obj.mood_effects.copy() if success else {},
            "skill_gained": 0.01 if success else 0.0  # Small skill improvement
        }

        if success:
            result["message"] = f"Successfully {interaction_type} the {obj.name}"
            # Slight wear on object
            obj.durability = max(0.1, obj.durability - 0.005)
        else:
            result["message"] = f"Failed to {interaction_type} the {obj.name}"
            # More wear on failed attempts
            obj.durability = max(0.1, obj.durability - 0.01)

        return result

    def get_location_mood_effects(self, location_id: str, character_count: int) -> Dict[str, float]:
        """Get mood effects from being in a location."""
        location = self.get_location(location_id)
        if not location:
            return {}

        effects = location.environmental_effects.copy()

        # Add base comfort effect
        effects["comfort"] = effects.get("comfort", 0.0) + location.comfort_level * 0.1

        # Overcrowding penalty
        if character_count > location.capacity:
            overcrowding_penalty = (character_count - location.capacity) * 0.1
            effects["stress"] = effects.get("stress", 0.0) + overcrowding_penalty
            effects["social"] = effects.get("social", 0.0) - overcrowding_penalty * 0.5

        # Privacy effects
        if character_count == 1 and location.privacy_level > 0.7:
            effects["privacy"] = effects.get("privacy", 0.0) + 0.2

        # Weather effects for outdoor locations
        if location.location_type == LocationType.OUTDOOR:
            weather_effects = self.get_weather_effects()
            for effect, value in weather_effects.items():
                effects[effect] = effects.get(effect, 0.0) + value

        return effects

    def trigger_environmental_event(self, event_type: str) -> Dict[str, Any]:
        """Trigger a world environmental event."""
        events = {
            "power_outage": {
                "description": "The power goes out across the house",
                "effects": {"all_electronics_disabled": True, "lighting": "dark"},
                "duration": 3600,  # 1 hour in seconds
                "mood_impact": {"stress": 0.2, "adventure": 0.1}
            },
            "water_leak": {
                "description": "A pipe bursts in the bathroom",
                "effects": {"bathroom_flooded": True, "water_damage": True},
                "duration": 1800,  # 30 minutes
                "mood_impact": {"stress": 0.3, "cleanup_needed": True}
            },
            "noise_complaint": {
                "description": "Neighbors complain about noise levels",
                "effects": {"noise_restriction": True},
                "duration": 7200,  # 2 hours
                "mood_impact": {"social_anxiety": 0.2}
            },
            "delivery_arrival": {
                "description": "A package arrives at the door",
                "effects": {"package_available": True},
                "duration": 0,  # Instant
                "mood_impact": {"excitement": 0.3, "curiosity": 0.2}
            }
        }

        event = events.get(event_type, {})
        if event:
            self.world_events.append({
                "type": event_type,
                "start_time": time.time(),
                "duration": event["duration"],
                "active": True,
                **event
            })

        return event

    def update_world_events(self):
        """Update active world events and remove expired ones."""
        current_time = time.time()

        for event in self.world_events:
            if event["active"] and event["duration"] > 0:
                elapsed_time = current_time - event["start_time"]
                if elapsed_time >= event["duration"]:
                    event["active"] = False

        # Remove old inactive events (keep last 10 for history)
        inactive_events = [e for e in self.world_events if not e["active"]]
        if len(inactive_events) > 10:
            self.world_events = [e for e in self.world_events if e["active"]] + inactive_events[-10:]

    def get_active_events(self) -> List[Dict[str, Any]]:
        """Get currently active world events."""
        return [event for event in self.world_events if event["active"]]

    def get_world_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world state."""
        return {
            "weather": self.current_weather.value,
            "time_of_day": self.time_of_day,
            "season": self.season,
            "temperature": self.global_temperature,
            "air_quality": self.air_quality,
            "active_events": len(self.get_active_events()),
            "total_locations": len(self.locations),
            "total_objects": len(self.objects)
        }