from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any
import json

class Character(BaseModel):
    name: str
    role: str
    secret: str
    arc: str

class Location(BaseModel):
    name: str
    description: str
    visual_description: str

class EmotionalBeat(BaseModel):
    episode: int
    tone: str
    key_moment: str

class Episode(BaseModel):
    episode_number: int
    title: str
    brief: str
    cliffhanger: str
    locations_used: List[str] = Field(default_factory=list)
    characters_featured: List[str] = Field(default_factory=list)
    central_question: str = ""
    picks_up_from: str = ""
    thread_carried_forward: str = ""

class StoryBible(BaseModel):
    story_title: str
    story_summary: str
    thematic_quote: str
    quote_author: str = ""
    quote_source: str = ""
    total_episodes: int
    characters: List[Character] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    emotional_beats: List[EmotionalBeat] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_consistency(self):
        episode_count = len(self.episodes)
        if episode_count < 5:
            raise ValueError(
                f"Story bible must contain 5–7 episodes, but got {episode_count}. "
                f"Claude must return all episode objects in the 'episodes' array."
            )
        if episode_count > 7:
            self.episodes = self.episodes[:7]
            episode_count = 7
        self.total_episodes = episode_count
        return self

def validate_json_response(json_str: str, schema: BaseModel) -> Dict[str, Any]:
    from utils.errors import ValidationException
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationException(f"Invalid JSON: {e}")
    try:
        validated = schema(**data)
        return validated.model_dump()
    except Exception as e:
        raise ValidationException(f"Schema validation failed: {e}")