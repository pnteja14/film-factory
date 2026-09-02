import json
import anthropic
from json_repair import repair_json
from pydantic import ValidationError
from utils.logger import AgentLogger
from utils.errors import StoryArchitectException, RetryableException
from utils.retry import retry_with_backoff
from utils.validators import StoryBible

logger = AgentLogger("StoryArchitect")

def read_brand_voice() -> str:
    """Read brand voice guidelines."""
    try:
        with open("memory/brand_voice.md", "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                raise StoryArchitectException("brand_voice.md is empty")
            return content
    except FileNotFoundError:
        raise StoryArchitectException("brand_voice.md not found in memory/ folder")
    except Exception as e:
        raise StoryArchitectException(f"Failed to read brand_voice.md: {e}")

@retry_with_backoff(max_retries=5, base_wait=10.0, max_wait=60.0, retryable_exceptions=(RetryableException,))
def call_claude_for_story_plan(prompt: str) -> str:
    """Call Claude API with retry logic."""
    try:
        import os
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API call failed: {e}", e)
        raise RetryableException(f"Claude call failed: {e}")

def extract_json_from_response(raw_response: str) -> dict:
    """
    Extract and repair JSON from LLM response.
    
    Handles markdown code blocks, extra text before/after JSON.
    """
    try:
        # Remove markdown code blocks
        raw_response = raw_response.replace("```json", "").replace("```", "")
        
        # Find JSON boundaries
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        
        if start == -1 or end == 0:
            raise StoryArchitectException("No JSON found in response")
        
        json_str = raw_response[start:end]
        
        # Attempt repair
        repaired = repair_json(json_str)
        return json.loads(repaired)
        
    except json.JSONDecodeError as e:
        raise StoryArchitectException(f"JSON parsing failed: {e}")
    except Exception as e:
        raise StoryArchitectException(f"JSON extraction failed: {e}")

def validate_story_plan(story_plan: dict) -> dict:
    """
    Validate story plan against Pydantic schema.
    
    Returns validated and normalized data.
    """
    try:
        # Ensure all required fields exist
        # Auto-generate episodes if missing
        if "episodes" not in story_plan:
            total = story_plan.get("total_episodes", len(story_plan.get("emotional_beats", [])))
            total = max(5, min(7, total))
            story_plan["episodes"] = [
                {
                    "episode_number": i + 1,
                    "title": f"Episode {i + 1}",
                    "brief": (story_plan.get("emotional_beats") or [{}])[i].get("key_moment", "") if i < len(story_plan.get("emotional_beats") or []) else "",
                    "cliffhanger": "To be continued...",
                    "locations_used": [story_plan["locations"][0]["name"]] if story_plan.get("locations") else [],
                    "characters_featured": [story_plan["characters"][0]["name"]] if story_plan.get("characters") else []
                }
                for i in range(total)
            ]
            logger.warning(f"episodes field missing — auto-generated {total} episodes")

        if not all(key in story_plan for key in [
            "story_title", "story_summary", "thematic_quote",
            "characters", "locations", "episodes", "emotional_beats"
        ]):
            missing = [k for k in [
                "story_title", "story_summary", "thematic_quote",
                "characters", "locations", "episodes", "emotional_beats"
            ] if k not in story_plan]
            raise StoryArchitectException(f"Missing required fields: {missing}")
        
        # Validate total episodes
        total_episodes = story_plan.get("total_episodes", len(story_plan.get("episodes", [])))
        if not (5 <= total_episodes <= 7):
            total_episodes = min(7, max(5, total_episodes))
            logger.warning(f"Adjusted total_episodes to {total_episodes}")
            story_plan["total_episodes"] = total_episodes
        
        # Validate with Pydantic
        validated = StoryBible(**story_plan)
        return validated.model_dump()
        
    except ValidationError as e:
        raise StoryArchitectException(f"Story plan validation failed: {e}")
    except Exception as e:
        raise StoryArchitectException(f"Validation error: {e}")

def plan_story(context: dict) -> tuple[dict, dict]:
    """
    Agent 1: Generate complete story bible.
    
    Args:
        context: Current pipeline context
    
    Returns:
        Tuple of (updated context, story_plan dict)
    
    Raises:
        StoryArchitectException: If planning fails
    """
    logger.info("Starting story planning...")
    
    try:
        # Read brand voice
        brand_voice = read_brand_voice()
        logger.info("Brand voice loaded")
        
        # Build prompt
        prompt = f"""
You are the lead story architect for a serialized true crime short film series on Instagram.

CREATIVE BIBLE:
{brand_voice}

STORY IDEA:
{context["story_idea"]}

YOUR TASK:
Plan a complete multi-episode story arc and a detailed story bible that ensures consistency across all episodes.

REQUIREMENTS:
- Plan exactly between 5 to 7 episodes (strongly prefer 5 or 6, never fewer than 5)
- Every episode must end on a cliffhanger or quiet devastating revelation
- Emotional arc: curiosity → unease → emotional conflict
- Character names feel grounded and real
- Story is emotionally layered, not just plot driven
- Choose or write a thematic quote that captures the soul of the entire story
- Every episode metaphorically proves this quote is true without stating it
- Define at least 6 DISTINCT locations with varied atmospheres (interior/exterior, private/public, day/night)
- Every episode must use AT LEAST 4 different locations from the story bible — no episode may list fewer than 4
- Define every character's full arc across all episodes
- Map emotional beat for every episode

SERIALIZED ARC STRUCTURE — assign one role per episode:
- Episode 1 (ESTABLISH): introduce protagonist's world, plant the central question, deliver first revelation
- Middle episodes (ESCALATE): deepen mystery, layer in complications, contradict earlier assumptions
- Second-to-last episode (CRISIS): all threads collide, everything worsens, darkest moment of the story
- Final episode (RECKONING): emotional and factual resolution; central question answered; thematic quote returns

Each episode must include:
- What central question it answers or deepens
- Where it picks up from the previous episode's cliffhanger (empty for episode 1)
- What unresolved thread it hands to the next episode

RESPOND IN THIS EXACT JSON FORMAT AND NOTHING ELSE:
{{
    "story_title": "title here",
    "story_summary": "2 to 3 sentence summary",
    "thematic_quote": "the quote here",
    "quote_author": "First Last, fictional author",
    "quote_source": "Title of fictional work, year",
    "total_episodes": 5,
    "characters": [
        {{
            "name": "character name",
            "role": "their role in the story",
            "secret": "what they are hiding",
            "arc": "their full journey across all episodes"
        }}
    ],
    "locations": [
        {{
            "name": "location name",
            "description": "narrative description",
            "visual_description": "physical environment only — lighting, colors, textures, materials, objects, atmosphere. NO character names. NO people. NO camera movement. NO pronouns. Only describe what exists in the space itself as a static environment."
        }}
    ],
    "emotional_beats": [
        {{
            "episode": 1,
            "tone": "the emotional tone of this episode",
            "key_moment": "the most important moment"
        }}
    ],
    "episodes": [
        {{
            "episode_number": 1,
            "title": "episode title",
            "brief": "what happens in this episode",
            "cliffhanger": "how this episode ends",
            "locations_used": ["location name"],
            "characters_featured": ["character name"],
            "central_question": "what is this episode trying to answer or deepen?",
            "picks_up_from": "exact last line or moment from previous episode's cliffhanger — empty string for episode 1",
            "thread_carried_forward": "what unresolved question or tension the next episode must pick up"
        }}
    ]
}}
"""
        
        # Call Claude API — retry up to 3 times if episode count is wrong
        story_plan = None
        for attempt in range(3):
            logger.info(f"Calling Claude API for story plan (attempt {attempt + 1})...")
            raw_response = call_claude_for_story_plan(prompt)

            logger.info("Extracting and parsing JSON...")
            story_plan = extract_json_from_response(raw_response)

            actual_ep_count = len(story_plan.get("episodes", []))
            if 5 <= actual_ep_count <= 7:
                break

            logger.warning(
                f"Attempt {attempt + 1}: Claude returned {actual_ep_count} episode(s) "
                f"in 'episodes' array — need 5–7. Retrying..."
            )
            if attempt == 2:
                raise StoryArchitectException(
                    f"Claude returned {actual_ep_count} episode(s) after 3 attempts. "
                    f"Expected 5–7 fully defined episode objects."
                )

        # Validate
        logger.info("Validating story plan...")
        story_plan = validate_story_plan(story_plan)
        
        # Save to context
        context["story_title"] = story_plan["story_title"]
        context["story_summary"] = story_plan["story_summary"]
        context["thematic_quote"] = story_plan["thematic_quote"]
        context["quote_author"] = story_plan.get("quote_author", "")
        context["quote_source"] = story_plan.get("quote_source", "")
        context["characters"] = story_plan["characters"]
        context["total_episodes"] = story_plan["total_episodes"]
        
        # Build story bible
        context["story_bible"]["locations"] = story_plan.get("locations", [])
        
        # Build location visual lookup
        for loc in story_plan.get("locations", []):
            context["story_bible"]["location_visuals"][loc["name"]] = \
                loc.get("visual_description", loc.get("description", ""))
        
        # Character arcs
        context["story_bible"]["character_arcs"] = [
            {
                "name": c["name"],
                "arc": c["arc"],
                "secret": c["secret"]
            }
            for c in story_plan["characters"]
        ]
        
        # Emotional beats
        context["story_bible"]["emotional_beats"] = story_plan["emotional_beats"]
        
        # Build episode list
        context["episode_plans"] = story_plan["episodes"]
        context["episodes"] = [
            f"Episode {ep['episode_number']}: {ep['title']}. "
            f"{ep['brief']} "
            f"Cliffhanger: {ep['cliffhanger']} "
            f"Locations: {', '.join(ep.get('locations_used', []))}. "
            f"Characters: {', '.join(ep.get('characters_featured', []))}."
            for ep in story_plan["episodes"]
        ]
        
        logger.info(f"Story planning complete: '{story_plan['story_title']}' - {story_plan['total_episodes']} episodes")
        return context, story_plan
        
    except StoryArchitectException as e:
        logger.error(f"Story planning failed: {e}", e)
        context["errors"].append({
            "agent": "story_architect",
            "error": str(e)
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error in story planning: {e}", e)
        context["errors"].append({
            "agent": "story_architect",
            "error": f"Unexpected: {str(e)}"
        })
        raise StoryArchitectException(f"Unexpected error: {e}")