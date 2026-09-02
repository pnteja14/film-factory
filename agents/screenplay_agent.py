import anthropic
import os
from utils.logger import AgentLogger
from utils.errors import ScreenplayWriterException, RetryableException
from utils.retry import retry_with_backoff

logger = AgentLogger("ScreenplayWriter")

def read_brand_voice() -> str:
    """Read brand voice guidelines."""
    try:
        with open("memory/brand_voice.md", "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                raise ScreenplayWriterException("brand_voice.md is empty")
            return content
    except FileNotFoundError:
        raise ScreenplayWriterException("brand_voice.md not found")
    except Exception as e:
        raise ScreenplayWriterException(f"Failed to read brand_voice.md: {e}")

def build_character_guide(context: dict) -> str:
    """Build character guide from story bible."""
    try:
        if not context.get("story_bible", {}).get("character_arcs"):
            return "No character arcs available"
        
        lines = []
        for c in context["story_bible"]["character_arcs"]:
            lines.append(f"- {c['name']}: {c['arc']} (Secret: {c['secret']})")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to build character guide: {e}")
        return ""

def build_location_guide(context: dict) -> str:
    """Build location guide from story bible."""
    try:
        if not context.get("story_bible", {}).get("locations"):
            return "No locations available"
        
        lines = []
        for loc in context["story_bible"]["locations"]:
            lines.append(f"- {loc['name']}: {loc['description']}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to build location guide: {e}")
        return ""

def build_previous_summaries(context: dict) -> str:
    """Build summaries of previous episodes."""
    if not context.get("episode_summaries"):
        return "This is the first episode. No previous episodes."
    
    lines = []
    for i, summary in enumerate(context["episode_summaries"]):
        lines.append(f"Episode {i + 1} summary:\n{summary}")
    return "\n\n".join(lines)

def get_previous_cliffhanger(context: dict, episode_number: int) -> str:
    """Extract the last 2-3 sentences of the previous episode as the cliffhanger to continue from."""
    if episode_number == 0:
        return ""
    scripts = context.get("scripts", [])
    if episode_number - 1 >= len(scripts):
        return ""
    prev = scripts[episode_number - 1]
    if not prev:
        return ""
    sentences = [s.strip() for s in prev.replace("\n", " ").split(".") if s.strip()]
    tail = sentences[-3:] if len(sentences) >= 3 else sentences
    return ". ".join(tail) + "."

def get_thread_carried_forward(context: dict, episode_number: int) -> str:
    """Get the unresolved thread from the previous episode's plan."""
    if episode_number == 0:
        return ""
    plans = context.get("episode_plans", [])
    if episode_number - 1 >= len(plans):
        return ""
    return plans[episode_number - 1].get("thread_carried_forward", "")

def get_episode_narrative_role(context: dict, episode_number: int) -> str:
    """Map episode position to serialized arc role."""
    total = context.get("total_episodes", 5)
    if episode_number == 0:
        return "ESTABLISH — introduce the world, plant the central question, deliver the first revelation"
    elif episode_number == total - 1:
        return "RECKONING — answer the central question emotionally and factually; return to the thematic quote"
    elif episode_number == total - 2:
        return "CRISIS — all threads collide, everything worsens, the darkest moment of the story"
    else:
        return "ESCALATE — deepen the mystery, layer in complications, contradict earlier assumptions"

def clean_script(script: str) -> str:
    """Clean and normalize script formatting."""
    import re
    
    try:
        # Remove markdown code blocks
        script = script.replace("```", "").replace("`", "")
        
        # Remove artifact patterns
        script = script.replace("[Blank line]", "").replace("[blank line]", "")
        script = script.replace("Narration:", "").replace("NARRATION:", "")
        script = script.replace("Cliffhanger:", "").replace("CLIFFHANGER:", "")
        script = script.replace("Here is the revised script:", "")
        script = script.replace("Here is the corrected script:", "")
        
        # Ensure camera directions have proper spacing
        script = re.sub(r'([^\n])\[', r'\1\n\n[', script)  # Newline before bracket
        script = re.sub(r'\]([^\n])', r']\n\n\1', script)  # Newline after bracket
        
        # Remove excessive blank lines
        script = re.sub(r'\n{3,}', '\n\n', script)
        
        return script.strip()
    except Exception as e:
        logger.warning(f"Script cleaning failed: {e}")
        return script

def trim_to_word_limit(script: str, limit: int = 220) -> str:
    """Trim script to word limit while preserving structure."""
    words = script.split()
    
    if len(words) <= limit:
        return script
    
    # Find cut point at word limit
    trimmed = " ".join(words[:limit])
    
    # Find last complete sentence or bracket
    last_period = trimmed.rfind(".")
    last_bracket = trimmed.rfind("]")
    cut_point = max(last_period, last_bracket)
    
    if cut_point == -1:
        return trimmed
    
    return trimmed[:cut_point + 1]

@retry_with_backoff(max_retries=5, base_wait=10.0, max_wait=60.0, retryable_exceptions=(RetryableException,))
def call_claude_for_script(prompt: str) -> str:
    """Call Claude API for script generation."""
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API call failed: {e}", e)
        raise RetryableException(f"Claude call failed: {e}")
def generate_script(context: dict, episode_number: int) -> str:
    """
    Agent 2: Generate narration script for an episode.
    
    Args:
        context: Pipeline context with story plan from Agent 1
        episode_number: Zero-indexed episode number
    
    Returns:
        Formatted screenplay script
    
    Raises:
        ScreenplayWriterException: If generation fails
    """
    logger.info(f"Generating script for episode {episode_number + 1}...")
    
    try:
        # Validate inputs
        if episode_number >= len(context.get("episodes", [])):
            raise ScreenplayWriterException(
                f"Episode {episode_number + 1} out of range. "
                f"Total episodes: {context.get('total_episodes')}"
            )
        
        # Get emotional beat for this episode
        beats = context["story_bible"]["emotional_beats"]
        if episode_number < len(beats):
            emotional_beat = beats[episode_number]
        else:
            emotional_beat = {
                "tone": "tension and revelation",
                "key_moment": "a shocking discovery"
            }
        
        brand_voice = read_brand_voice()
        episode_info = context["episodes"][episode_number]
        character_guide = build_character_guide(context)
        location_guide = build_location_guide(context)
        previous_summaries = build_previous_summaries(context)
        previous_cliffhanger = get_previous_cliffhanger(context, episode_number)
        thread_to_continue = get_thread_carried_forward(context, episode_number)
        narrative_role = get_episode_narrative_role(context, episode_number)

        # Episode plan fields (optional, gracefully absent on first run)
        ep_plans = context.get("episode_plans", [])
        ep_plan = ep_plans[episode_number] if episode_number < len(ep_plans) else {}
        central_question = ep_plan.get("central_question", "")
        picks_up_from = ep_plan.get("picks_up_from", previous_cliffhanger)

        is_first = episode_number == 0
        is_last = episode_number == context.get("total_episodes", 1) - 1

        thematic_quote_instruction = ""
        if is_first:
            thematic_quote_instruction = (
                f'EPISODE 1 RULE: Open with this disclaimer and thematic quote before the narration begins:\n'
                f'"The following story is entirely fictional. Any resemblance to real persons, living or dead, '
                f'events or places is purely coincidental. Viewer discretion is advised."\n'
                f'Then: "{context["thematic_quote"]}" — {context.get("quote_author", "")}. '
                f'{context.get("quote_source", "")}\n'
                f'Do not count these words toward the word limit.'
            )
        elif is_last:
            thematic_quote_instruction = (
                f'FINAL EPISODE RULE: After the story concludes, close with the thematic quote one more time:\n'
                f'"{context["thematic_quote"]}" — {context.get("quote_author", "")}. '
                f'{context.get("quote_source", "")}\n'
                f'Do not count this toward the word limit.'
            )

        continuation_block = ""
        if picks_up_from:
            continuation_block = f"\nPICKS UP FROM (last moment of previous episode):\n{picks_up_from}\n"
        if thread_to_continue:
            continuation_block += f"\nTHREAD TO CONTINUE: {thread_to_continue}\n"

        prompt = f"""
You are a narration writer for a serialized true crime short film series on Instagram.

CREATIVE BIBLE:
{brand_voice}

STORY TITLE: {context['story_title']}
STORY SUMMARY: {context['story_summary']}
THEMATIC QUOTE: "{context['thematic_quote']}"

TOTAL EPISODES: {context['total_episodes']}
WRITING EPISODE: {episode_number + 1} of {context['total_episodes']}
NARRATIVE ROLE: {narrative_role}
{f"CENTRAL QUESTION THIS EPISODE MUST ANSWER: {central_question}" if central_question else ""}

CHARACTER GUIDE:
{character_guide}

LOCATION GUIDE:
{location_guide}

PREVIOUS EPISODES:
{previous_summaries}
{continuation_block}
EMOTIONAL BEAT FOR THIS EPISODE:
Tone: {emotional_beat['tone']}
Key moment: {emotional_beat['key_moment']}

EPISODE BRIEF:
{episode_info}

{thematic_quote_instruction}

MANDATORY 4-BEAT STRUCTURE (write in this exact order):
1. HOOK (25-35 words): A quiet, specific, unsettling detail that grabs immediately. No setup, no context. One image, one sensation, one fact.
2. ADVANCE (240-280 words): Deepen the world across multiple layers. Reveal something that changes what the audience thought they knew. Surface a contradiction, a buried fact, a character's hidden motive. Build tension through accumulation. Each paragraph adds a new dimension. Earn the next beat.
3. KEY REVEAL (60-80 words): One sharp pivot that recontextualises everything before it. Delivered in plain language across two to three sentences. No dramatic flair. The horror is in the facts themselves. Take your time and let it land.
4. CLIFFHANGER (50-65 words): Stop mid-truth across two to three sentences. Escalate to a specific visible or audible detail the audience cannot resolve. The last sentence must be cut short. Leave them with an image, not a question.

FORMAT RULES:
- Camera directions in square brackets on their own line, blank line before and after
- First person POV always — camera sees what I see, never the narrator
- Never describe the protagonist from outside — no faces, no hands unless reaching
- Use "I" and "me" for narrator — never character names in the narration itself
- 380 to 450 words total (not counting disclaimer or thematic quote)
- The cliffhanger must be a specific visible or spoken detail, not a vague question
- Every line earns its place — cut anything that doesn't push the story forward
- This is a 2-minute episode — give the audience time to breathe between revelations

Write only the script. Nothing else.
"""

        logger.info("Calling Claude API...")
        draft = call_claude_for_script(prompt)

        logger.info("Cleaning and formatting script...")
        formatted = clean_script(draft)
        final_script = trim_to_word_limit(formatted, limit=450)
        
        word_count = len(final_script.split())
        logger.info(f"Script generated: {word_count} words")
        
        return final_script
        
    except ScreenplayWriterException as e:
        logger.error(f"Script generation failed: {e}", e)
        context["errors"].append({
            "agent": "screenplay_writer",
            "episode": episode_number + 1,
            "error": str(e)
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", e)
        context["errors"].append({
            "agent": "screenplay_writer",
            "episode": episode_number + 1,
            "error": f"Unexpected: {str(e)}"
        })
        raise ScreenplayWriterException(f"Failed to generate script: {e}")

def summarize_episode(script: str, episode_number: int) -> str:
    """
    Summarize episode script for context feeding to next episode.
    
    Args:
        script: Full episode script
        episode_number: Zero-indexed episode number
    
    Returns:
        Summary bullet points
    """
    logger.info(f"Summarizing episode {episode_number + 1}...")
    
    try:
        prompt = f"""
Summarize this episode script in 4 to 5 bullet points.
Focus only on confirmed story facts:
- What happened
- What was discovered
- Where the episode ended
- What the cliffhanger revealed

SCRIPT:
{script}

Write only bullet points. Nothing else.
"""
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        summary = message.content[0].text.strip()
        logger.info(f"Episode {episode_number + 1} summarized")
        return summary
        
    except Exception as e:
        logger.warning(f"Failed to summarize episode {episode_number + 1}: {e}")
        return f"Episode {episode_number + 1} (summary unavailable)"