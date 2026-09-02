import os
import base64
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import AgentLogger
from utils.errors import AgentException, RetryableException
from utils.retry import retry_with_backoff

load_dotenv()

logger = AgentLogger("PromptCorrector")

VISUAL_ANALYSIS_PROMPT = """
You are analyzing an anchor image for a true crime short film.

Describe ONLY the physical objects and environmental details you can see:
- Specific furniture and fixtures
- Any documents, files, equipment visible
- Lighting sources and quality
- Colors and textures
- Architectural details

Be specific and concrete. Under 80 words.
Return only the description. Nothing else.
"""

PROMPT_REWRITE_TEMPLATE = """
You are rewriting an image generation prompt for FLUX.1-dev.

ORIGINAL PROMPT:
{original_prompt}

WHAT ACTUALLY EXISTS IN THE ANCHOR IMAGE:
{image_description}

RULES:
- Replace any referenced objects with objects confirmed in the image description
- If original references something not in the image, replace with the closest visible object
- Keep first person POV
- Keep the emotional tone and atmosphere
- Keep under 100 words
- Keep ending: cinematic grain, dark atmospheric, 4K
- Never invent new objects not in the image description
- No camera movement language — this is a still image prompt

Return only the rewritten prompt. Nothing else.
"""


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


@retry_with_backoff(max_retries=2, base_wait=3.0, retryable_exceptions=(RetryableException,))
def analyze_anchor_image(image_path: str) -> str:
    """Use Claude Vision to describe what objects exist in the anchor image."""
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        image_data = encode_image_to_base64(image_path)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": VISUAL_ANALYSIS_PROMPT
                    }
                ]
            }]
        )

        description = message.content[0].text.strip()
        logger.info(f"Image analyzed: {len(description)} chars")
        return description

    except Exception as e:
        raise RetryableException(f"Image analysis failed: {e}")


@retry_with_backoff(max_retries=2, base_wait=3.0, retryable_exceptions=(RetryableException,))
def rewrite_prompt(original_prompt: str, image_description: str) -> str:
    """
    Rewrite image_prompt to only reference objects
    confirmed to exist in the anchor image.
    """
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": PROMPT_REWRITE_TEMPLATE.format(
                    original_prompt=original_prompt,
                    image_description=image_description
                )
            }]
        )

        return message.content[0].text.strip()

    except Exception as e:
        raise RetryableException(f"Prompt rewrite failed: {e}")


def run_agent_5a5(context: dict) -> dict:
    """
    Agent 5a.5: Visual Prompt Corrector.

    Analyzes Episode 1 anchor images and rewrites scene image_prompts
    to only reference objects confirmed to exist in those images.

    Only runs on Episode 1 as a quality gate.
    Episodes 2+ use original Agent 3 prompts.
    """
    logger.info("Starting visual prompt correction for Episode 1...")

    try:
        anchor_images = context.get("story_bible", {}).get("anchor_images", {})

        if not anchor_images:
            logger.warning("No anchor images found — skipping prompt correction")
            return context

        scene_prompts = context.get("scene_prompts", [])

        if not scene_prompts:
            logger.warning("No scene prompts found")
            return context

        episode_1_prompts = scene_prompts[0]

        if not episode_1_prompts:
            logger.warning("Episode 1 has no scene prompts")
            return context

        logger.info(f"Analyzing {len(anchor_images)} anchor images...")

        # Analyze each anchor image once and cache the description
        image_descriptions = {}
        for location_name, anchor_data in anchor_images.items():
            image_path = anchor_data.get("image_path", "")
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Anchor image missing for: {location_name}")
                continue

            logger.info(f"  Analyzing: {location_name}")
            description = analyze_anchor_image(image_path)
            image_descriptions[location_name] = description

        logger.info(f"Rewriting {len(episode_1_prompts)} Episode 1 prompts...")

        import re as _re
        _STOP = {"the", "of", "a", "an", "and", "in", "at", "to", "for", "with", "from", "by", "s"}

        def _loc_slug(name: str) -> str:
            words = [w for w in _re.sub(r"[^a-z0-9 ]+", " ", name.lower()).split() if w not in _STOP]
            return "_".join(words)

        slug_descriptions = {_loc_slug(k): v for k, v in image_descriptions.items()}

        # Only "environment" generation_strategy scenes go through correction.
        # Forensic, object_closeup, person_trace, document scenes are bypassed —
        # their prompts are precisely written and anchor-grounding would corrupt them.
        _BYPASS_STRATEGIES = {"forensic", "object_closeup", "person_trace", "document"}
        _SKIP_TYPES = {"evidence", "object", "person_trace"}  # legacy fallback

        corrected_prompts = []
        for scene in episode_1_prompts:
            location = scene.get("location", "unknown")
            original_prompt = scene.get("image_prompt", "")  # ← image_prompt not video_prompt

            # Skip correction for evidence/object/forensic scenes — anchor reference
            # would force them to match an environmental image (e.g. staircase).
            # Bypass by generation_strategy (primary) or legacy image_type (fallback)
            if (scene.get("generation_strategy", "environment") in _BYPASS_STRATEGIES
                    or scene.get("is_impact")
                    or scene.get("image_type") in _SKIP_TYPES
                    or scene.get("photo_evidence_treatment")):
                corrected_prompts.append(scene)
                scene["prompt_corrected"] = False
                continue

            # Exact slug match — avoids stop-word collisions like "the"
            scene_slug = _loc_slug(location)
            image_desc = slug_descriptions.get(scene_slug)

            if image_desc and original_prompt:
                try:
                    corrected = rewrite_prompt(original_prompt, image_desc)
                    scene["flux_prompt"] = corrected    # ← image_prompt not video_prompt
                    scene["prompt_corrected"] = True
                    logger.info(f"  Scene {scene['scene_number']} corrected")
                except Exception as e:
                    logger.warning(
                        f"  Scene {scene['scene_number']} correction failed: {e}"
                    )
                    scene["prompt_corrected"] = False
            else:
                logger.warning(
                    f"  Scene {scene['scene_number']} — "
                    f"no matching image found for: {location}"
                )
                scene["prompt_corrected"] = False

            corrected_prompts.append(scene)

        context["scene_prompts"][0] = corrected_prompts
        context["anchor_image_descriptions"] = image_descriptions

        corrected_count = sum(1 for s in corrected_prompts if s.get("prompt_corrected"))
        logger.info(
            f"Prompt correction complete: "
            f"{corrected_count}/{len(corrected_prompts)} corrected"
        )

        return context

    except Exception as e:
        logger.error(f"Prompt corrector failed: {e}")
        context["warnings"].append({
            "agent": "prompt_corrector",
            "warning": str(e)
        })
        return context