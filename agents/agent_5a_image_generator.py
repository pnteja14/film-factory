import os
import json
import hashlib
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from utils.logger import AgentLogger
from utils.errors import ImageGeneratorException, RetryableException, PermanentException
from utils.retry import retry_with_backoff

load_dotenv()

logger = AgentLogger("ImageGenerator")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ANCHOR_DIR = Path("outputs/anchors")
REVEAL_DIR = Path("outputs/reveal_images")
SCENE_DIR = Path("outputs/scene_images")

for d in [ANCHOR_DIR, REVEAL_DIR, SCENE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# FLUX model for HF Inference API.
# FLUX.1-dev: higher quality, better prompt adherence, required for cross-episode consistency.
# Requires HF PRO subscription for Inference API access.
# Set HF_FLUX_MODEL=black-forest-labs/FLUX.1-schnell in .env for free-tier fallback.
MODEL_ID      = os.getenv("HF_FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
_FLUX_IS_DEV  = "dev" in MODEL_ID.lower()
_FLUX_STEPS   = 28 if _FLUX_IS_DEV else 4
_FLUX_GUIDANCE = 3.5 if _FLUX_IS_DEV else 0.0

# IMAGE_SOURCE controls where image generation runs.
# "hf"     — HuggingFace Inference API with FLUX.1-dev (default, PRO required)
# "kaggle" — Kaggle notebook serving FLUX.1-dev via HTTP (higher quality, free 30h/week)
IMAGE_SOURCE    = os.getenv("IMAGE_SOURCE", "hf")
KAGGLE_IMAGE_URL = os.getenv("KAGGLE_IMAGE_URL", "").rstrip("/")

INSERT_DIR = Path("outputs/insert_images")
INSERT_DIR.mkdir(parents=True, exist_ok=True)

# Set True (or env REUSE_EXISTING_IMAGES=true) to force reuse of ALL existing
# images regardless of story change. Default False — pipeline auto-detects
# story changes and clears old images so a new story never inherits old visuals.
REUSE_EXISTING_IMAGES = os.getenv("REUSE_EXISTING_IMAGES", "false").lower() == "true"

STORY_HASH_FILE = Path("outputs/story_hash.txt")


def _story_hash(context: dict) -> str:
    """Stable 8-char hash of the current story identity."""
    key = (context.get("story_title", "") + context.get("story_summary", ""))[:400]
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _story_changed(context: dict) -> bool:
    """Return True if the story is different from the last image-generation run."""
    current = _story_hash(context)
    if STORY_HASH_FILE.exists():
        try:
            return STORY_HASH_FILE.read_text().strip() != current
        except Exception:
            return True
    return True  # no hash on disk → treat as new story


def _save_story_hash(context: dict) -> None:
    STORY_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    STORY_HASH_FILE.write_text(_story_hash(context))

# Validation: how well must the generated image match the visual_brief to pass?
# Score is 0-1 from Claude Haiku vision. Set VALIDATION_THRESHOLD=0 to skip.
VALIDATION_THRESHOLD = 0.55
VALIDATION_MAX_RETRIES = 2  # up to 3 total attempts per scene image

# ── IMAGE DIMENSIONS — 16:9 native ───────────────────────────────────────────
# FLUX.1-dev on A100 — multiples of 64; 1280×720 unlocks higher detail
SCENE_WIDTH = 1280
SCENE_HEIGHT = 720

# Reveal/evidence images — 16:9 landscape so compositor displays without cropping
REVEAL_WIDTH  = 1280
REVEAL_HEIGHT = 720

# ── UNIVERSAL DARK TONE STYLE ─────────────────────────────────────────────────
# Applied to every single image regardless of story.
# Designed to feel like actual crime scene / investigation footage.
UNIVERSAL_DARK_SUFFIX = (
    "extremely dark and shadowy, dim single light source, "
    "deep crushed blacks, desaturated near monochrome, "
    "cold blue-grey tones, heavy oppressive shadow, "
    "cinematic grain 35mm film, dark atmospheric mystery, "
    "dramatic cinematic lighting, ominous still atmosphere, "
    "photorealistic, no people, no faces, no hands, no figures, "
    "no silhouettes, environmental storytelling only, 4K"
)

# Scene images — wide atmospheric shots of locations
SCENE_STYLE = (
    "wide establishing shot, extreme low key lighting, "
    "single motivated light source only, "
    "vast dark negative space, "
    "architectural detail emerging from shadow, "
    "still and suffocating atmosphere, "
    "no movement implied, "
    + UNIVERSAL_DARK_SUFFIX
)

# Evidence/reveal images — forensic close-ups
EVIDENCE_STYLE = (
    "extreme macro close-up, harsh single overhead light, "
    "forensic evidence photography, isolated object on dark surface, "
    "high contrast object edges, deep shadow surround, "
    "clinical cold documentation style, "
    + UNIVERSAL_DARK_SUFFIX
)

# Subject images — backlit silhouettes only
SUBJECT_STYLE = (
    "lone distant silhouette, heavy rim backlight only, "
    "face and features completely lost in shadow, "
    "unidentifiable anonymous figure, "
    "ominous presence implied not shown, "
    + UNIVERSAL_DARK_SUFFIX
)

NEGATIVE_PROMPT_SCENE = (
    "bright, colorful, daylight, sunshine, people, face, hands, fingers, "
    "human, figure, silhouette, shadow figure, hooded figure, person, body, "
    "crowd, text, watermark, cartoon, illustration, "
    "painting, warm colors, cheerful, happy, clean"
)

NEGATIVE_PROMPT_EVIDENCE = (
    "person, human, face, people, "
    "hands, fingers, palm, arm, wrist, skin, human hand, holding object, "
    "person holding, fingers touching, hand near object, "
    "bright background, colorful, text, watermark, cartoon"
)

NEGATIVE_PROMPT_SUBJECT = (
    "face visible, recognizable features, hands visible, "
    "bright scene, text, watermark, cartoon, color"
)


def validate_image_against_brief(
    image_path: str,
    visual_brief: str,
    required_elements: list = None,
) -> float:
    """
    Score (0.0–1.0) how well the generated image matches visual_brief.
    When required_elements is provided, each missing element deducts from the score.
    Returns -1.0 when validation is unavailable.
    """
    if not visual_brief or not Path(image_path).exists():
        return -1.0
    try:
        import anthropic
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        elements_text = ""
        if required_elements:
            elements_text = (
                f"\nRequired elements that MUST be visible: {', '.join(required_elements)}\n"
                "Deduct heavily if any required element is missing or wrong."
            )

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": (
                    f"Rate how well this image matches: {visual_brief}{elements_text}\n"
                    "Score 0.0-1.0 where: 1.0=perfect match including all required elements, "
                    "0.6=correct location type and mood, "
                    "0.4=correct atmosphere but wrong period/details or missing required elements, "
                    "0.1=completely wrong scene. "
                    "Reply with ONLY a decimal number, nothing else."
                )},
            ]}]
        )
        return max(0.0, min(1.0, float(msg.content[0].text.strip())))
    except Exception as e:
        logger.debug(f"Image validation skipped: {e}")
        return -1.0


def validate_api_token() -> None:
    if not HF_API_TOKEN:
        raise PermanentException("HF_API_TOKEN not found in .env")


def build_scene_prompt(visual_description: str) -> str:
    """
    Build FLUX prompt for scene/anchor image.
    Strips the raw visual description and wraps in universal dark style.
    Story-agnostic — works for any true crime story automatically.
    """
    clean = visual_description.strip().rstrip(".,")
    # Remove any accidental brightness or color references
    bright_words = ["bright", "warm", "sunny", "light-filled", "cheerful", "colorful", "vibrant"]
    for word in bright_words:
        clean = clean.replace(word, "dim").replace(word.capitalize(), "dim")
    return f"{clean}, {SCENE_STYLE}"


def build_reveal_prompt(image_brief: str, image_type: str) -> str:
    """Build FLUX prompt for reveal/evidence image."""
    clean = image_brief.strip().rstrip(".,")
    if image_type == "subject":
        return f"{clean}, {SUBJECT_STYLE}"
    else:
        return f"{clean}, {EVIDENCE_STYLE}"


def get_safe_filename(name: str) -> str:
    safe = name.replace(" ", "_").lower()
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")
    return safe[:60]


def load_seed_map() -> dict:
    seed_map_path = ANCHOR_DIR / "seed_map.json"
    if seed_map_path.exists():
        try:
            with open(seed_map_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_seed_map(seed_map: dict) -> None:
    try:
        with open(ANCHOR_DIR / "seed_map.json", "w") as f:
            json.dump(seed_map, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save seed map: {e}")


def _force_dimensions(img, target_w: int, target_h: int):
    """
    Cover-resize + center-crop to exact target dimensions.
    Wavespeed (and some other providers) ignore width/height params and
    return 1024×1024 squares. This corrects the aspect ratio in one step
    so every saved image is always 16:9 — no further cropping in the
    compositor is needed.
    """
    from PIL import Image as _Image
    iw, ih = img.size
    if iw == target_w and ih == target_h:
        return img
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), _Image.LANCZOS)
    left = (nw - target_w) // 2
    top  = (nh - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# FLUX.1-schnell ignores negative prompts and CFG — only used as free-tier fallback.
# FLUX.1-dev supports negative prompts, full CFG, and produces far more consistent
# results across episodes when given the same seed and style lock.
_SCHNELL_POSITIVE_PREFIX = (
    "completely empty environment, zero human presence, inanimate objects only, "
    "abandoned and desolate, no living beings, "
)


@retry_with_backoff(
    max_retries=3,
    base_wait=5.0,
    max_wait=60.0,
    retryable_exceptions=(RetryableException,)
)
def generate_image_via_hf(
    prompt: str,
    negative_prompt: str,
    seed: int,
    client: InferenceClient,
    width: int = SCENE_WIDTH,
    height: int = SCENE_HEIGHT,
):
    """Generate one image via FLUX (dev or schnell). Returns PIL Image at exact (width, height)."""
    # Schnell ignores negative prompts — prepend positive-phrased safety language.
    # Dev supports negative prompts and CFG natively.
    actual_prompt = prompt if _FLUX_IS_DEV else (_SCHNELL_POSITIVE_PREFIX + prompt)
    actual_neg    = negative_prompt if _FLUX_IS_DEV else ""
    try:
        image = client.text_to_image(
            prompt=actual_prompt,
            model=MODEL_ID,
            seed=seed,
            width=width,
            height=height,
            num_inference_steps=_FLUX_STEPS,
            guidance_scale=_FLUX_GUIDANCE,
            negative_prompt=actual_neg,
        )
        # Guarantee 16:9 output regardless of what the provider returned
        return _force_dimensions(image, width, height)
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str:
            raise RetryableException(f"Rate limited: {e}")
        elif "503" in error_str or "model is loading" in error_str:
            raise RetryableException(f"Model loading: {e}")
        elif "401" in error_str or "unauthorized" in error_str:
            raise PermanentException(f"Auth failed: {e}")
        elif "402" in error_str or "payment" in error_str or "403" in error_str or "forbidden" in error_str:
            raise PermanentException(
                f"FLUX.1-dev requires HF PRO subscription for Inference API. "
                f"Set HF_FLUX_MODEL=black-forest-labs/FLUX.1-schnell in .env for free tier. Error: {e}"
            )
        else:
            raise RetryableException(f"API failed: {e}")


def generate_image_via_kaggle(
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int = SCENE_WIDTH,
    height: int = SCENE_HEIGHT,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
):
    """Generate one image via the Kaggle FLUX.1-dev notebook server. Returns PIL Image."""
    import requests as _req
    import base64 as _b64
    import io as _io
    from PIL import Image as _PILImage

    if not KAGGLE_IMAGE_URL:
        raise PermanentException(
            "IMAGE_SOURCE=kaggle but KAGGLE_IMAGE_URL not set in .env. "
            "Run the kaggle_flux_server.py notebook and paste the ngrok URL."
        )

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": (width  // 16) * 16,
        "height": (height // 16) * 16,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }

    try:
        resp = _req.post(f"{KAGGLE_IMAGE_URL}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RetryableException(f"Kaggle server error: {data['error']}")
        img_bytes = _b64.b64decode(data["image_base64"])
        img = _PILImage.open(_io.BytesIO(img_bytes)).convert("RGB")
        return _force_dimensions(img, width, height)
    except _req.exceptions.Timeout:
        raise RetryableException("Kaggle server timeout (>600s)")
    except _req.exceptions.ConnectionError as e:
        raise RetryableException(f"Kaggle server unreachable: {e}")
    except RetryableException:
        raise
    except Exception as e:
        raise RetryableException(f"Kaggle generation failed: {e}")


def generate_image(
    prompt: str,
    negative_prompt: str,
    seed: int,
    client,
    width: int = SCENE_WIDTH,
    height: int = SCENE_HEIGHT,
    num_inference_steps: int = None,
    guidance_scale: float = None,
):
    """
    Dispatch to Kaggle (FLUX.1-dev) or HuggingFace (FLUX.1-schnell)
    based on IMAGE_SOURCE env var.
    """
    if IMAGE_SOURCE == "kaggle":
        steps = num_inference_steps or 28
        scale = guidance_scale or 3.5
        return generate_image_via_kaggle(
            prompt, negative_prompt, seed, width, height, steps, scale
        )
    else:
        return generate_image_via_hf(
            prompt, negative_prompt, seed, client, width, height
        )


def delete_existing_images() -> None:
    """
    Delete all previously generated images so we regenerate fresh
    with correct 16:9 dimensions and updated dark prompts.
    """
    deleted = 0
    for directory in [ANCHOR_DIR, REVEAL_DIR, SCENE_DIR]:
        for img_file in directory.glob("*.png"):
            try:
                img_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete {img_file}: {e}")

    # Also delete seed map so seeds reset
    seed_map_path = ANCHOR_DIR / "seed_map.json"
    if seed_map_path.exists():
        seed_map_path.unlink()

    logger.info(f"Deleted {deleted} existing images — fresh generation")


def generate_anchor_image(
    location_name: str,
    visual_description: str,
    seed: int,
    client: InferenceClient
) -> str:
    """Generate one 16:9 anchor/scene image for a story location."""
    safe_name = get_safe_filename(location_name)
    image_path = ANCHOR_DIR / f"{safe_name}.png"

    if image_path.exists():
        logger.info(f"Anchor exists: '{location_name}' — skipping")
        return str(image_path)

    prompt = build_scene_prompt(visual_description)
    logger.info(f"Generating anchor: '{location_name}' ({SCENE_WIDTH}x{SCENE_HEIGHT})")
    logger.debug(f"Prompt: {prompt[:100]}...")

    try:
        image = generate_image(
            prompt, NEGATIVE_PROMPT_SCENE, seed, client,
            width=SCENE_WIDTH, height=SCENE_HEIGHT
        )
        if image is None:
            raise ImageGeneratorException("Image returned None")

        image.save(str(image_path))
        logger.info(f"Saved: {image_path}")
        return str(image_path)

    except PermanentException as e:
        raise ImageGeneratorException(f"Permanent: {e}")
    except RetryableException as e:
        raise ImageGeneratorException(f"Max retries: {e}")
    except Exception as e:
        raise ImageGeneratorException(f"Unexpected: {e}")


def generate_reveal_image(
    episode_number: int,
    reveal_index: int,
    reveal: dict,
    seed: int,
    client: InferenceClient,
    story_slug: str = "",
) -> str:
    """Generate one reveal/evidence close-up image."""
    image_type = reveal.get("image_type", "evidence")
    image_brief = reveal.get("image_brief", "")

    # Include story slug so reveals never reuse images from a different story
    slug_part = f"_{story_slug}" if story_slug else ""
    filename = f"ep{episode_number+1}_reveal{reveal_index+1}_{image_type}{slug_part}.png"
    image_path = REVEAL_DIR / filename

    if image_path.exists():
        logger.info(f"Reveal exists: {filename} — skipping")
        return str(image_path)

    if not image_brief:
        raise ImageGeneratorException(f"No image_brief for reveal {reveal_index+1}")

    prompt = build_reveal_prompt(image_brief, image_type)

    if image_type == "subject":
        negative = NEGATIVE_PROMPT_SUBJECT
    else:
        negative = NEGATIVE_PROMPT_EVIDENCE

    logger.info(f"Generating reveal [{image_type}] ep{episode_number+1} r{reveal_index+1}")

    try:
        image = generate_image(
            prompt, negative, seed, client,
            width=REVEAL_WIDTH, height=REVEAL_HEIGHT
        )
        if image is None:
            raise ImageGeneratorException("Image returned None")

        image.save(str(image_path))
        logger.info(f"Saved reveal: {image_path}")
        return str(image_path)

    except PermanentException as e:
        raise ImageGeneratorException(f"Permanent: {e}")
    except RetryableException as e:
        raise ImageGeneratorException(f"Max retries: {e}")
    except Exception as e:
        raise ImageGeneratorException(f"Unexpected: {e}")


def _build_location_style_suffix(location: str, location_visuals: dict) -> str:
    """
    Return a compact style-lock phrase for a location (≤15 words).
    Extracted from the story bible's visual description for that location.
    Returns empty string if no description exists.
    """
    desc = location_visuals.get(location, "")
    if not desc:
        # Fuzzy fallback: try case-insensitive partial match
        loc_lower = location.lower()
        for key, val in location_visuals.items():
            if key.lower() in loc_lower or loc_lower in key.lower():
                desc = val
                break
    if not desc:
        return ""
    words = desc.split()
    return " ".join(words[:30])


def generate_scene_images_for_episode(
    episode_number: int,
    scene_prompts: list,
    anchor_map: dict,
    seed_map: dict,
    client: InferenceClient,
    location_visuals: dict = None,
) -> list:
    """
    Generate scene background images for one episode.
    Anchor image is reused only for the FIRST scene at each location.
    location_visuals: story_bible dict — used to append a location style
    suffix to every prompt, ensuring visual consistency within a location.
    Scenes with a visual_brief are validated by Claude Haiku vision;
    if the score is below VALIDATION_THRESHOLD the image is regenerated
    (up to VALIDATION_MAX_RETRIES extra attempts; best image is always kept).
    """
    logger.info(f"Episode {episode_number+1} scene images...")
    anchor_used: set = set()
    location_visuals = location_visuals or {}

    for i, scene in enumerate(scene_prompts):
        location = scene.get("location_name", "") or scene.get("location", "unknown")

        # Reuse anchor for the very first scene at this location
        if location in anchor_map and location not in anchor_used:
            anchor_path = anchor_map[location].get("image_path", "")
            if anchor_path and Path(anchor_path).exists():
                scene["image_path"] = anchor_path
                anchor_used.add(location)
                logger.info(f"  Scene {i+1}: anchor '{location}' (first use)")
                continue

        filename = f"ep{episode_number+1}_scene{i+1}_{get_safe_filename(location)}.png"
        image_path = SCENE_DIR / filename

        if image_path.exists():
            scene["image_path"] = str(image_path)
            logger.info(f"  Scene {i+1}: existing reused")
            continue

        seed = seed_map.get(location, 1000 + i * 111)
        image_prompt = scene.get("flux_prompt", "") or scene.get("image_prompt", "")
        visual_brief = scene.get("visual_brief", "")
        if not image_prompt:
            logger.warning(f"  Scene {i+1}: no image_prompt")
            continue

        # Append location style-lock suffix — keeps all scenes at the same
        # location visually consistent across every episode.
        # Bypass for non-environment strategies (forensic, object_closeup, etc.)
        _bypass_strats = {"forensic", "object_closeup", "person_trace", "document"}
        if scene.get("generation_strategy", "environment") not in _bypass_strats:
            style_suffix = _build_location_style_suffix(location, location_visuals)
            if style_suffix and style_suffix.lower() not in image_prompt.lower():
                image_prompt = f"{image_prompt}, {style_suffix}"

        best_image = None
        best_score = -2.0  # lower than any real score

        generation_strategy = scene.get("generation_strategy", "environment")
        required_elements   = scene.get("required_elements", [])

        # Progressive prompt strengthening across retries
        def _strengthen_prompt(base_prompt: str, attempt: int, req_els: list) -> str:
            if attempt == 0 or not req_els:
                return base_prompt
            must_show = ", ".join(req_els)
            if attempt == 1:
                return f"MUST SHOW: {must_show}. {base_prompt}"
            # attempt 2+: switch to forensic isolation style
            return f"{must_show}, isolated on dark surface, single overhead light, extreme close-up, {EVIDENCE_STYLE}"

        for attempt in range(VALIDATION_MAX_RETRIES + 1):
            try:
                attempt_seed   = seed + (attempt * 997)
                attempt_prompt = _strengthen_prompt(image_prompt, attempt, required_elements)
                image = generate_image(
                    attempt_prompt, NEGATIVE_PROMPT_SCENE, attempt_seed, client,
                    width=SCENE_WIDTH, height=SCENE_HEIGHT
                )
                if not image:
                    continue

                # Validate only when brief is present and threshold > 0
                if visual_brief and VALIDATION_THRESHOLD > 0:
                    temp_path = SCENE_DIR / f"_tmp_{episode_number+1}_{i+1}_{attempt}.png"
                    image.save(str(temp_path))
                    score = validate_image_against_brief(str(temp_path), visual_brief, required_elements)
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

                    logger.info(f"  Scene {i+1} attempt {attempt+1}: validation score={score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_image = image

                    if score >= VALIDATION_THRESHOLD:
                        best_image.save(str(image_path))
                        scene["image_path"] = str(image_path)
                        logger.info(
                            f"  Scene {i+1}: generated '{location}' (score {score:.2f})"
                        )
                        break
                    elif attempt < VALIDATION_MAX_RETRIES:
                        logger.info(
                            f"  Scene {i+1}: score {score:.2f} < {VALIDATION_THRESHOLD} — retrying"
                        )
                        time.sleep(1)
                    else:
                        # All attempts done — keep best
                        if best_image:
                            best_image.save(str(image_path))
                            scene["image_path"] = str(image_path)
                            logger.info(
                                f"  Scene {i+1}: kept best attempt "
                                f"(score {best_score:.2f}) for '{location}'"
                            )
                else:
                    # No brief or threshold disabled — save immediately
                    image.save(str(image_path))
                    scene["image_path"] = str(image_path)
                    logger.info(f"  Scene {i+1}: generated '{location}'")
                    break

            except Exception as e:
                logger.error(f"  Scene {i+1} attempt {attempt+1} failed: {e}")
                if attempt == VALIDATION_MAX_RETRIES:
                    break

        # Final fallback: if image_path still unset but best_image exists
        if not scene.get("image_path") and best_image:
            best_image.save(str(image_path))
            scene["image_path"] = str(image_path)
            logger.info(f"  Scene {i+1}: saved best attempt for '{location}'")

        time.sleep(1)

    return scene_prompts


def run_agent_5a(context: dict) -> dict:
    """
    Agent 5a: Generate all images for the pipeline.

    Phase 0 — Delete all existing images (fresh 16:9 generation)
    Phase 1 — Anchor images (one per story location, 1280x720)
    Phase 2 — Scene images per episode (reuse anchors where possible)
    Phase 3 — Reveal images per episode (evidence close-ups, 1024x1024)
    """
    logger.info("Starting image generation (Agent 5a)...")

    try:
        validate_api_token()
        client = InferenceClient(api_key=HF_API_TOKEN)
        logger.info("HuggingFace client initialized")

        # ── PHASE 0: DELETE OLD IMAGES ─────────────────────────────────
        story_slug = _story_hash(context)
        if REUSE_EXISTING_IMAGES:
            logger.info("REUSE_EXISTING_IMAGES=True — skipping deletion, reusing cached images")
        elif _story_changed(context):
            logger.info("Story changed — clearing old images for fresh generation")
            delete_existing_images()
            _save_story_hash(context)
        else:
            logger.info("Same story detected — reusing existing images (set REUSE_EXISTING_IMAGES=True to force)")

        location_visuals = context.get("story_bible", {}).get("location_visuals", {})
        seed_map = load_seed_map()

        # Assign seeds to locations
        base_seed = 1000
        for i, loc in enumerate(location_visuals.keys()):
            if loc not in seed_map:
                seed_map[loc] = base_seed + (i * 137)
        save_seed_map(seed_map)

        # ── PHASE 1: ANCHOR IMAGES ─────────────────────────────────────
        logger.info(f"Phase 1: {len(location_visuals)} anchor images at {SCENE_WIDTH}x{SCENE_HEIGHT}...")
        anchor_map = {}

        for location_name, visual_description in location_visuals.items():
            seed = seed_map[location_name]
            try:
                image_path = generate_anchor_image(
                    location_name, visual_description, seed, client
                )
                anchor_map[location_name] = {
                    "image_path": image_path,
                    "seed": seed,
                    "visual_description": visual_description
                }
            except ImageGeneratorException as e:
                logger.error(f"Anchor failed '{location_name}': {e}")
                context["errors"].append({
                    "agent": "image_generator",
                    "phase": "anchor",
                    "location": location_name,
                    "error": str(e)
                })
            time.sleep(2)

        context["story_bible"]["anchor_images"] = anchor_map
        logger.info(f"Phase 1 complete: {len(anchor_map)}/{len(location_visuals)}")

        # ── PHASE 1.5: PROMPT CORRECTION (anchor-grounded) ────────────
        # Analyze generated anchor images and rewrite scene prompts so they
        # only reference objects confirmed visible in those anchors.
        # Running BEFORE Phase 2 means corrected prompts drive image generation.
        # allow_color scenes are skipped — their color-permissive prompts must
        # not be overwritten with a dark anchor-grounded rewrite.
        logger.info("Phase 1.5: Correcting scene prompts against anchor images...")
        try:
            from agents.agent_5a5_prompt_corrector import analyze_anchor_image, rewrite_prompt

            image_descriptions: dict = {}
            for location_name, anchor_data in anchor_map.items():
                anchor_path = anchor_data.get("image_path", "")
                if anchor_path and Path(anchor_path).exists():
                    try:
                        desc = analyze_anchor_image(anchor_path)
                        image_descriptions[location_name] = desc
                        logger.info(f"  Anchor analyzed: '{location_name}'")
                    except Exception as e:
                        logger.warning(f"  Anchor analysis skipped '{location_name}': {e}")

            import re as _re5a
            _STOP5a = {"the", "of", "a", "an", "and", "in", "at", "to", "for", "with", "from", "by", "s"}
            # Only "environment" strategy scenes go through the corrector.
            # All others (forensic, object_closeup, person_trace, document) are bypassed
            # because their prompts are precisely written for that treatment and
            # anchor-grounded rewriting would corrupt them.
            _CORRECTOR_BYPASS_STRATEGIES = {"forensic", "object_closeup", "person_trace", "document"}

            def _loc_slug5a(name: str) -> str:
                words = [w for w in _re5a.sub(r"[^a-z0-9 ]+", " ", name.lower()).split() if w not in _STOP5a]
                return "_".join(words)

            slug_descs5a = {_loc_slug5a(k): v for k, v in image_descriptions.items()}

            corrected_total = 0
            for ep_idx, episode_scenes in enumerate(context.get("scene_prompts", [])):
                for scene in episode_scenes:
                    # Bypass corrector for non-environment generation strategies
                    if (scene.get("allow_color")
                            or scene.get("generation_strategy", "environment") in _CORRECTOR_BYPASS_STRATEGIES
                            or scene.get("is_impact")):
                        scene["prompt_corrected"] = False
                        continue
                    location = scene.get("location", "")
                    original_prompt = scene.get("image_prompt", "")
                    if not original_prompt:
                        continue

                    # Use slug-based matching to avoid stop-word collisions like "the"
                    image_desc = slug_descs5a.get(_loc_slug5a(location))

                    if image_desc:
                        try:
                            corrected = rewrite_prompt(original_prompt, image_desc)
                            scene["flux_prompt"] = corrected
                            scene["prompt_corrected"] = True
                            corrected_total += 1
                        except Exception as e:
                            scene["prompt_corrected"] = False
                    else:
                        scene["prompt_corrected"] = False

            context["anchor_image_descriptions"] = image_descriptions
            logger.info(f"Phase 1.5 complete: {corrected_total} prompts corrected")

        except Exception as e:
            logger.warning(f"Phase 1.5 (prompt correction) failed — continuing with original prompts: {e}")

        # ── PHASE 2: SCENE IMAGES ──────────────────────────────────────
        scene_prompts_all = context.get("scene_prompts", [])
        logger.info(f"Phase 2: Scene images for {len(scene_prompts_all)} episodes...")

        for ep_idx, episode_scenes in enumerate(scene_prompts_all):
            if not episode_scenes:
                continue
            updated = generate_scene_images_for_episode(
                ep_idx, episode_scenes, anchor_map, seed_map, client,
                location_visuals=location_visuals,
            )
            context["scene_prompts"][ep_idx] = updated
            time.sleep(1)

        # ── PHASE 3: REVEAL IMAGES ─────────────────────────────────────
        reveal_moments_all = context.get("reveal_moments", [])
        logger.info(f"Phase 3: Reveal images for {len(reveal_moments_all)} episodes...")

        for ep_idx, episode_reveals in enumerate(reveal_moments_all):
            if not episode_reveals:
                continue

            logger.info(f"  Episode {ep_idx+1}: {len(episode_reveals)} reveals...")

            for r_idx, reveal in enumerate(episode_reveals):
                reveal_seed = 5000 + (ep_idx * 100) + (r_idx * 17)
                try:
                    image_path = generate_reveal_image(
                        ep_idx, r_idx, reveal, reveal_seed, client,
                        story_slug=story_slug,
                    )
                    context["reveal_moments"][ep_idx][r_idx]["image_path"] = image_path
                    logger.info(f"    Reveal {r_idx+1}: {reveal.get('image_type')} saved")
                except ImageGeneratorException as e:
                    logger.error(f"    Reveal {r_idx+1} failed: {e}")
                    context["warnings"].append({
                        "agent": "image_generator",
                        "episode": ep_idx+1,
                        "reveal": r_idx+1,
                        "warning": str(e)
                    })
                time.sleep(2)

        # ── PHASE 3.5: INSERT SHOT IMAGES ─────────────────────────────
        # Generate close-up images for word-level insert shots added by scene_director.
        logger.info("Phase 3.5: Insert shot images...")
        total_inserts = 0
        gen_inserts   = 0

        for ep_idx, episode_scenes in enumerate(context.get("scene_prompts", [])):
            for s_idx, scene in enumerate(episode_scenes):
                for ins_idx, ins in enumerate(scene.get("insert_shots", [])):
                    total_inserts += 1
                    slug = get_safe_filename(ins.get("trigger_word", f"insert{ins_idx}"))
                    filename = f"ep{ep_idx+1}_s{s_idx+1}_ins{ins_idx+1}_{slug}.png"
                    ins_path = INSERT_DIR / filename

                    if ins_path.exists():
                        ins["image_path"] = str(ins_path)
                        gen_inserts += 1
                        continue

                    ins_brief    = ins.get("visual_brief", "")
                    ins_strategy = ins.get("generation_strategy", "object_closeup")
                    if not ins_brief:
                        continue

                    if ins_strategy == "forensic":
                        ins_prompt   = f"{ins_brief}, {EVIDENCE_STYLE}"
                        ins_negative = NEGATIVE_PROMPT_EVIDENCE
                    else:
                        ins_prompt   = f"{ins_brief}, extreme close-up, isolated, {EVIDENCE_STYLE}"
                        ins_negative = NEGATIVE_PROMPT_EVIDENCE

                    ins_seed = 9000 + ep_idx * 1000 + s_idx * 100 + ins_idx * 17
                    try:
                        ins_image = generate_image(
                            ins_prompt, ins_negative, ins_seed, client,
                            width=REVEAL_WIDTH, height=REVEAL_HEIGHT,
                        )
                        if ins_image:
                            ins_image.save(str(ins_path))
                            ins["image_path"] = str(ins_path)
                            gen_inserts += 1
                            logger.info(
                                f"  Insert ep{ep_idx+1} s{s_idx+1} '{ins.get('trigger_word')}': saved"
                            )
                    except Exception as e:
                        logger.warning(
                            f"  Insert ep{ep_idx+1} s{s_idx+1} '{ins.get('trigger_word')}' failed: {e}"
                        )
                    time.sleep(1)

        logger.info(f"Phase 3.5 complete: {gen_inserts}/{total_inserts} insert shots generated")

        # ── SUMMARY ────────────────────────────────────────────────────
        total_anchors = len(anchor_map)
        total_scenes = sum(
            sum(1 for s in ep if s.get("image_path"))
            for ep in context.get("scene_prompts", [])
        )
        total_reveals = sum(
            sum(1 for r in ep if r.get("image_path"))
            for ep in context.get("reveal_moments", [])
        )

        logger.info(
            f"Image generation complete — "
            f"{total_anchors} anchors, "
            f"{total_scenes} scenes, "
            f"{total_reveals} reveals | "
            f"All at {SCENE_WIDTH}x{SCENE_HEIGHT} (16:9)"
        )

        if total_anchors == 0 and len(location_visuals) > 0:
            raise ImageGeneratorException("All anchor generations failed")

        return context

    except ImageGeneratorException as e:
        logger.error(f"Agent 5a failed: {e}")
        context["errors"].append({"agent": "image_generator", "error": str(e)})
        raise
    except Exception as e:
        logger.error(f"Unexpected: {e}")
        context["errors"].append({"agent": "image_generator", "error": f"Unexpected: {e}"})
        raise ImageGeneratorException(f"Unexpected: {e}")