import re
import json
import anthropic
import os
from dotenv import load_dotenv
from utils.logger import AgentLogger
from utils.errors import SceneDirectorException, RetryableException
from utils.retry import retry_with_backoff

load_dotenv()

logger = AgentLogger("SceneDirector")

# ── STYLE SUFFIXES ─────────────────────────────────────────────────────────────
# Hard-enforced on every scene image prompt. NO people. NO silhouettes. EVER.

STYLE_SUFFIX = (
    "first person POV, empty location, no people present, "
    "no humans, no figures, no faces, no hands, no bodies, no silhouettes, "
    "dark atmospheric, dim natural light, deep shadows, "
    "desaturated muted tones, cinematic grain, "
    "film noir, true crime documentary aesthetic, "
    "photorealistic, 4K"
)

EVIDENCE_STYLE_SUFFIX = (
    "extreme close-up, forensic evidence photography, isolated object, "
    "dark surface, single harsh overhead light, high contrast, "
    "crime scene documentation style, photorealistic, 4K, "
    "no hands, no people, no faces"
)

DOCUMENT_STYLE_SUFFIX = (
    "extreme close-up of paper document on dark wooden desk, "
    "single overhead lamp, printed or handwritten text visible, "
    "official legal document aesthetic, aged paper, dark vignette edges, "
    "no hands, no people, no faces, photorealistic, 4K"
)

SUBJECT_STYLE_SUFFIX = (
    "distant silhouette, heavy backlight, figure completely unidentifiable, "
    "face entirely obscured in shadow, no visible features whatsoever, "
    "dark atmospheric background, cinematic grain, 4K"
)

COLOR_STYLE_SUFFIX = (
    "first person POV, empty location, no people present, "
    "no humans, no figures, no faces, no hands, no bodies, no silhouettes, "
    "natural ambient lighting, muted but perceptible color tones, "
    "cinematic grain, true crime documentary aesthetic, "
    "photorealistic, 4K"
)

# ── CINEMATOGRAPHY VOCABULARY ─────────────────────────────────────────────────
# These terms are appended to FLUX prompts for environment-type scenes.
# Evidence/person_trace/allow_color scenes use their own controlled suffixes.

FRAMING_TO_FLUX = {
    "wide_establishing": "wide establishing shot, full environment visible, expansive negative space",
    "medium_wide":       "medium wide angle, room interior fully visible, environmental context",
    "medium":            "medium shot, single object or surface at center frame",
    "medium_close":      "medium close-up, intimate object detail, subject fills center third",
    "extreme_close":     "extreme close-up, forensic macro, one surface detail fills entire frame",
    "low_angle":         "low angle shot from floor level, ceiling visible above, looking up",
    "high_angle":        "high angle overhead looking straight down, floor pattern prominent",
    "dutch_tilt":        "dutch tilt 15 degrees, canted horizon, psychological unease",
}

LENS_TO_FLUX = {
    "24mm":  "24mm wide-angle lens, slight natural barrel distortion at frame edges",
    "35mm":  "35mm prime lens, natural perspective, comfortable field of view",
    "50mm":  "50mm standard lens, human-eye perspective, neutral compression",
    "85mm":  "85mm portrait lens, shallow depth of field, background compression",
    "macro": "macro lens, extreme depth of field, forensic 1:1 reproduction scale",
}

# ── ARC VISUAL VARIATION ──────────────────────────────────────────────────────
# Appended to the location visual description so each scene at the same
# location shows a distinct angle/detail rather than the same composition.
ARC_SCENE_VARIATION = {
    "establish":     (
        "wide establishing shot, exterior approach angle, "
        "full location geometry visible, distance and scale emphasized"
    ),
    "build":         (
        "medium interior shot, receding perspective down a corridor or hall, "
        "single light source creating depth, architectural texture in focus"
    ),
    "investigation": (
        "close on a single specific detail — door handle, numbered placard, "
        "frosted glass panel, institutional signage, evidence of access or use"
    ),
    "revelation":    (
        "low dramatic angle, looking up at ceiling or high wall, "
        "light source directly above, disorienting compressed perspective"
    ),
    "cliffhanger":   (
        "extreme close-up on one ominous element — flickering light, "
        "cracked wall, partial shadow, deep black surround, isolating frame"
    ),
}

# ── ARC POSITIONS ──────────────────────────────────────────────────────────────
ARC_POSITIONS = {
    "establish":     {"duration": "short",  "zoom": "out"},
    "build":         {"duration": "medium", "zoom": "in"},
    "investigation": {"duration": "medium", "zoom": "in"},
    "revelation":    {"duration": "long",   "zoom": "in"},
    "cliffhanger":   {"duration": "long",   "zoom": "none"},
}

DURATION_SECONDS = {"short": 5, "medium": 8, "long": 11}
DURATION_FRAMES  = {"short": 81, "medium": 129, "long": 161}

# ── REVEAL MOMENT PROMPT ───────────────────────────────────────────────────────
DECOMPOSE_PROMPT = """
You are a cinematographer for a true crime narration series on Instagram.
Your job: for each spoken narration chunk, decide exactly what the camera shows.

STORY:
{story_context}

AVAILABLE LOCATIONS (use these exact names):
{locations_list}

NARRATION CHUNKS (each is one sentence or short paragraph):
{chunks_text}

For EACH chunk assign ONE visual. Ask: what does the camera see as the narrator speaks these words?

visual_type:
  environment  — wide shot of a place; use when narration describes a setting, time, or atmosphere
  object       — close-up of a specific physical item; use when narration names an object
  evidence     — extreme forensic close-up; use when narration mentions clues, records, dates, inscribed text
  person_trace — indirect signs of presence only; use when narration mentions a person doing something

image_style:
  atmospheric — cinematic, muted tones, wide
  forensic    — harsh single overhead light, isolated object on dark surface
  trace       — space that feels recently occupied, ambient emptiness

CORE RULES:
- visual_brief: 15-20 words, concrete and specific, describe ONLY what IS visible in the frame
  NO camera movement words, NO people, NO faces, NO hands
- location: must match one of the available locations EXACTLY
- vary locations — do NOT repeat the same location more than 2 times in a row
- last chunk: use the most dramatic available location

VISUAL CONTINUITY — read chunks in sequence before assigning:
- Each visual should feel like a natural follow-on from the previous
- If chunk N shows a location exterior, chunk N+1 can show an interior detail at the same location
- If chunk N describes finding something, chunk N+1 can show that thing in detail

SECRET LOCATION RULE — critical for discovery moments:
- If a location is hidden, secret, or accessed through concealment (a room behind a panel, a locked space, a secret compartment), do NOT assign it until the narration explicitly describes the narrator ENTERING or BEING INSIDE it
- When narration describes the ACT of discovering (moving furniture, pressing a wall, hearing hollowness, feeling a give), assign the CURRENT VISIBLE location (where the narrator was before the discovery) — show what the narrator sees BEFORE the revelation
- The hidden location appears ONLY when the narration places the narrator physically inside it

CLIFFHANGER EVIDENCE RULE — for final tension moments:
- When the last 1-2 chunks describe a specific inscribed object (a date written on something, a name, a number, text visible on a photograph or document), ALWAYS use visual_type "evidence"
- visual_brief must describe that specific object in extreme close-up (e.g., "handwritten date in faint ink on torn photograph corner, dark surface, extreme close-up")
- Do NOT use environment or person_trace for these cliffhanger inscription moments

PERSON ORIENTATION RULE:
- When narration describes a person's exact orientation or posture (back to camera, facing away, shoulders turned, reaching), use visual_type "person_trace" and describe only indirect evidence of that orientation (e.g., "man's silhouette at kitchen counter, back completely to light source, shoulders turned slightly right")
- Never show a face or identifiable features

ALLOW COLOR RULE:
- allow_color: true ONLY when the narration EXPLICITLY states something "is in color" or when a specific vivid color is the entire story point of that moment
- allow_color: false for ALL other scenes

CINEMATOGRAPHY — assign one framing and one lens per scene:

framing options:
  wide_establishing  — exterior or full room visible, maximum negative space
  medium_wide        — room or corridor, environmental context preserved
  medium             — desk or doorway at center, contained intimate space
  medium_close       — single object focus, subject fills center of frame
  extreme_close      — one detail fills entire frame, forensic precision
  low_angle          — camera near floor, looking up, ceiling dominates top
  high_angle         — looking straight down, floor pattern fills frame
  dutch_tilt         — canted 15-degree horizon, psychological unease

lens options:
  24mm   — wide, immersive space, slight edge distortion
  35mm   — natural, comfortable, slightly wider than human eye
  50mm   — neutral, human-eye equivalent, standard perspective
  85mm   — compressed depth, background blur, intimate isolation
  macro  — extreme close only, forensic 1:1 detail

CINEMATOGRAPHY RULES:
- establish arc: wide_establishing + 24mm
- build arc: medium_wide or medium + 35mm
- investigation arc: medium_close + 85mm
- revelation arc: extreme_close + 85mm OR dutch_tilt + 50mm
- cliffhanger arc: dutch_tilt or low_angle + 50mm
- evidence visual_type: ALWAYS extreme_close + macro
- person_trace visual_type: medium_wide + 35mm

PERSIST VISUAL RULE:
- persist_visual: true when THIS chunk continues describing the SAME subject/object/scene as the PREVIOUS chunk
  (e.g. chunk N says "I found a body", chunk N+1 says "It was slumped against the desk" → chunk N+1 persist_visual=true)
- persist_visual: false when the narration subject changes or this is the first chunk
- When persist_visual=true the primary scene image does NOT change — the previous image holds on screen

IMPACT RULE:
- is_impact: true for high-drama discovery moments — first sight of a body, shocking evidence, major revelation
- is_impact: false for ordinary scene descriptions, transitions, atmosphere
- Impact images hold on screen until persist_visual chain ends

REQUIRED ELEMENTS — list 1-3 specific objects/elements that MUST visually appear in the generated image:
- For environment scenes: the key architectural features that define this location
- For forensic/evidence scenes: the exact object being described (e.g. ["crumpled letter", "dark wooden floor"])
- For is_impact scenes: the shocking element (e.g. ["body", "slumped posture", "dark floor"])
- Keep each element under 5 words

INSERT SHOTS — for each specific named object/evidence mentioned in the narration:
- trigger_word: the exact word(s) in the narration that cues the cut (e.g. "letter", "photograph", "knife")
- visual_brief: extreme close-up description of that object (under 20 words)
- generation_strategy: "object_closeup" or "forensic"
- hold_seconds: 1.5-3.0 (shorter for casual mentions, longer for key evidence)
- Only create insert shots for CONCRETE NAMED OBJECTS — not for abstract concepts or atmosphere
- Max 2 insert shots per chunk

RESPOND ONLY WITH A JSON ARRAY — one object per chunk, exactly {n_chunks} items:
[
  {{
    "chunk_index": 0,
    "visual_type": "environment",
    "visual_brief": "dim parking structure at 2am, single orange sodium lamp, oil stains on bare concrete",
    "location": "Exact Location Name From List",
    "image_style": "atmospheric",
    "allow_color": false,
    "framing": "wide_establishing",
    "lens": "24mm",
    "persist_visual": false,
    "is_impact": false,
    "required_elements": ["parking structure", "sodium lamp", "concrete floor"],
    "insert_shots": []
  }},
  {{
    "chunk_index": 1,
    "visual_type": "evidence",
    "visual_brief": "crumpled letter on dark wooden floor, faint handwriting visible",
    "location": "Exact Location Name From List",
    "image_style": "forensic",
    "allow_color": false,
    "framing": "extreme_close",
    "lens": "macro",
    "persist_visual": false,
    "is_impact": true,
    "required_elements": ["crumpled letter", "handwriting", "dark floor"],
    "insert_shots": [
      {{
        "trigger_word": "letter",
        "visual_brief": "crumpled envelope on dark floor, wax seal broken, single overhead lamp",
        "generation_strategy": "object_closeup",
        "hold_seconds": 2.5
      }}
    ]
  }}
]
"""

SCENE_VISUAL_MOMENT_PROMPT = """
You are a cinematographer for a true crime short film series on Instagram.
Each background scene must visually reflect the narration happening at that moment in the story.

NARRATION MAPPED TO EACH SCENE (scene number → what narrator says near that point):
{narration_map}

SCENES TO DIRECT:
{scene_list}

For EACH scene, provide ONE specific visual detail — an object, lighting condition, or
atmospheric element visible IN that location that connects to what is being narrated.

STRICT RULES:
- NO people, NO faces, NO hands, NO bodies, NO silhouettes EVER
- NO camera movement words (pan, tilt, dolly, zoom) — describe only what EXISTS in the frame
- Under 20 words per visual_moment, highly specific and concrete
- Ground it in the location setting — do not contradict it
- If narration mentions a TIME of night/day → reflect it in the lighting
- If narration mentions a PHYSICAL OBJECT → show it sitting in the environment
- If narration mentions an EMOTION or TENSION → pick the most atmospheric environmental echo
- If no obvious link → choose the most ominous static detail of that location

RESPOND ONLY WITH A JSON ARRAY. No explanation. No markdown.
[
  {{"scene_number": 1, "visual_moment": "rotary phone receiver off hook on dispatch console, coiled cord trailing to the floor"}},
  {{"scene_number": 2, "visual_moment": "wall clock frozen at 11:47, glass face cracked at the two"}}
]
"""

REVEAL_MOMENT_PROMPT = """
You are designing cinematic reveal moments for a true crime narration video.
These are moments where the narration mentions an object, clue, evidence,
or critical detail — and the video CUTS to a close-up image of it.

EPISODE SCRIPT:
{script}

NARRATION CHUNKS (in order):
{chunks}

STORY CONTEXT:
{story_context}

Identify 3 to 5 moments where a visual reveal would have MAXIMUM dramatic impact.
Think like a true crime documentary editor.

REVEAL SELECTION CRITERIA — choose moments where narrator mentions:
- A physical object (knife, phone, photograph, letter, key, car, bag, watch)
- A piece of evidence
- A location being described for the FIRST time
- A date or time that matters
- The final line of the episode (always make this a reveal)
- A person watching, hiding, or lurking — use image_type: "subject" (backlit silhouette only)
- A victim or body — use image_type: "subject" (silhouette on floor, unidentifiable)
- A suspect or perpetrator mentioned by name doing something — use image_type: "subject"

SUBJECT TYPE RULES (only when image_type is "subject"):
- Always backlit, always unidentifiable, no face ever visible
- Victim on floor: crumpled silhouette shape, no gore, no blood visible
- Lurking figure: distant dark shape in doorway or shadow
- Maximum 1 subject reveal per episode

For each reveal:
- chunk_index: which narration chunk triggers the cut (0-based)
- trigger_phrase: exact words being spoken at the cut (max 6 words)
- image_type: one of: evidence / location_detail / subject / document
- image_brief: vivid description of the close-up image (under 50 words, NO people, NO faces)
- cut_type: one of: hard / flash / smash / dissolve
- hold_seconds: how long the reveal stays on screen (2 to 4)
- text_overlay: null OR short string like "EXHIBIT A" / "DAY 3" / location name
- return_to_scene: true

RESPOND ONLY WITH JSON ARRAY. No explanation. No markdown.
[
  {{
    "chunk_index": 2,
    "trigger_phrase": "the knife was still there",
    "image_type": "evidence",
    "image_brief": "kitchen knife on dark wooden floor, single overhead light, dried rust-colored stain on blade, extreme close-up",
    "cut_type": "flash",
    "hold_seconds": 2,
    "text_overlay": "EXHIBIT 7",
    "return_to_scene": true
  }}
]
"""


# ── GENERATION STRATEGY ───────────────────────────────────────────────────────
# Maps visual_type → how the image generator should treat this scene.
# "environment"   → standard prompt, prompt corrector allowed
# "forensic"      → evidence style, NO corrector, Claude Vision validation mandatory
# "object_closeup"→ extreme macro style, object isolated on dark surface
# "person_trace"  → silhouette/backlit rules, NO corrector
# "document"      → document style, text legibility priority

GENERATION_STRATEGY_MAP = {
    "environment":  "environment",
    "object":       "object_closeup",
    "evidence":     "forensic",
    "person_trace": "person_trace",
}


def derive_generation_strategy(visual_type: str, is_impact: bool) -> str:
    """Derive generation strategy from visual_type. Impact scenes always use forensic path."""
    if is_impact:
        return "forensic"
    return GENERATION_STRATEGY_MAP.get(visual_type, "environment")


@retry_with_backoff(max_retries=3, base_wait=5.0, retryable_exceptions=(RetryableException,))
def call_claude(prompt: str, max_tokens: int = 1500) -> str:
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        raise RetryableException(f"Claude call failed: {e}")


def assign_arc_position(scene_index: int, total_scenes: int) -> str:
    """Pure logic — no API call needed."""
    ratio = scene_index / max(total_scenes - 1, 1)
    if ratio <= 0.15:
        return "establish"
    elif ratio <= 0.45:
        return "build"
    elif ratio <= 0.65:
        return "investigation"
    elif ratio <= 0.85:
        return "revelation"
    else:
        return "cliffhanger"


def build_scene_flux_prompt(location_visual: str, arc_position: str = "build", visual_moment: str = "") -> str:
    """
    Build FLUX prompt for a scene image (dark atmospheric tone).
    Uses location_visual from story_bible + arc-specific angle + optional narration-derived
    visual moment + hard STYLE_SUFFIX.
    """
    clean = location_visual.strip().rstrip(".")
    variation = ARC_SCENE_VARIATION.get(arc_position, ARC_SCENE_VARIATION["build"])
    if visual_moment:
        return f"{clean}, {variation}, {visual_moment}, {STYLE_SUFFIX}"
    return f"{clean}, {variation}, {STYLE_SUFFIX}"


def build_scene_color_prompt(location_visual: str, arc_position: str = "build", visual_moment: str = "") -> str:
    """
    Build FLUX prompt for a scene where color must be preserved (allow_color=True).
    Uses COLOR_STYLE_SUFFIX instead of STYLE_SUFFIX to avoid forcing monochrome.
    """
    clean = location_visual.strip().rstrip(".")
    variation = ARC_SCENE_VARIATION.get(arc_position, ARC_SCENE_VARIATION["build"])
    if visual_moment:
        return f"{clean}, {variation}, {visual_moment}, {COLOR_STYLE_SUFFIX}"
    return f"{clean}, {variation}, {COLOR_STYLE_SUFFIX}"


def build_reveal_flux_prompt(image_brief: str, image_type: str) -> str:
    """Build FLUX prompt for a reveal image."""
    clean = image_brief.strip().rstrip(".")
    if image_type == "subject":
        return f"{clean}, {SUBJECT_STYLE_SUFFIX}"
    elif image_type == "document":
        return f"{clean}, {DOCUMENT_STYLE_SUFFIX}"
    else:
        return f"{clean}, {EVIDENCE_STYLE_SUFFIX}"


def generate_scene_visual_moments(
    episode_chunks: list,
    scene_list: list,
    episode_number: int,
) -> dict:
    """
    ONE Claude call per episode.
    Maps each scene to nearby narration and returns {scene_number: visual_moment_str}.
    Falls back to empty dict on any failure — scenes still render with location-only prompts.
    """
    try:
        total_chunks = len(episode_chunks)
        total_scenes = len(scene_list)

        narration_lines = []
        for scene in scene_list:
            i = scene["scene_number"] - 1
            chunk_idx = min(round(i * total_chunks / max(total_scenes, 1)), total_chunks - 1)
            text = episode_chunks[chunk_idx].get("text", "")[:120] if episode_chunks else ""
            narration_lines.append(f"Scene {scene['scene_number']}: \"{text}\"")

        scene_lines = [
            f"Scene {s['scene_number']} [{s['arc_position']}] at \"{s['location_name']}\""
            for s in scene_list
        ]

        prompt = SCENE_VISUAL_MOMENT_PROMPT.format(
            narration_map="\n".join(narration_lines),
            scene_list="\n".join(scene_lines),
        )

        raw = call_claude(prompt, max_tokens=2000)
        raw = raw.replace("```json", "").replace("```", "").strip()
        moments = json.loads(raw)

        result = {m["scene_number"]: m.get("visual_moment", "") for m in moments}
        logger.info(f"Episode {episode_number + 1}: generated {len(result)} scene visual moments")
        return result

    except Exception as e:
        logger.warning(f"Episode {episode_number + 1}: scene visual moments failed — using location-only prompts ({e})")
        return {}


def generate_reveal_moments(
    script: str,
    structured_chunks: list,
    context: dict,
    episode_number: int,
) -> list:
    """
    ONE Claude call per episode.
    Identifies dramatic reveal moments and designs cut specs.
    """
    try:
        chunk_summary = []
        for i, chunk in enumerate(structured_chunks):
            chunk_summary.append(
                f"[{i}] ({chunk.get('type', '?')}): {chunk.get('text', '')[:80]}"
            )
        chunks_text = "\n".join(chunk_summary)

        story_context = (
            f"Title: {context.get('story_title', 'Unknown')}\n"
            f"Summary: {context.get('story_summary', '')[:200]}\n"
            f"Episode: {episode_number + 1} of {context.get('total_episodes', 1)}"
        )

        prompt = REVEAL_MOMENT_PROMPT.format(
            script=script[:2000],
            chunks=chunks_text,
            story_context=story_context,
        )

        raw = call_claude(prompt, max_tokens=1500)
        raw = raw.replace("```json", "").replace("```", "").strip()
        reveals = json.loads(raw)

        validated = []
        total_chunks = len(structured_chunks)

        for r in reveals:
            chunk_idx = int(r.get("chunk_index", 0))
            if chunk_idx >= total_chunks:
                chunk_idx = total_chunks - 1

            image_type = r.get("image_type", "evidence")
            image_brief = r.get("image_brief", "")
            flux_prompt = build_reveal_flux_prompt(image_brief, image_type)

            validated.append({
                "chunk_index": chunk_idx,
                "trigger_phrase": r.get("trigger_phrase", ""),
                "image_type": image_type,
                "image_brief": image_brief,
                "flux_prompt": flux_prompt,
                "cut_type": r.get("cut_type", "hard"),
                "hold_seconds": min(max(int(r.get("hold_seconds", 2)), 2), 4),
                "text_overlay": r.get("text_overlay", None),
                "return_to_scene": r.get("return_to_scene", True),
                "image_path": None,
            })

        logger.info(f"Episode {episode_number + 1}: {len(validated)} reveal moments")
        return validated

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse reveals JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Reveal generation failed: {e}")
        return []


def decompose_chunks_to_visuals(
    episode_chunks: list,
    context: dict,
    episode_number: int,
) -> list:
    """
    ONE Claude Haiku call per episode.
    Returns [{chunk_index, visual_type, visual_brief, location, image_style}] — one per chunk.
    Falls back to empty list on any failure (caller uses location-cycling instead).
    """
    try:
        location_names = list(context.get("story_bible", {}).get("location_visuals", {}).keys())
        if not location_names:
            return []

        story_context = (
            f"Title: {context.get('story_title', 'Unknown')}\n"
            f"Summary: {context.get('story_summary', '')[:300]}\n"
            f"Episode: {episode_number + 1} of {context.get('total_episodes', 1)}"
        )

        chunks_text = "\n".join(
            f"[{i}] {chunk.get('text', '')[:160]}"
            for i, chunk in enumerate(episode_chunks)
        )

        prompt = DECOMPOSE_PROMPT.format(
            story_context=story_context,
            locations_list="\n".join(f"- {n}" for n in location_names),
            chunks_text=chunks_text,
            n_chunks=len(episode_chunks),
        )

        raw = call_claude(prompt, max_tokens=5000)
        raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array in decompose response")

        visuals = json.loads(raw[start:end])

        # Pad/trim to match chunk count
        while len(visuals) < len(episode_chunks):
            i = len(visuals)
            visuals.append({
                "chunk_index": i,
                "visual_type": "environment",
                "visual_brief": "",
                "location": location_names[i % len(location_names)],
                "image_style": "atmospheric",
                "persist_visual": False,
                "is_impact": False,
                "required_elements": [],
                "insert_shots": [],
            })
        visuals = visuals[:len(episode_chunks)]

        # Force correct chunk indices + default new fields
        for i, v in enumerate(visuals):
            v["chunk_index"] = i
            v.setdefault("persist_visual", False)
            v.setdefault("is_impact", False)
            v.setdefault("required_elements", [])
            v.setdefault("insert_shots", [])
            # Validate insert shots structure
            clean_inserts = []
            for ins in v.get("insert_shots", []):
                if isinstance(ins, dict) and ins.get("trigger_word") and ins.get("visual_brief"):
                    ins.setdefault("generation_strategy", "object_closeup")
                    ins.setdefault("hold_seconds", 2.0)
                    ins["image_path"] = None
                    clean_inserts.append(ins)
            v["insert_shots"] = clean_inserts

        # Validate/fix location names
        for v in visuals:
            if v.get("location") not in location_names:
                v["location"] = location_names[0]

        n_inserts = sum(len(v.get("insert_shots", [])) for v in visuals)
        n_impact  = sum(1 for v in visuals if v.get("is_impact"))
        n_persist = sum(1 for v in visuals if v.get("persist_visual"))
        logger.info(
            f"Episode {episode_number + 1}: decomposed {len(visuals)} chunks — "
            f"{n_impact} impact, {n_persist} persist, {n_inserts} insert shots"
        )
        return visuals

    except Exception as e:
        logger.warning(
            f"Episode {episode_number + 1}: chunk decomposition failed ({e}) — "
            "falling back to location cycling"
        )
        return []


def generate_scene_prompts(context: dict, episode_number: int) -> list:
    """
    Agent 3: Build scene list + reveal moments for one episode.

    Narration-first path (primary):
      ONE Claude Haiku call decomposes each narration chunk into a visual spec.
      Each chunk gets exactly one scene — content tracks what is being narrated.

    Location-cycling path (fallback):
      Used when no narration chunks exist or decomposition fails.
      Assigns locations by cycling, adds visual moments via a second Claude call.

    Reveals: one Claude call per episode (both paths).
    """
    logger.info(f"Directing episode {episode_number + 1}...")

    try:
        if episode_number >= len(context.get("scripts", [])):
            raise SceneDirectorException(f"Episode {episode_number + 1} script not found")

        script = context["scripts"][episode_number]
        if not script or not script.strip():
            raise SceneDirectorException("Episode script is empty")

        location_visuals = context.get("story_bible", {}).get("location_visuals", {})
        if not location_visuals:
            raise SceneDirectorException("No location visuals in story bible")

        location_names = list(location_visuals.keys())

        structured = context.get("structured_narration", [])
        episode_chunks = structured[episode_number] if episode_number < len(structured) else []

        prompts = []

        # ── PATH 1: NARRATION-FIRST DECOMPOSITION ─────────────────────
        if episode_chunks:
            visual_plan = decompose_chunks_to_visuals(episode_chunks, context, episode_number)

            if visual_plan:
                total_scenes = len(visual_plan)

                for i, vis in enumerate(visual_plan):
                    chunk_idx = vis["chunk_index"]
                    location = vis.get("location", location_names[i % len(location_names)])
                    visual_type = vis.get("visual_type", "environment")
                    visual_brief = vis.get("visual_brief", "")

                    if location not in location_visuals:
                        location = location_names[i % len(location_names)]

                    location_visual = location_visuals.get(location, list(location_visuals.values())[0])
                    arc_position = assign_arc_position(i, total_scenes)
                    arc_data = ARC_POSITIONS[arc_position]
                    allow_color = vis.get("allow_color", False)

                    # Build FLUX prompt by visual type
                    framing = vis.get("framing", "")
                    lens    = vis.get("lens", "")

                    is_impact  = vis.get("is_impact", False)
                    persist_visual = vis.get("persist_visual", False)
                    required_elements = vis.get("required_elements", [])
                    insert_shots = vis.get("insert_shots", [])
                    generation_strategy = derive_generation_strategy(visual_type, is_impact)

                    if visual_type in ("object", "evidence") or generation_strategy == "forensic":
                        image_prompt = (
                            f"{visual_brief}, {EVIDENCE_STYLE_SUFFIX}"
                            if visual_brief
                            else build_scene_flux_prompt(location_visual, arc_position)
                        )
                        photo_evidence = True
                    elif visual_type == "person_trace":
                        image_prompt = (
                            f"{visual_brief}, {SUBJECT_STYLE_SUFFIX}"
                            if visual_brief
                            else build_scene_flux_prompt(location_visual, arc_position)
                        )
                        photo_evidence = False
                    elif allow_color:
                        image_prompt = build_scene_color_prompt(location_visual, arc_position, visual_brief)
                        photo_evidence = False
                    else:
                        image_prompt = build_scene_flux_prompt(location_visual, arc_position, visual_brief)
                        cine_parts = []
                        if framing and framing in FRAMING_TO_FLUX:
                            cine_parts.append(FRAMING_TO_FLUX[framing])
                        if lens and lens in LENS_TO_FLUX:
                            cine_parts.append(LENS_TO_FLUX[lens])
                        if cine_parts:
                            image_prompt = image_prompt + ", " + ", ".join(cine_parts)
                        photo_evidence = False

                    prompts.append({
                        "scene_number": i + 1,
                        "chunk_index": chunk_idx,
                        "image_prompt": image_prompt,
                        "arc_position": arc_position,
                        "duration": arc_data["duration"],
                        "duration_seconds": DURATION_SECONDS[arc_data["duration"]],
                        "frames": DURATION_FRAMES[arc_data["duration"]],
                        "location": location,
                        "location_name": location,
                        "zoom_direction": arc_data["zoom"],
                        "image_type": visual_type,
                        "visual_brief": visual_brief,
                        "allow_color": allow_color,
                        "photo_evidence_treatment": photo_evidence,
                        "generation_strategy": generation_strategy,
                        "required_elements": required_elements,
                        "is_impact": is_impact,
                        "persist_visual": persist_visual,
                        "insert_shots": insert_shots,
                        "image_path": None,
                        "is_padding": False,
                    })

                    logger.info(
                        f"  Scene {i+1}/{total_scenes} [{arc_position}] "
                        f"[{visual_type}] — {location}"
                    )

        # ── PATH 2: LOCATION-CYCLING FALLBACK ─────────────────────────
        if not prompts:
            if episode_chunks:
                logger.warning(
                    f"Episode {episode_number + 1}: decomposition returned nothing — "
                    "falling back to location cycling"
                )
            else:
                logger.warning(
                    f"No structured narration for episode {episode_number + 1} — "
                    "using location-only scene prompts"
                )

            episode_plans = context.get("episode_plans", [])
            ep_data = episode_plans[episode_number] if episode_number < len(episode_plans) else {}
            locations_used = ep_data.get("locations_used", location_names)

            audio_files = context.get("audio_files", [])
            voiceover_ts = context.get("voiceover_timestamps", [])
            if episode_number < len(voiceover_ts) and voiceover_ts[episode_number]:
                audio_duration_s = voiceover_ts[episode_number][-1]["end_ms"] / 1000
            elif episode_number < len(audio_files) and audio_files[episode_number]:
                try:
                    size_bytes = os.path.getsize(audio_files[episode_number])
                    audio_duration_s = size_bytes / (24000 * 2)
                except Exception:
                    audio_duration_s = 60
            else:
                audio_duration_s = 60

            min_scenes = max(5, int(audio_duration_s / 8))
            scene_locations = list(locations_used) if locations_used else list(location_names)
            while len(scene_locations) < min_scenes:
                scene_locations = scene_locations + scene_locations
            scene_locations = scene_locations[:min_scenes]

            total_scenes = len(scene_locations)

            for i, location in enumerate(scene_locations):
                arc_position = assign_arc_position(i, total_scenes)
                arc_data = ARC_POSITIONS[arc_position]
                visual_desc = location_visuals.get(location, "") or list(location_visuals.values())[0]
                image_prompt = build_scene_flux_prompt(visual_desc, arc_position)

                prompts.append({
                    "scene_number": i + 1,
                    "chunk_index": i,
                    "image_prompt": image_prompt,
                    "arc_position": arc_position,
                    "duration": arc_data["duration"],
                    "duration_seconds": DURATION_SECONDS[arc_data["duration"]],
                    "frames": DURATION_FRAMES[arc_data["duration"]],
                    "location": location,
                    "location_name": location,
                    "zoom_direction": arc_data["zoom"],
                    "image_type": "narrative",
                    "visual_brief": "",
                    "photo_evidence_treatment": False,
                    "image_path": None,
                    "is_padding": False,
                })
                logger.info(f"  Scene {i+1}/{total_scenes} [{arc_position}] — {location}")

            # Narration-aware visual moments in fallback mode
            if episode_chunks:
                visual_moments = generate_scene_visual_moments(episode_chunks, prompts, episode_number)
                for p in prompts:
                    moment = visual_moments.get(p["scene_number"], "")
                    if moment:
                        visual_desc = (
                            location_visuals.get(p["location_name"], "")
                            or list(location_visuals.values())[0]
                        )
                        p["image_prompt"] = build_scene_flux_prompt(
                            visual_desc, p["arc_position"], moment
                        )
                        p["visual_brief"] = moment

        # ── REVEAL MOMENTS — one Claude call ──────────────────────────
        if episode_chunks:
            reveals = generate_reveal_moments(script, episode_chunks, context, episode_number)
        else:
            reveals = []

        if "reveal_moments" not in context:
            context["reveal_moments"] = []
        while len(context["reveal_moments"]) <= episode_number:
            context["reveal_moments"].append([])
        context["reveal_moments"][episode_number] = reveals

        total_seconds = sum(p["duration_seconds"] for p in prompts)
        logger.info(
            f"Episode {episode_number + 1}: {len(prompts)} scenes, "
            f"{len(reveals)} reveals, ~{total_seconds}s"
        )

        return prompts

    except SceneDirectorException as e:
        logger.error(f"Scene direction failed: {e}")
        context["errors"].append({
            "agent": "scene_director",
            "episode": episode_number + 1,
            "error": str(e)
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise SceneDirectorException(f"Unexpected: {e}")