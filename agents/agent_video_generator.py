"""
Agent: Video Generator — Wan2.1-I2V via Kaggle

WHEN TO USE WAN vs KEN BURNS
  Wan2.1-I2V excels at organic ENVIRONMENTAL motion: mist drifting, fire
  flickering, rain streaking, smoke curling.  It does NOT reliably simulate
  camera motion (zooms, pushes) — Ken Burns does that more precisely and with
  zero failure rate.

  A scene qualifies for Wan only when:
    1. arc_position is "revelation" or "cliffhanger"  (high-impact moments)
    2. The scene's image_prompt / location text contains at least one
       environmental motion keyword (hard gate — Ken Burns used otherwise)

ROBUSTNESS LAYERS
  1. Environmental keyword gate       — 80 % of scenes filtered to Ken Burns
  2. Content-first prompts            — describe WHAT moves, not camera ops
  3. 49 frames (~3 s at 16 fps)       — less drift than 81 frames
  4. guidance_scale = 7.0             — stronger prompt adherence
  5. Hardened negative prompt         — penalises morphing / flickering / drift
  6. Post-generation size validation  — clips < 300 KB are rejected
  7. Per-episode cap (3 clips max)    — conserves Kaggle GPU quota
  8. Compositor also guards size      — double-check before Path-V branch

VIDEO_SOURCE=kaggle in .env enables this agent.
KAGGLE_VIDEO_URL must point to the same ngrok URL as KAGGLE_IMAGE_URL.
"""

import os
import re
import base64
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import AgentLogger
from utils.retry import retry_with_backoff
from utils.errors import RetryableException

load_dotenv()

logger = AgentLogger("VideoGenerator")

VIDEO_SOURCE     = os.getenv("VIDEO_SOURCE", "none")
KAGGLE_VIDEO_URL = os.getenv("KAGGLE_VIDEO_URL", "").rstrip("/")
VIDEO_CLIP_DIR   = Path("outputs/video_clips")
VIDEO_CLIP_DIR.mkdir(parents=True, exist_ok=True)

# Hard limits
_MIN_CLIP_BYTES        = 300_000   # 300 KB — any real 33-frame 832x480 MP4 exceeds this

# Only revelation / cliffhanger arc positions are high-impact enough to justify
# the Kaggle GPU cost.  Forensic / object_closeup are intentionally excluded —
# they need exact detail preservation that Wan destroys.
_VIDEO_ARC_POSITIONS = {"revelation", "cliffhanger"}

# ---------------------------------------------------------------------------
# Environmental motion vocabulary
# Key  : substring searched in image_prompt + location (case-insensitive)
# Value: what Wan should animate — always organic content motion, never camera
# ---------------------------------------------------------------------------
_ENVIRONMENTAL_MOTION_MAP = {
    "mist":        "mist slowly drifting and curling through the darkness",
    "fog":         "fog rolling in, dim light filtering through shifting haze",
    "fire":        "fire flickering and dancing, warm light casting moving shadows",
    "flame":       "flame gently wavering, warm light rippling across surfaces",
    "candle":      "candlelight flickering softly, shadows trembling on the walls",
    "candlelight": "candlelight trembling, soft golden shadows slowly shifting",
    "firelight":   "firelight pulsing softly, warm glow dancing across the room",
    "smoke":       "smoke curling upward in slow thin wisps",
    "ember":       "embers glowing and fading, tiny sparks drifting upward",
    "ash":         "ash drifting slowly downward through still, heavy air",
    "spark":       "sparks floating upward and fading in the darkness",
    "rain":        "rain streaking down a window, drops shattering on stone below",
    "water":       "water surface shimmering with gentle slow ripples",
    "ripple":      "slow ripples spreading outward and fading on the surface",
    "reflection":  "reflections shimmering and slowly shifting on the dark surface",
    "curtain":     "heavy curtain slowly swaying in a barely perceptible draft",
    "drape":       "drapes shifting very slowly in a faint invisible breeze",
    "shadow":      "shadows slowly deepening and shifting across the stone walls",
    "dust":        "dust motes floating slowly in a shaft of dim, cold light",
    "steam":       "steam rising in slow wisps, curling and dissipating upward",
    "haze":        "atmospheric haze slowly drifting, light diffusing through it",
    "moonlight":   "moonlight slowly moving across the floor as clouds drift past",
    "lantern":     "lantern flame flickering softly, light swaying across stone",
    "torch":       "torchlight flickering, thin smoke curling upward slowly",
    "drift":       "atmospheric particles slowly drifting through the light",
    "vapor":       "vapour rising in slow wisps, curling and thinning",
    "aurora":      "aurora slowly rippling and shifting across the sky",
    "lightning":   "distant lightning briefly illuminating the scene then fading",
}

# Hardened negative prompt — based on official Wan2.1 README + community practice.
# DO NOT include "no motion" — CFG can invert this and suppress all motion.
# "still picture, static" is the documented phrase that actually helps.
NEGATIVE_VIDEO_PROMPT = (
    "bright tones, overexposed, still picture, static, blurred details, "
    "low quality, worst quality, JPEG compression residue, "
    "fast motion, abrupt motion, camera shake, jump cut, "
    "morphing, warping, melting, flickering, temporal artifacts, "
    "glitching, color shift, object distortion, dissolving, "
    "smearing, streaking, excessive noise, face distortion, "
    "person, human, face, figure, text, watermark, subtitle, logo"
)

# Per-arc atmospheric mood suffix (appended after the motion description)
_ARC_MOOD = {
    "revelation":    "dark, tense, suspenseful atmosphere",
    "cliffhanger":   "deeply ominous, breathless, still atmosphere",
    "establish":     "moody, slowly building atmosphere",
    "build":         "brooding, atmospheric dim lighting",
    "investigation": "cold, clinical, tense atmosphere",
}

def _cliffhanger_video_prompt(scene: dict) -> str:
    """
    Wan I2V prompt for the cliffhanger.

    Key rules from official docs + community research:
    - Use explicit motion VERBS ("drifts and curls"), not just modifiers ("slow")
    - Include "static locked-off camera" to prevent unwanted camera drift
    - No quality tokens (4K, cinematic) — Wan's T5 encoder ignores them
    - 30–60 words is the sweet spot; under 80
    - Do NOT include "no motion" or "still" in positive — that suppresses motion
    """
    keyword, motion_desc = _find_environmental_motion(scene)
    location = scene.get("location", "")
    if motion_desc:
        if location:
            return f"{motion_desc}. Static locked-off camera. {location}."
        return f"{motion_desc}. Static locked-off camera."

    # No environmental keyword — derive from the scene's own visual description.
    visual_brief = scene.get("visual_brief", "").strip()
    image_prompt  = scene.get("image_prompt", "").strip()
    base = " ".join((visual_brief or image_prompt or "").split()[:12])
    if base:
        return f"Subtle atmospheric motion gently shifts across the scene. {base}. Static locked-off camera."
    return "Subtle atmospheric motion, wisps of vapor gently drifting, faint environmental movement. Static locked-off camera."


def _find_environmental_motion(scene: dict) -> tuple:
    """
    Scan visual-only fields for environmental motion keywords.
    Uses word-boundary matching to prevent substring false-positives
    (e.g. "grain" matching "rain", "training" matching "rain").
    Returns (keyword, motion_description) if found, else (None, None).
    """
    corpus = " ".join(filter(None, [
        scene.get("image_prompt", ""),
        scene.get("location", ""),
        scene.get("visual_brief", ""),
    ])).lower()

    for keyword, motion_desc in _ENVIRONMENTAL_MOTION_MAP.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', corpus):
            return keyword, motion_desc
    return None, None


def _video_prompt_for_scene(scene: dict):
    """
    Wan I2V prompt.  Returns None when no environmental motion qualifies
    (Ken Burns used instead).
    """
    keyword, motion_desc = _find_environmental_motion(scene)
    if not motion_desc:
        return None

    location = scene.get("location", "")
    if location:
        return f"{motion_desc}. Static locked-off camera. {location}."
    return f"{motion_desc}. Static locked-off camera."


def _guaranteed_video_prompt(scene: dict) -> str:
    """
    Always returns a Wan I2V prompt — no keyword gate.
    Uses explicit motion verbs (documented to drive motion in Wan2.1 vs. just
    "slow" modifier) and "static locked-off camera" to prevent drift artifacts.
    """
    keyword, motion_desc = _find_environmental_motion(scene)
    location = scene.get("location", "")

    if motion_desc:
        if location:
            return f"{motion_desc}. Static locked-off camera. {location}."
        return f"{motion_desc}. Static locked-off camera."

    # No keyword — derive motion anchor from the scene's own visual description.
    visual_brief = scene.get("visual_brief", "").strip()
    image_prompt  = scene.get("image_prompt", "").strip()
    base = " ".join((visual_brief or image_prompt or "").split()[:12])

    if base and location:
        return f"Subtle atmospheric motion gently shifts across the scene. {base}. {location}. Static locked-off camera."
    if base:
        return f"Subtle atmospheric motion gently shifts across the scene. {base}. Static locked-off camera."
    if location:
        return f"Subtle atmospheric motion, wisps of vapor gently drifting. {location}. Static locked-off camera."
    return "Subtle atmospheric motion, wisps of vapor gently drifting, faint environmental movement. Static locked-off camera."


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _validate_clip(path: Path) -> bool:
    """
    Return True only if the clip file meets minimum quality thresholds.
    Size gate: any real 49-frame 832x480 H.264 MP4 will be well above 300 KB.
    A file below that threshold is a corrupt, static, or truncated output.
    """
    try:
        size = path.stat().st_size
        if size < _MIN_CLIP_BYTES:
            logger.warning(
                f"Clip rejected — too small: {size // 1024} KB "
                f"(min {_MIN_CLIP_BYTES // 1024} KB): {path.name}"
            )
            return False
        return True
    except OSError as e:
        logger.warning(f"Clip validation error: {e}")
        return False


def _upscale_wan_clip(clip_path: Path, target_w: int = 1920, target_h: int = 1080) -> bool:
    """
    Upscale a Wan clip from 832x480 to 1920x1080 using Real-ESRGAN 4x.
    Operates in-place: replaces clip_path with the upscaled version.

    Only active on Colab (requires basicsr + realesrgan packages).
    Returns True on success; False when the packages aren't installed (silent
    fallback — the compositor will handle upscaling via bicubic instead).

    VRAM budget on A100-80GB:
      Wan model (Flask process):  ~45 GB
      ESRGAN inference (here):    ~1-2 GB (fp16, full-frame — no tiling needed)
      Headroom:                   >30 GB — safe.
    """
    try:
        import cv2
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        return False  # Not on Colab — skip silently

    import urllib.request
    import tempfile

    try:
        model_path = "/content/RealESRGAN_x4plus.pth"
        if not Path(model_path).exists():
            logger.info("Downloading RealESRGAN_x4plus weights (~64 MB)...")
            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model_path,
            )

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,    # safe tile size — prevents OOM if Wan leaves less free VRAM
            tile_pad=10,
            pre_pad=0,
            half=True,   # fp16 — A100 handles this natively
        )

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return False

        logger.info(f"Real-ESRGAN 4x: {len(frames)} frames @ {fps:.0f}fps → {target_w}x{target_h}")
        t0 = time.time()
        upscaled = []
        for frame in frames:
            # enhance() returns BGR; outscale=4 gives 3328x1920, then resize to exact target
            out, _ = upsampler.enhance(frame, outscale=4)
            out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            upscaled.append(out)
        logger.info(f"Real-ESRGAN done in {time.time()-t0:.1f}s")

        # Write raw frames via cv2, then re-encode with ffmpeg for proper H.264/yuv420p.
        # Write to a temp output first — only replace the original on success so a
        # failed ffmpeg call never corrupts the original Wan clip.
        tmp_raw = str(clip_path) + "_sr_raw.mp4"
        tmp_out = str(clip_path) + "_sr_out.mp4"
        writer = cv2.VideoWriter(
            tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h),
        )
        for frame in upscaled:
            writer.write(frame)
        writer.release()

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_raw,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-b:v", "8M",
             tmp_out],
            capture_output=True, timeout=300,
        )
        Path(tmp_raw).unlink(missing_ok=True)

        if result.returncode == 0 and Path(tmp_out).exists() and Path(tmp_out).stat().st_size > _MIN_CLIP_BYTES:
            Path(tmp_out).replace(clip_path)  # atomic rename — original only replaced on success
            logger.info(f"Upscaled clip: {clip_path.name} ({clip_path.stat().st_size // 1024} KB)")
            return True
        else:
            Path(tmp_out).unlink(missing_ok=True)
            logger.warning(f"ffmpeg re-encode failed (keeping original Wan clip): {result.stderr.decode()[-300:]}")
            return False

    except Exception as e:
        logger.warning(f"Real-ESRGAN upscale failed (non-fatal, keeping original): {e}")
        return False


@retry_with_backoff(max_retries=3, base_wait=8.0, retryable_exceptions=(RetryableException,))
def generate_video_clip(image_path: str, prompt: str, clip_name: str) -> str:
    """
    POST image + prompt to the Kaggle /generate_video endpoint.
    Returns local path to the saved MP4 clip, or raises on failure.

    Parameters sent to Wan2.1-I2V (per official README + Issue #73/#356):
      num_frames          = 81   (480P model may hardcode 81 regardless of value)
      guidance_scale      = 5.0  (official I2V default; avoids static-lock artifacts)
      flow_shift          = 4.0  (>default 3.0 breaks AI-image static failure)
      num_inference_steps = 40   (official I2V recommendation; T2V uses 50)
    """
    import requests

    out_path = VIDEO_CLIP_DIR / f"{clip_name}.mp4"
    if out_path.exists() and _validate_clip(out_path):
        logger.info(f"Video clip cached and valid: {out_path}")
        return str(out_path)

    if not KAGGLE_VIDEO_URL:
        raise RetryableException("KAGGLE_VIDEO_URL not set")

    img_b64 = _encode_image(image_path)
    payload = {
        "image_base64":    img_b64,
        "prompt":          prompt,
        "negative_prompt": NEGATIVE_VIDEO_PROMPT,
        "num_frames":      81,    # 480P model may hardcode 81 regardless (GitHub Issue #73)
        "guidance_scale":  5.0,   # official README default for I2V; 5.5+ risks over-sharpening
        "flow_shift":      4.0,   # >3.0 default breaks AI-image static-lock-in (GitHub Issue #356)
        "num_inference_steps": 40, # official README recommends 40 for I2V (not 50 like T2V)
    }

    try:
        # Use async endpoint so each 2-3 min Wan clip doesn't block the connection.
        # Submit the job, get a job_id, poll every 30s until done.
        async_url  = f"{KAGGLE_VIDEO_URL}/generate_video_async"
        status_url = f"{KAGGLE_VIDEO_URL}/generate_video_status"

        resp = requests.post(async_url, json=payload, timeout=30)
        if resp.status_code == 404:
            # Fallback: server doesn't have async endpoint yet
            resp = requests.post(f"{KAGGLE_VIDEO_URL}/generate_video",
                                 json=payload, timeout=360)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RetryableException(f"Server error: {data['error'][:200]}")
            video_bytes = base64.b64decode(data["video_base64"])
            out_path.write_bytes(video_bytes)
        else:
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            logger.info(f"Video job submitted: {job_id} — polling every 30s")

            deadline = time.time() + 1800  # 30-minute max
            while time.time() < deadline:
                time.sleep(30)
                poll = requests.get(f"{status_url}/{job_id}", timeout=15)
                poll.raise_for_status()
                job = poll.json()
                status = job.get("status", "unknown")
                if status == "done":
                    video_bytes = base64.b64decode(job["video_base64"])
                    out_path.write_bytes(video_bytes)
                    break
                elif status == "error":
                    raise RetryableException(f"Server error: {job.get('error','')[:200]}")
                # status == "running" or "queued" — keep polling
            else:
                raise RetryableException("Video generation timed out after 30 minutes")

        # Upscale from Wan's native 832x480 to 1920x1080 with Real-ESRGAN.
        # Runs only on Colab (where basicsr/realesrgan are installed); silent no-op elsewhere.
        # The compositor's Path V scale/crop step then becomes a pixel-exact pass-through.
        _upscale_wan_clip(out_path)

        if not _validate_clip(out_path):
            out_path.unlink(missing_ok=True)
            raise RetryableException("Clip failed size validation — likely static or corrupt")

        logger.info(f"Video clip saved: {out_path.name} ({out_path.stat().st_size // 1024} KB)")
        return str(out_path)

    except RetryableException:
        raise
    except Exception as e:
        raise RetryableException(f"Video generation failed: {e}")


def run_agent_video_generator(context: dict) -> dict:
    """
    Generate exactly 3 Wan clips per episode — guaranteed, regardless of story content.

    Slot 1 (cliffhanger): always generated, wide-environment shot. Keyword detection
    tried first; falls back to scene's own visual description if no keyword matches.

    Slots 2–3 (revelation): always generated for the top 2 high-impact scenes
    (reveal_moments first, then revelation/cliffhanger arc scenes). Keyword detection
    tried first for best prompt accuracy; guaranteed fallback if no keyword matches.

    Only runs when VIDEO_SOURCE=kaggle and KAGGLE_VIDEO_URL is set.
    """
    if VIDEO_SOURCE != "kaggle" or not KAGGLE_VIDEO_URL:
        logger.info("Video generation skipped (VIDEO_SOURCE != kaggle or URL not set)")
        return context

    logger.info("Generating Wan2.1-I2V clips: 1 cliffhanger + 2 revelation (guaranteed — 3 total)...")

    for ep_idx, episode_scenes in enumerate(context.get("scene_prompts", [])):
        cliff_generated = 0   # cliffhanger slot: 0 or 1 (mandatory wide-shot)
        rev_generated   = 0   # revelation slots: 0–2 (keyword-gated)
        failed          = 0

        # ── PHASE 1: CLIFFHANGER (mandatory, no keyword gate) ─────────────────
        # Always generate one Wan wide-environment clip for the cliffhanger scene.
        # This becomes the episode's final image — Wan motion is far more visceral
        # than a frozen still. We pick the first scene with arc_position=cliffhanger.
        for s_idx, scene in enumerate(episode_scenes):
            if cliff_generated >= 1:
                break
            if scene.get("arc_position") != "cliffhanger":
                continue
            img = scene.get("image_path", "")
            if not img or not Path(img).exists():
                continue
            if scene.get("video_path") and Path(scene["video_path"]).exists():
                if _validate_clip(Path(scene["video_path"])):
                    logger.info(f"ep{ep_idx+1} cliffhanger: cached clip reused")
                    cliff_generated += 1
                    continue
            prompt = _cliffhanger_video_prompt(scene)
            clip_name = f"ep{ep_idx+1}_scene{s_idx+1}_cliffhanger"
            logger.info(f"ep{ep_idx+1} cliffhanger wide-shot: {prompt[:70]}...")
            try:
                vp = generate_video_clip(img, prompt, clip_name)
                context["scene_prompts"][ep_idx][s_idx]["video_path"] = vp
                cliff_generated += 1
                time.sleep(3)
            except Exception as e:
                logger.warning(f"ep{ep_idx+1} cliffhanger Wan failed: {e}")
                failed += 1

        # ── PHASE 2: REVELATION scenes (2 guaranteed slots) ───────────────────
        # Keyword detection is tried first; if no keyword matches, falls back to
        # scene-derived prompt — so these 2 clips always generate regardless of
        # whether the scene mentions mist/fire/rain/etc.
        _MAX_REVELATION_CLIPS = 2

        ep_reveals = []
        if ep_idx < len(context.get("reveal_moments", [])):
            ep_reveals = context["reveal_moments"][ep_idx]

        for r_idx, reveal in enumerate(ep_reveals):
            if rev_generated >= _MAX_REVELATION_CLIPS:
                break
            img = reveal.get("image_path", "")
            if not img or not Path(img).exists():
                continue
            if reveal.get("video_path") and Path(reveal["video_path"]).exists():
                if _validate_clip(Path(reveal["video_path"])):
                    rev_generated += 1
                    continue
            prompt = _guaranteed_video_prompt(reveal)
            clip_name = f"ep{ep_idx+1}_reveal{r_idx+1}"
            logger.info(f"ep{ep_idx+1} reveal{r_idx+1}: {prompt[:60]}...")
            try:
                vp = generate_video_clip(img, prompt, clip_name)
                context["reveal_moments"][ep_idx][r_idx]["video_path"] = vp
                rev_generated += 1
                time.sleep(3)
            except Exception as e:
                logger.warning(f"ep{ep_idx+1} reveal{r_idx+1} video failed: {e}")
                failed += 1

        # If reveals didn't fill both slots, pull from high-impact scene arcs
        for s_idx, scene in enumerate(episode_scenes):
            if rev_generated >= _MAX_REVELATION_CLIPS:
                break
            arc = scene.get("arc_position", "")
            if arc not in _VIDEO_ARC_POSITIONS:
                continue
            if arc == "cliffhanger":
                continue  # already handled in phase 1
            img = scene.get("image_path", "")
            if not img or not Path(img).exists():
                continue
            if scene.get("video_path") and Path(scene["video_path"]).exists():
                if _validate_clip(Path(scene["video_path"])):
                    rev_generated += 1
                    continue
            prompt = _guaranteed_video_prompt(scene)
            clip_name = f"ep{ep_idx+1}_scene{s_idx+1}_{arc}"
            logger.info(f"ep{ep_idx+1} scene{s_idx+1} ({arc}): {prompt[:60]}...")
            try:
                vp = generate_video_clip(img, prompt, clip_name)
                context["scene_prompts"][ep_idx][s_idx]["video_path"] = vp
                rev_generated += 1
                time.sleep(3)
            except Exception as e:
                logger.warning(f"ep{ep_idx+1} scene{s_idx+1} video failed: {e}")
                failed += 1

        logger.info(
            f"ep{ep_idx+1}: {cliff_generated} cliffhanger clip + {rev_generated} revelation clip(s) "
            f"generated, {failed} failed, rest use Ken Burns"
        )

    return context
