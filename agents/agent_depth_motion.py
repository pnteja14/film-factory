"""
Agent Depth Motion — generates depth maps for scene/reveal/insert images.

Uses Depth-Anything-V2-Small (transformers pipeline) to estimate per-pixel depth.
Falls back to luminance-based heuristic if the model is unavailable.

Depth maps are stored alongside scene images and used by the compositor
to apply 2-layer parallax (near pixels lead, far pixels lag) without
semantic segmentation — architectural elements at the same depth stay together.
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import AgentLogger

load_dotenv()
logger = AgentLogger("DepthMotion")

DEPTH_DIR = Path("outputs/depth_maps")
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

_depth_pipe = None
_depth_available = None


def _load_depth_model():
    global _depth_pipe, _depth_available
    if _depth_available is not None:
        return _depth_available
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading Depth-Anything-V2-Small...")
        _depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        _depth_available = True
        logger.info("Depth-Anything-V2-Small loaded.")
    except Exception as e:
        logger.warning(f"Depth model unavailable ({e}) — using luminance fallback.")
        _depth_available = False
    return _depth_available


def _depth_from_model(image_path: str) -> np.ndarray:
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    result = _depth_pipe(img)
    depth = np.array(result["depth"])
    return depth


def _depth_from_luminance(image_path: str) -> np.ndarray:
    """Luminance heuristic: brighter regions treated as closer (foreground)."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth array to 0-255 uint8."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros(depth.shape, dtype=np.uint8)
    normalized = (depth - d_min) / (d_max - d_min) * 255.0
    return normalized.astype(np.uint8)


def generate_depth_map(image_path: str, slug: str) -> str | None:
    """
    Generate and save a depth map PNG for the given image.
    Returns the depth map path, or None on failure.
    """
    out_path = str(DEPTH_DIR / f"{slug}_depth.png")

    if Path(out_path).exists():
        return out_path

    if not Path(image_path).exists():
        logger.warning(f"Image missing for depth: {image_path}")
        return None

    try:
        from PIL import Image as PILImage

        use_model = _load_depth_model()
        if use_model:
            depth_arr = _depth_from_model(image_path)
        else:
            depth_arr = _depth_from_luminance(image_path)

        depth_uint8 = _normalize_depth(depth_arr.astype(np.float32))
        depth_img = PILImage.fromarray(depth_uint8, mode="L")
        depth_img.save(out_path)
        return out_path

    except Exception as e:
        logger.warning(f"Depth map failed for {image_path}: {e}")
        return None


def _slug(path: str) -> str:
    return Path(path).stem


def run_agent_depth_motion(context: dict) -> dict:
    """
    Agent Depth Motion: generate depth maps for all scene, reveal, and insert images.

    Adds depth_map_path to each scene/reveal/insert shot dict.
    Sets context["depth_motion_available"] = True/False.
    """
    logger.info("Generating depth maps...")

    available = _load_depth_model()
    context["depth_motion_available"] = available

    if not available:
        logger.warning("Depth model not available — depth_motion_available=False. "
                       "Compositor will use Ken Burns for all scenes.")

    total = 0
    done  = 0

    scene_prompts_all  = context.get("scene_prompts",  [])
    reveal_moments_all = context.get("reveal_moments", [])

    # Count total images
    for ep_scenes in scene_prompts_all:
        for scene in ep_scenes:
            if scene.get("image_path"):
                total += 1
            for ins in scene.get("insert_shots", []):
                if ins.get("image_path"):
                    total += 1
    for ep_reveals in reveal_moments_all:
        for rev in ep_reveals:
            if rev.get("image_path"):
                total += 1

    def _process(item: dict, label: str):
        nonlocal done
        image_path = item.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            return
        slug = _slug(image_path)
        depth_path = generate_depth_map(image_path, slug)
        if depth_path:
            item["depth_map_path"] = depth_path
            done += 1
            logger.info(f"  [{done}/{total}] {label}: {Path(image_path).name}")

    for ep_idx, ep_scenes in enumerate(scene_prompts_all):
        for scene in ep_scenes:
            _process(scene, f"ep{ep_idx+1} scene {scene.get('scene_number','?')}")
            for ins in scene.get("insert_shots", []):
                _process(ins, f"ep{ep_idx+1} scene {scene.get('scene_number','?')} insert")

    for ep_idx, ep_reveals in enumerate(reveal_moments_all):
        for r_idx, rev in enumerate(ep_reveals):
            _process(rev, f"ep{ep_idx+1} reveal {r_idx+1}")

    logger.info(f"Depth maps complete: {done}/{total} generated.")
    return context
