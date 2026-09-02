"""
Agent Image Analyzer — per-image adaptive cinematography grade.

Runs after image generation, before the compositor.
For each generated image (scene, reveal, insert) it reads the actual pixel
data and derives grade parameters that *complement* what's already in the
image instead of fighting it.

Problems solved:
  - FLUX dark images + additional brightness: -0.06 → crushed black smear
  - Already-saturated AI output + saturation: 1.10 boost → garish/blown colour
  - Flat/foggy images + low contrast setting → washed out

Method: luminance histogram + per-channel statistics.
No API calls. Pillow + numpy. ~5ms per image.

Output: adds `adaptive_grade` dict to every entry that has an image_path.
Compositor reads `entry.get("adaptive_grade") or _ARC_GRADE[arc_pos]`.
"""

import numpy as np
from pathlib import Path
from utils.logger import AgentLogger

logger = AgentLogger("ImageAnalyzer")


# ---------------------------------------------------------------------------
# Grade computation
# ---------------------------------------------------------------------------

def _image_stats(image_path: str) -> dict:
    """
    Read the image and return perceptual statistics.
    Returns empty dict on failure — compositor falls back to arc grade.
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0

        # Perceived luminance (Rec. 709 coefficients)
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

        mean_lum = float(np.mean(lum))
        std_lum  = float(np.std(lum))
        p5       = float(np.percentile(lum, 5))
        p95      = float(np.percentile(lum, 95))

        # Saturation: HSV-style (max_channel - min_channel) / max_channel
        max_c    = arr.max(axis=2)
        min_c    = arr.min(axis=2)
        sat      = (max_c - min_c) / np.maximum(max_c, 1e-5)
        mean_sat = float(np.mean(sat))

        return {
            "mean_lum":        mean_lum,
            "std_lum":         std_lum,
            "contrast_spread": p95 - p5,
            "mean_sat":        mean_sat,
        }
    except ImportError:
        logger.warning("Pillow not installed — adaptive grading skipped. pip install Pillow")
        return {}
    except Exception as e:
        logger.warning(f"Could not read {image_path}: {e}")
        return {}


def compute_adaptive_grade(image_path: str, arc_grade: dict) -> dict:
    """
    Derive FFmpeg eq / colorbalance parameters that complement the image.

    Rules
    ─────
    Brightness
      Arc grades always darken (negative values).  A dark image must not be
      darkened further — we scale the darkening proportionally to how much
      headroom the image has above black.
      Scale factor = clamp((mean_lum − 0.15) / 0.35, 0.15, 1.0)
      → image at mean 0.15 → scale 0.15 (barely touch it)
      → image at mean 0.50 → scale 1.0  (full arc darkening)

    Contrast
      Arc grades boost contrast (values 1.10–1.22).  A high-contrast image
      must not be pushed further — it will clip.  A flat image can take more.
      Scale = clamp(1.20 − (contrast_spread − 0.30) / 0.35 * 0.70, 0.40, 1.30)
      → spread 0.65+ (already contrasty) → scale ≤ 0.50 (half the boost)
      → spread 0.30−  (flat)             → scale 1.20  (extra boost)

    Saturation
      AI-generated images often come pre-saturated.  Scale the boost back
      when saturation is already high.  Desaturation (cliffhanger) is kept.
      Boost scale = clamp(1.0 − (mean_sat − 0.20) / 0.30, 0.30, 1.0)

    Gamma
      Gamma < 1 deepens shadows.  Same scaling as brightness so we don't
      crush shadow detail on already-dark images.

    Colorbalance tints (shadow/mid channel shifts) are kept fixed — they are
    arc-appropriate and their magnitude is already very subtle.
    """
    stats = _image_stats(image_path)
    if not stats:
        return {}

    mean_lum        = stats["mean_lum"]
    contrast_spread = stats["contrast_spread"]
    mean_sat        = stats["mean_sat"]

    # ── Brightness ────────────────────────────────────────────────────────────
    # clamp to [0.15, 1.0] so the darkest images still get a tiny touch
    bright_scale = min(1.0, max(0.15, (mean_lum - 0.15) / 0.35))
    brightness   = arc_grade.get("brightness", 0.0) * bright_scale

    # ── Contrast ──────────────────────────────────────────────────────────────
    arc_contrast_extra = arc_grade.get("contrast", 1.0) - 1.0
    if contrast_spread >= 0.65:
        contrast_scale = 0.40
    elif contrast_spread <= 0.30:
        contrast_scale = 1.20
    else:
        contrast_scale = 1.20 - (contrast_spread - 0.30) / 0.35 * 0.80
    contrast = 1.0 + arc_contrast_extra * max(0.40, contrast_scale)

    # ── Saturation ────────────────────────────────────────────────────────────
    arc_sat_extra = arc_grade.get("saturation", 1.0) - 1.0
    if arc_sat_extra > 0:
        # Boosting: pull back when already saturated
        sat_scale = max(0.30, 1.0 - (mean_sat - 0.20) / 0.30)
    else:
        # Reducing (cliffhanger desaturation): always apply in full
        sat_scale = 1.0
    saturation = 1.0 + arc_sat_extra * sat_scale
    saturation = max(0.50, saturation)  # never go below 0.50

    # ── Gamma ─────────────────────────────────────────────────────────────────
    arc_gamma_extra = arc_grade.get("gamma", 1.0) - 1.0
    gamma = 1.0 + arc_gamma_extra * bright_scale
    gamma = max(0.82, gamma)  # floor: don't let dark images go too far

    return {
        "brightness": round(brightness,  4),
        "contrast":   round(contrast,    4),
        "saturation": round(saturation,  4),
        "gamma":      round(gamma,       4),
        # Tonal tints unchanged — arc-appropriate, magnitudes are tiny
        "shadow_r":   arc_grade.get("shadow_r", 0.0),
        "shadow_b":   arc_grade.get("shadow_b", 0.0),
        "mid_r":      arc_grade.get("mid_r",    0.0),
        "mid_b":      arc_grade.get("mid_b",    0.0),
        # Stats kept for debug inspection
        "_mean_lum":        round(mean_lum,        3),
        "_contrast_spread": round(contrast_spread, 3),
        "_mean_sat":        round(mean_sat,         3),
        "_bright_scale":    round(bright_scale,     3),
    }


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

# Arc grade table is duplicated here to avoid circular imports with compositor
_ARC_GRADE = {
    "establish":     {"brightness": -0.02, "contrast": 1.10, "saturation": 1.08, "gamma": 0.93,
                      "shadow_r": +0.010, "shadow_b": -0.005, "mid_r": +0.005, "mid_b": -0.003},
    "build":         {"brightness": -0.03, "contrast": 1.12, "saturation": 1.05, "gamma": 0.91,
                      "shadow_r":  0.000, "shadow_b":  0.000, "mid_r":  0.000, "mid_b":  0.000},
    "investigation": {"brightness": -0.05, "contrast": 1.15, "saturation": 1.02, "gamma": 0.89,
                      "shadow_r": -0.005, "shadow_b": +0.010, "mid_r": -0.003, "mid_b": +0.005},
    "revelation":    {"brightness": -0.04, "contrast": 1.18, "saturation": 1.10, "gamma": 0.88,
                      "shadow_r": -0.010, "shadow_b": +0.015, "mid_r": -0.005, "mid_b": +0.008},
    "cliffhanger":   {"brightness": -0.06, "contrast": 1.22, "saturation": 0.97, "gamma": 0.85,
                      "shadow_r": -0.015, "shadow_b": +0.020, "mid_r": -0.008, "mid_b": +0.010},
}
_ARC_GRADE_DEFAULT = _ARC_GRADE["build"]


def run_image_analyzer(context: dict) -> dict:
    """
    Analyze every generated image and attach adaptive_grade to its entry.
    Runs after agent_5a (image generation), before the compositor.
    """
    logger.info("Analyzing image luminance for adaptive grading...")
    analyzed = 0
    skipped  = 0

    for ep_idx, ep_scenes in enumerate(context.get("scene_prompts", [])):
        for scene in ep_scenes:
            img = scene.get("image_path", "")
            if not img or not Path(img).exists():
                skipped += 1
                continue
            arc = scene.get("arc_position", "build")
            base_grade = _ARC_GRADE.get(arc, _ARC_GRADE_DEFAULT)
            ag = compute_adaptive_grade(img, base_grade)
            if ag:
                scene["adaptive_grade"] = ag
                analyzed += 1
            else:
                skipped += 1

            # Same for insert shots within this scene
            for ins in scene.get("insert_shots", []):
                ins_img = ins.get("image_path", "")
                if ins_img and Path(ins_img).exists():
                    ins_grade = compute_adaptive_grade(ins_img, _ARC_GRADE["cliffhanger"])
                    if ins_grade:
                        ins["adaptive_grade"] = ins_grade

    for ep_idx, ep_reveals in enumerate(context.get("reveal_moments", [])):
        for reveal in ep_reveals:
            img = reveal.get("image_path", "")
            if not img or not Path(img).exists():
                skipped += 1
                continue
            # Reveals always use revelation grade as base
            base_grade = _ARC_GRADE["revelation"]
            ag = compute_adaptive_grade(img, base_grade)
            if ag:
                reveal["adaptive_grade"] = ag
                analyzed += 1
            else:
                skipped += 1

    logger.info(f"Image analysis complete: {analyzed} adapted, {skipped} skipped (no image or Pillow)")
    return context
