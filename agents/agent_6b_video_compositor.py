"""
Agent 6b — Video Compositor (FFmpeg filter_complex architecture)

Architecture: every EDL entry renders to a short MP4 segment via a single
ffmpeg subprocess call using a fully-built filter_complex graph.
Segments are concatenated with the ffmpeg concat demuxer.
Audio is mixed in Python (numpy) — identical to the original mix_audio_tracks.

Target: under 3 minutes total render time per episode.
"""

import os
import math
import hashlib
import subprocess
import wave
import numpy as np
from pathlib import Path
from utils.logger import AgentLogger
from utils.errors import AgentException

logger = AgentLogger("VideoCompositor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_WIDTH   = 1920
OUTPUT_HEIGHT  = 1080
OUTPUT_FPS     = 24
OUTPUT_BITRATE = "6M"
SAMPLE_RATE    = 44100
LETTERBOX_BAR_H   = 80    # subtle 2.1:1 cinematic crop
OVERSCAN          = 1.10  # 10% overscan canvas for zoompan without edge warp
MIN_VIDEO_CLIP_BYTES = 300_000  # reject Wan clips below this — corrupt/static

FONT_PATH_WIN = "C:/Windows/Fonts/arialbd.ttf"
FONT_PATH_LIN = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
_FONT_FALLBACKS_LIN = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
]

OW = OUTPUT_WIDTH
OH = OUTPUT_HEIGHT

# ---------------------------------------------------------------------------
# Per-arc cinematic grade parameters
# Philosophy: enhance whatever colors are already in the image, darken shadows,
# add mood with a whisper of tonal tint — never impose a uniform palette.
#
# eq:          brightness → slightly darker overall
#              contrast   → more visual punch per arc intensity
#              saturation → ≥ 1.0 so source colors pop rather than wash out
#              gamma      → < 1.0 deepens shadows without crushing blacks
# colorbalance (shadow/mid channels only — tiny values):
#              establish    → warm amber hint (r+, b−) enhances golden interiors
#              build        → neutral (no tint)
#              investigation→ cool hint (r−, b+) enhances cold/grey environments
#              revelation   → colder/sharper (stronger cool push)
#              cliffhanger  → coldest + slight desaturation for maximum tension
# ---------------------------------------------------------------------------
_ARC_GRADE = {
    "establish": {
        "brightness": -0.02, "contrast": 1.10, "saturation": 1.08, "gamma": 0.93,
        "shadow_r": +0.010, "shadow_b": -0.005, "mid_r": +0.005, "mid_b": -0.003,
    },
    "build": {
        "brightness": -0.03, "contrast": 1.12, "saturation": 1.05, "gamma": 0.91,
        "shadow_r":  0.000, "shadow_b":  0.000, "mid_r":  0.000, "mid_b":  0.000,
    },
    "investigation": {
        "brightness": -0.05, "contrast": 1.15, "saturation": 1.02, "gamma": 0.89,
        "shadow_r": -0.005, "shadow_b": +0.010, "mid_r": -0.003, "mid_b": +0.005,
    },
    "revelation": {
        "brightness": -0.04, "contrast": 1.18, "saturation": 1.10, "gamma": 0.88,
        "shadow_r": -0.010, "shadow_b": +0.015, "mid_r": -0.005, "mid_b": +0.008,
    },
    "cliffhanger": {
        "brightness": -0.06, "contrast": 1.22, "saturation": 0.97, "gamma": 0.85,
        "shadow_r": -0.015, "shadow_b": +0.020, "mid_r": -0.008, "mid_b": +0.010,
    },
}
_ARC_GRADE_DEFAULT = _ARC_GRADE["build"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_font_path() -> str:
    """Returns the first available bold font path, or empty string."""
    if Path(FONT_PATH_WIN).exists():
        return FONT_PATH_WIN
    for fp in _FONT_FALLBACKS_LIN:
        if Path(fp).exists():
            return fp
    return ""


# ---------------------------------------------------------------------------
# Zoompan expression builder
# ---------------------------------------------------------------------------

def build_zoompan_expr(
    zoom_dir: str,
    zoom_rate: float,
    duration_s: float,
    speed_factor: float = 1.0,
    movement: str = "push_center",
) -> str:
    """
    Returns a complete ffmpeg zoompan filter string.

    movement selects one of 8 named Ken Burns patterns:
      push_center   — zoom in toward frame center (default)
      push_left     — zoom in, drift rightward (camera moves left)
      push_right    — zoom in, drift leftward
      rise          — zoom in, drift upward (crane up)
      diagonal_in   — zoom in toward top-left anchor
      diagonal_out  — zoom out from bottom-right anchor
      pull_back     — slow zoom out (establish/breathe)
      breathe       — gentle sine in/out (cliffhanger freeze)

    Zoom capped at 1.35 to avoid edge warp on 10% overscan canvas.
    d=1 ensures one output frame per input frame.
    """
    total_frames  = max(1, int(duration_s * OUTPUT_FPS))
    effective_rate = zoom_rate * speed_factor * duration_s
    r = effective_rate

    # z floor at 1.001 — avoids the z=1.0 degenerate case in some FFmpeg builds
    # where exact z=1.0 is handled differently from z=1.001, producing a visible
    # pop on the first frame.  The 0.001 difference is imperceptible visually.
    #
    # x/y clamped to [0, iw-iw/zoom] and [0, ih-ih/zoom] — the legal viewport
    # range.  Without clamping, large pans on short clips can drift past the
    # overscan canvas edge, sampling uninitialised or replicated pixels.

    if movement == "push_left":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*on/{total_frames}))"
        x_expr = f"max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)+iw*0.004*on/{total_frames}))"
        y_expr = "max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)))"

    elif movement == "push_right":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*on/{total_frames}))"
        x_expr = f"max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)-iw*0.004*on/{total_frames}))"
        y_expr = "max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)))"

    elif movement == "rise":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*on/{total_frames}))"
        x_expr = "max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)))"
        y_expr = f"max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)-ih*0.003*on/{total_frames}))"

    elif movement == "diagonal_in":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*on/{total_frames}))"
        x_expr = f"max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)-iw*0.003*on/{total_frames}))"
        y_expr = f"max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)-ih*0.002*on/{total_frames}))"

    elif movement == "diagonal_out":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*(1-on/{total_frames})))"
        x_expr = f"max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)+iw*0.003*on/{total_frames}))"
        y_expr = f"max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)+ih*0.002*on/{total_frames}))"

    elif movement == "pull_back" or zoom_dir == "out":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*(1-on/{total_frames})))"
        x_expr = "max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)))"
        y_expr = "max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)))"

    elif movement == "breathe" or zoom_dir == "none":
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*sin(PI*on/{total_frames})))"
        x_expr = "max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)))"
        y_expr = "max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)))"

    else:  # push_center (default)
        z_expr = f"max(1.001,min(1.35,1.0+{r:.6f}*on/{total_frames}))"
        x_expr = "max(0,min(iw-iw/zoom,iw/2-(iw/zoom/2)))"
        y_expr = "max(0,min(ih-ih/zoom,ih/2-(ih/zoom/2)))"

    return (
        f"zoompan=z='{z_expr}':"
        f"x='{x_expr}':"
        f"y='{y_expr}':"
        f"d=1:s={OW}x{OH}:fps={OUTPUT_FPS}"
    )


# Movement pattern assigned by arc_position + scene index parity
# Alternates axis so adjacent scenes never use the same direction
_ARC_MOVEMENTS = {
    "establish":     ["pull_back",    "diagonal_out"],
    "build":         ["push_left",    "push_right"],
    "investigation": ["push_center",  "diagonal_in"],
    "revelation":    ["rise",         "push_center"],
    "cliffhanger":   ["breathe",      "breathe"],
}

def get_movement_for_entry(entry: dict, entry_index: int) -> str:
    arc  = entry.get("arc_position", "build")
    opts = _ARC_MOVEMENTS.get(arc, ["push_center", "push_left"])
    return opts[entry_index % len(opts)]


# ---------------------------------------------------------------------------
# Filter complex builder
# ---------------------------------------------------------------------------

def build_filter_complex(
    entry: dict,
    duration_s: float,
    features: dict,
) -> tuple:
    """
    Build and return (input_args: list[str], filter_complex: str).

    input_args are the ffmpeg input flags (-loop/-t/-i triples).
    filter_complex is the full filter chain string ending in label [out].

    See module docstring for chain step descriptions.
    """
    zoom_dir   = entry.get("zoom_direction", "in")
    zoom_rate  = float(entry.get("zoom_rate", 0.02))
    image_path = entry.get("image_path", "")
    arc_pos    = entry.get("arc_position", "build")
    entry_type = entry.get("type", "scene")

    overscan_w = int(OW * OVERSCAN)
    overscan_h = int(OH * OVERSCAN)

    input_args = []
    chains = []  # list of filter chain strings (joined by ";")

    # ------------------------------------------------------------------
    # STEP 1 — Motion source selection
    #   Path V: AI-generated video clip (Wan2.1-I2V) — scale/loop to fit, no zoompan
    #   Path K: Ken Burns with 8-movement vocabulary (default)
    # ------------------------------------------------------------------
    video_clip  = entry.get("video_path", "")
    _clip_path  = Path(video_clip) if video_clip else None
    _clip_valid = (
        _clip_path is not None
        and _clip_path.exists()
        and _clip_path.stat().st_size >= MIN_VIDEO_CLIP_BYTES
    )
    if _clip_valid:
        # Path V — use Wan-generated video clip directly.
        # On Colab, Real-ESRGAN has already upscaled the clip to 1920x1080, so the
        # scale step is a pass-through (force_original_aspect_ratio=increase + crop
        # is a no-op when src == dst resolution).  On Windows (local test), the clip
        # is 832x480 and the scale fills the frame with center-crop — no letterbox bars.
        # No unsharp needed: ESRGAN handles sharpening at the frame level before this point.
        # stream_loop -1 repeats the clip to fill longer scene durations; the fade-in/out
        # added after the letterbox step masks the loop seam.
        clip_p = video_clip.replace("\\", "/")
        input_args += ["-stream_loop", "-1", "-t", str(duration_s), "-an", "-i", clip_p]
        chains.append(
            f"[0:v]scale={OW}:{OH}:force_original_aspect_ratio=increase,"
            f"crop={OW}:{OH},setsar=1[comp]"
        )
    else:
        # Path K — Ken Burns with 8-movement vocabulary
        img_p = image_path.replace("\\", "/")
        input_args += ["-loop", "1", "-t", str(duration_s), "-i", img_p]
        # lanczos gives sharper upscale from 1280x720 → overscan canvas; setsar prevents
        # SAR drift when the source image has non-square pixels in its metadata
        chains.append(
            f"[0:v]scale={overscan_w}:{overscan_h}:flags=lanczos,setsar=1[img_s]"
        )
        movement = features.get("movement", "push_center")
        zp = build_zoompan_expr(zoom_dir, zoom_rate, duration_s, speed_factor=1.0, movement=movement)
        # setsar=1 after zoompan ensures SAR is clean for downstream filters
        chains.append(f"[img_s]{zp},setsar=1[comp]")

    # ------------------------------------------------------------------
    # STEP 2 — Image-adaptive grade: parameters computed from actual pixel
    # stats of this specific image, then blended with arc tonal direction.
    # Dark images get less additional darkening; flat images get more contrast.
    # Insert / photo-evidence entries keep the fixed clinical override.
    # ------------------------------------------------------------------
    if entry_type == "insert" or entry.get("photo_evidence_treatment"):
        _g = _ARC_GRADE["cliffhanger"]
    else:
        # Prefer per-image adaptive grade (set by agent_image_analyzer)
        _g = entry.get("adaptive_grade") or _ARC_GRADE.get(arc_pos, _ARC_GRADE_DEFAULT)
    _eq  = (f"eq=brightness={_g['brightness']}:contrast={_g['contrast']}"
            f":saturation={_g['saturation']}:gamma={_g['gamma']}")
    _cb  = (f"colorbalance=rs={_g['shadow_r']}:gs=0:bs={_g['shadow_b']}"
            f":rm={_g['mid_r']}:gm=0:bm={_g['mid_b']}:rh=0:gh=0:bh=0")
    chains.append(f"[comp]{_eq}[_eq1];[_eq1]{_cb}[grade]")

    # ------------------------------------------------------------------
    # STEP 3 — Halation (highlight bloom)
    # ------------------------------------------------------------------
    chains.append(
        "[grade]split=2[main_h][bloom_src];"
        "[bloom_src]lutyuv=y='if(gt(val,210),val,0)':u='128':v='128',gblur=sigma=16[bloom];"
        "[main_h][bloom]blend=c0_mode=screen:c0_opacity=0.20:c1_expr='A':c2_expr='A'[ha]"
    )

    # ------------------------------------------------------------------
    # STEP 4 — Film grain (analog texture baked per-segment before encode)
    # Lift the black floor to ~5% before adding grain so near-black images
    # show grain as cinematic texture rather than video-noise artifact.
    # The 12/255 lift is imperceptible on well-exposed frames.
    # ------------------------------------------------------------------
    chains.append("[ha]curves=all='0/0.047 1/1'[ha_lift]")
    chains.append("[ha_lift]noise=alls=6:allf=t[gn]")

    # ------------------------------------------------------------------
    # STEP 5 — Lens vignette (moderate corner falloff, evaluated once)
    # ------------------------------------------------------------------
    chains.append("[gn]vignette=angle=PI/4.5:eval=init[vi]")

    # ------------------------------------------------------------------
    # STEP 6 — Entry-type-specific effects
    # Track cur_eff through steps 6-10 so each FFmpeg label is unique.
    # ------------------------------------------------------------------
    photo_evidence = entry.get("photo_evidence_treatment", False)
    cur_eff = "vi"  # starts at vignette output label

    if photo_evidence:
        # Cream-border photo evidence: desaturate + pad + tilt
        path_hash = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(path_hash & 0x7FFFFFFF)
        tilt_rad = float(rng.uniform(-0.035, 0.035))
        pad_w = OW + 56
        pad_h = OH + 56
        # trunc(x/2)*2 forces even pixel dimensions (required for yuv420p)
        chains.append(
            f"[{cur_eff}]eq=saturation=0.15[desat_vi];"
            f"[desat_vi]pad={pad_w}:{pad_h}:28:28:color=0xEBE8E1,"
            f"rotate=angle={tilt_rad:.6f}:fillcolor=0x0C0C10,"
            f"scale=trunc(iw*1.04/2)*2:trunc(ih*1.04/2)*2,"
            f"crop={OW}:{OH}"
            f"[s6]"
        )
        cur_eff = "s6"

    elif entry_type == "reveal":
        # VHS glitch burst on reveal cut (noise filter alls doesn't support expressions)
        chains.append(f"[{cur_eff}]noise=alls=20:allf=t[s6]")
        cur_eff = "s6"

    elif entry_type == "cliffhanger" and entry.get("desaturate"):
        # Desaturation ramp + dutch angle + fade-to-black in final 1.5s.
        # Fade starts at max(0, duration - 1.5) so the image goes black before
        # the TBC card cuts in — mimics the One Piece "silence before the reveal".
        fade_start = max(0.0, duration_s - 1.5)
        chains.append(
            f"[{cur_eff}]hue=s='max(0,1-t/{duration_s:.4f})'[s6_ds];"
            f"[s6_ds]rotate=angle=0.023:fillcolor=black,"
            f"scale=trunc(iw*1.04/2)*2:trunc(ih*1.04/2)*2,crop={OW}:{OH}[s6_r];"
            f"[s6_r]fade=type=out:start_time={fade_start:.3f}:duration=1.5[s6]"
        )
        cur_eff = "s6"

    elif arc_pos in ("revelation", "cliffhanger"):
        # Subtle dutch angle; over-scale then crop to hide black corners
        chains.append(
            f"[{cur_eff}]rotate=angle=0.012:fillcolor=black,"
            f"scale=trunc(iw*1.04/2)*2:trunc(ih*1.04/2)*2,crop={OW}:{OH}[s6]"
        )
        cur_eff = "s6"

    else:
        chains.append(f"[{cur_eff}]null[s6]")
        cur_eff = "s6"

    # ------------------------------------------------------------------
    # STEP 7 — Lens flare (revelation only, not cliffhanger which desaturates)
    # sigma=48:sigmaV=1 → horizontal streaks; sigma=1:sigmaV=45 was vertical
    # ------------------------------------------------------------------
    if features.get("show_lens_flare") and not photo_evidence and entry_type != "cliffhanger":
        chains.append(
            f"[{cur_eff}]split=2[lf_base][lf_src];"
            "[lf_src]lutyuv=y='if(gt(val,225),val,0)':u='128':v='128',gblur=sigma=48:sigmaV=1[lf_streak];"
            "[lf_base][lf_streak]blend=c0_mode=screen:c0_opacity=0.18:c1_expr='A':c2_expr='A'[s7]"
        )
        cur_eff = "s7"

    # ------------------------------------------------------------------
    # STEP 8 — Fog (establish / build arcs only)
    # ------------------------------------------------------------------
    if features.get("show_fog") and arc_pos in ("establish", "build"):
        fog_y = int(OH * 0.82)
        fog_h = OH - fog_y
        # Neutral very-dark grey at 6% opacity — avoids teal colour cast on dark scenes
        chains.append(
            f"[{cur_eff}]drawbox=x=0:y={fog_y}:w={OW}:h={fog_h}:"
            "color=0x181818@0.06:t=fill[s8]"
        )
        cur_eff = "s8"

    # ------------------------------------------------------------------
    # STEP 10 — Letterbox
    # ------------------------------------------------------------------
    chains.append(
        f"[{cur_eff}]drawbox=x=0:y=0:w=iw:h={LETTERBOX_BAR_H}:color=black@1:t=fill,"
        f"drawbox=x=0:y=ih-{LETTERBOX_BAR_H}:w=iw:h={LETTERBOX_BAR_H}:color=black@1:t=fill"
        f"[lb]"
    )

    # ------------------------------------------------------------------
    # STEP 10b — Wan clip fade-in / fade-out (Path V only, non-cliffhanger)
    # Bridges the quality gap between Ken Burns (crisp FLUX) and Wan (upscaled 480p).
    # 0.25 s fades: the cut FROM a Ken Burns segment hits black, then the Wan clip
    # fades in — the viewer reads it as an intentional chapter break, not a glitch.
    # Cliffhanger entries are excluded because they have their own 1.5s fade-out in Step 6.
    # ------------------------------------------------------------------
    if _clip_valid and entry_type != "cliffhanger":
        fade_d = min(0.25, duration_s / 4)
        fade_out_st = max(0.0, duration_s - fade_d)
        chains.append(
            f"[lb]fade=t=in:st=0:d={fade_d:.3f},"
            f"fade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}"
            f"[lb_fade]"
        )
        cur_label = "lb_fade"
    else:
        cur_label = "lb"

    # ------------------------------------------------------------------
    # STEP 11 — Subtitle sync (ASS burn-in)
    # ------------------------------------------------------------------
    subtitle_path = features.get("subtitles_path")
    font = get_font_path()

    if subtitle_path and Path(subtitle_path).exists() and font:
        # Normalise path separators for FFmpeg on Windows
        sub_p = subtitle_path.replace("\\", "/").replace(":", "\\:")
        chains.append(
            f"[{cur_label}]ass='{sub_p}'[sub_out]"
        )
        cur_label = "sub_out"

    # ------------------------------------------------------------------
    # STEP 12 — Timestamp overlay (scene label, bottom-left, unobtrusive)
    # ------------------------------------------------------------------
    timestamp_text = features.get("timestamp_text")
    if timestamp_text and font:
        # Escape special characters for drawtext
        ts_safe = (
            timestamp_text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
        )
        font_safe = font.replace("\\", "/").replace(":", "\\:")
        chains.append(
            f"[{cur_label}]drawtext="
            f"fontfile='{font_safe}':"
            f"text='{ts_safe}':"
            f"fontsize=26:"
            f"fontcolor=white@0.60:"
            f"x=60:y=h-th-200:"
            f"shadowcolor=black@0.8:shadowx=1:shadowy=1"
            f"[ts_out]"
        )
        cur_label = "ts_out"

    # ------------------------------------------------------------------
    # STEP 13 — Title card text (entry 0 only)
    # ------------------------------------------------------------------
    title_text = features.get("title_text")
    if features.get("entry_idx", -1) == 0 and title_text and font:
        title_safe = (
            title_text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
        )
        font_safe = font.replace("\\", "/").replace(":", "\\:")
        chains.append(
            f"[{cur_label}]drawtext="
            f"fontfile='{font_safe}':"
            f"text='{title_safe}':"
            f"fontsize=58:"
            f"fontcolor=white:"
            f"alpha='min(t/0.8,1)':"
            f"x=(w-tw)/2:y=(h-th-40)/2:"
            f"shadowcolor=black@0.9:shadowx=2:shadowy=2:"
            f"enable='lt(t,3.0)'"
            f"[out]"
        )
    else:
        chains.append(f"[{cur_label}]null[out]")

    # ------------------------------------------------------------------
    # Join all chains into the filter_complex string
    # ------------------------------------------------------------------
    # Join all chains — individual steps may already contain semicolons.
    filter_complex = ";".join(chains)

    return input_args, filter_complex


# ---------------------------------------------------------------------------
# Segment renderer
# ---------------------------------------------------------------------------


def render_entry_segment(
    entry: dict,
    seg_idx: int,
    ep_num: int,
    features: dict,
    tmp_dir: str,
) -> str:
    """
    Render one EDL entry to a short MP4 segment.
    Returns the segment path on success, or None on failure.
    """
    seg_path = f"{tmp_dir}/ep{ep_num}_seg{seg_idx:04d}.mp4"
    duration_s = max(0.1, entry.get("duration_ms", 1000) / 1000.0)

    input_args, filter_complex = build_filter_complex(entry, duration_s, features)

    cmd = (
        ["ffmpeg", "-y"]
        + input_args
        + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-r", str(OUTPUT_FPS),
            "-t", str(duration_s),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-b:v", OUTPUT_BITRATE,
            "-avoid_negative_ts", "make_zero",
            seg_path,
        ]
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="ignore")
            logger.error(
                f"FFmpeg failed for segment {seg_idx} (ep {ep_num}):\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR: {err[-800:]}"
            )
            return None
        return seg_path
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for segment {seg_idx} (ep {ep_num})")
        return None
    except Exception as e:
        logger.error(f"render_entry_segment exception seg {seg_idx}: {e}")
        return None


# ---------------------------------------------------------------------------
# Segment concatenation
# ---------------------------------------------------------------------------

def concat_segments(seg_paths: list, ep_num: int, tmp_dir: str) -> str:
    """
    Concatenate segment MP4 files into a single video using the concat demuxer.
    Returns the path to the concatenated video.
    """
    # Filter out None / missing paths
    valid = [p for p in seg_paths if p and Path(p).exists()]
    if not valid:
        raise AgentException(f"Episode {ep_num}: no valid segments to concatenate")

    concat_file = f"{tmp_dir}/ep{ep_num}_concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in valid:
            # Use forward slashes; ffmpeg concat demuxer needs absolute paths
            abs_p = str(Path(p).resolve()).replace("\\", "/")
            f.write(f"file '{abs_p}'\n")

    video_path = f"{tmp_dir}/ep{ep_num}_video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise AgentException(f"Concat failed: {err[-500:]}")

    logger.info(f"Concatenated {len(valid)} segments -> {video_path}")
    return video_path


# ---------------------------------------------------------------------------
# Post-concat subtitle burn
# ---------------------------------------------------------------------------

def burn_subtitles_on_video(video_path: str, subtitle_path: str, output_path: str) -> str:
    """
    Burn ASS subtitles onto a video file and return output_path.
    Falls back to returning video_path unchanged on failure.
    Applied post-concat so timestamps are absolute across the full episode.
    """
    font = get_font_path()
    if not font or not Path(subtitle_path).exists():
        logger.warning("Subtitle burn skipped: no font or missing subtitle file")
        return video_path
    sub_p = subtitle_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"ass='{sub_p}'",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-b:v", OUTPUT_BITRATE,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        logger.warning(f"Subtitle burn failed: {err[-400:]}")
        return video_path
    logger.info(f"Subtitles burned: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Post-concat shadow seal
# ---------------------------------------------------------------------------
# The per-segment chain already applies arc-adaptive grade, grain, and vignette.
# This pass applies one final global shadow crush to unify the concatenated cut
# without imposing any colour cast.


def apply_shadow_seal(video_path: str, sealed_path: str) -> str:
    """
    Minimal post-concat pass: tiny global contrast lift + shadow crush.
    No colour shift, no LUT, no grain (already baked per-segment).
    Non-fatal — returns original video_path on failure.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "eq=contrast=1.04:gamma=0.96",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-b:v", OUTPUT_BITRATE,
        sealed_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        logger.warning("Shadow seal skipped (non-fatal)")
        return video_path
    logger.info(f"Shadow seal applied: {sealed_path}")
    return sealed_path


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------

def ms_to_ass(ms: int) -> str:
    """Convert milliseconds to ASS timestamp format H:MM:SS.cc"""
    total_cs = ms // 10
    cc = total_cs % 100
    total_s  = total_cs // 100
    secs = total_s % 60
    total_m  = total_s // 60
    mins = total_m % 60
    hrs  = total_m // 60
    return f"{hrs}:{mins:02d}:{secs:02d}.{cc:02d}"


def _retime_words_from_chunks(word_timestamps: list, structured_chunks: list) -> list:
    """
    Replace Whisper word text with original chunk text, preserving Whisper timing.

    Groups Whisper words by chunk_index to get each chunk's time span, then
    distributes the original text words evenly across that span.  This corrects
    Whisper mis-transcriptions (e.g. "weeding" → "waiting") while keeping the
    timing accurate.
    """
    from collections import defaultdict

    if not structured_chunks or not word_timestamps:
        return word_timestamps

    chunk_whisper_words: dict = defaultdict(list)
    for w in word_timestamps:
        ci = w.get("chunk_index", 0)
        chunk_whisper_words[ci].append(w)

    retimed = []
    for ci, chunk in enumerate(structured_chunks):
        whisper_in_chunk = chunk_whisper_words.get(ci, [])
        if not whisper_in_chunk:
            continue

        chunk_start_ms = min(w["start_ms"] for w in whisper_in_chunk)
        chunk_end_ms = max(w["end_ms"] for w in whisper_in_chunk)
        chunk_dur_ms = max(chunk_end_ms - chunk_start_ms, 100)

        orig_text = chunk.get("text", "").replace("\n", " ")
        orig_words = [w.strip() for w in orig_text.split() if w.strip()]
        if not orig_words:
            continue

        n = len(orig_words)
        for j, word in enumerate(orig_words):
            w_start = chunk_start_ms + int(j * chunk_dur_ms / n)
            w_end = chunk_start_ms + int((j + 1) * chunk_dur_ms / n)
            retimed.append({
                "word": word,
                "start_ms": w_start,
                "end_ms": w_end,
                "chunk_index": ci,
            })

    return retimed if retimed else word_timestamps


def generate_ass_subtitles(
    word_timestamps: list,
    output_path: str,
    offset_ms: int = 0,
    structured_chunks: list = None,
) -> str:
    """
    Groups word timestamps into subtitle lines (up to MAX_WORDS words, max 4 s).
    Writes an ASS file and returns the output_path.

    word_timestamps: list of dicts with keys 'word', 'start_ms', 'end_ms'
    offset_ms: shift all timestamps forward (e.g. 3000 to account for a title card)
    structured_chunks: original narration chunks; when provided, replaces Whisper
      word text with original text while preserving Whisper timing — fixes
      transcription errors without losing sync.
    """
    if structured_chunks:
        word_timestamps = _retime_words_from_chunks(word_timestamps, structured_chunks)
    header = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,46,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,0,0,0,0,100,100,0,0,3,2,1,2,40,40,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines = []

    # Group word timestamps into lines
    if not word_timestamps:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
        return output_path

    # Normalise word timestamp dicts — accept various key conventions
    def _get_ms(wt: dict, key_start: bool) -> int:
        if key_start:
            return int(wt.get("start_ms", wt.get("start", 0)))
        return int(wt.get("end_ms", wt.get("end", 0)))

    words = [w for w in word_timestamps if w.get("word", "").strip()]

    MAX_WORDS = 4
    MAX_DURATION_MS = 3000

    group_start_idx = 0
    while group_start_idx < len(words):
        group = []
        line_start_ms = _get_ms(words[group_start_idx], key_start=True)
        line_end_ms   = line_start_ms

        for i in range(group_start_idx, len(words)):
            w = words[i]
            w_end = _get_ms(w, key_start=False)
            w_start = _get_ms(w, key_start=True)
            projected_duration = w_end - line_start_ms

            if (
                len(group) >= MAX_WORDS
                or (len(group) > 0 and projected_duration > MAX_DURATION_MS)
            ):
                break

            group.append(w.get("word", "").strip())
            line_end_ms = w_end
            group_start_idx = i + 1

        if not group:
            group_start_idx += 1
            continue

        text = " ".join(group)
        if text:
            text = text[0].upper() + text[1:]
        start_str = ms_to_ass(line_start_ms + offset_ms)
        end_str   = ms_to_ass(max(line_end_ms, line_start_ms + 100) + offset_ms)

        dialogue_lines.append(
            f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in dialogue_lines:
            f.write(line + "\n")

    logger.info(f"Subtitles written: {output_path} ({len(dialogue_lines)} lines)")
    return output_path


def _subtitle_groups_to_ass(
    subtitle_groups: list,
    output_path: str,
    offset_ms: int = 0,
) -> str:
    """
    Write an ASS subtitle file from event stream subtitle_groups.
    Fires at exact WhisperX word start_ms — no estimation.
    """
    header = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,46,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,0,0,0,0,100,100,0,0,3,2,1,2,40,40,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip()]
    for g in subtitle_groups:
        text = g.get("text", "").strip()
        if text:
            text = text[0].upper() + text[1:]
        if not text:
            continue
        start = ms_to_ass(max(0, g["start_ms"] + offset_ms))
        end   = ms_to_ass(max(0, g["end_ms"]   + offset_ms))
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Event-stream subtitles written: {output_path} ({len(lines) - 1} groups)")
    return output_path


# ---------------------------------------------------------------------------
# Title card generator
# ---------------------------------------------------------------------------

def generate_title_segment(
    episode_number: int,
    story_title: str,
    ep_num_int: int,
    tmp_dir: str,
    hook_image_path: str = None,
) -> str:
    """
    Generate a 4-second title card.
    If hook_image_path is provided (the episode's strongest image), it is used
    as the background with a dark overlay — giving viewers a visual hook in
    the first 5 seconds instead of pure black silence.
    """
    font = get_font_path()
    title_path = f"{tmp_dir}/ep{ep_num_int}_title.mp4"

    font_safe = font.replace("\\", "/").replace(":", "\\:") if font else ""
    show_name_safe = story_title.upper().replace("'", "\\'").replace(":", "\\:")
    ep_text_safe   = f"Episode {ep_num_int}".replace("'", "\\'").replace(":", "\\:")

    text_vf = ""
    if font_safe:
        text_vf = (
            f",drawtext=fontfile='{font_safe}':"
            f"text='{show_name_safe}':"
            f"fontsize=72:fontcolor=white:"
            f"alpha='min(t/0.8,1)':"
            f"x=(w-tw)/2:y=(h-th)/2-60:"
            f"shadowcolor=black@0.95:shadowx=3:shadowy=3,"
            f"drawtext=fontfile='{font_safe}':"
            f"text='{ep_text_safe}':"
            f"fontsize=38:fontcolor=white@0.85:"
            f"alpha='min(max(t-0.5,0)/0.8,1)':"
            f"x=(w-tw)/2:y=(h-th)/2+30"
        )

    use_hook = hook_image_path and Path(hook_image_path).exists()

    if use_hook:
        hook_p = hook_image_path.replace("\\", "/")
        # Slow Ken Burns push on the hook image + dark overlay + text
        vf = (
            f"scale={int(OW*1.05)}:{int(OH*1.05)},setsar=1,"
            f"zoompan=z='min(1.08,1.0+0.001*on)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={OW}x{OH}:fps={OUTPUT_FPS},"
            f"colorchannelmixer=rr=0.55:gg=0.55:bb=0.6,"  # cool dark overlay
            f"fade=t=in:st=0:d=0.6"
            f"{text_vf}"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", "4.0", "-i", hook_p,
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-t", "4.0",
            title_path,
        ]
    else:
        vf = f"fade=t=in:st=0:d=0.4{text_vf}"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={OW}x{OH}:r={OUTPUT_FPS}:d=4.0",
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            title_path,
        ]

    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise AgentException(f"Title card generation failed: {err[-300:]}")

    logger.info(f"Title card: {title_path}")
    return title_path


def generate_tbc_segment(story_title: str, episode_number: int, tmp_dir: str) -> str:
    """
    Generate a 4-second pure-black "NEXT TIME ON" card — One Piece-style.
    'NEXT TIME ON' fades in at 0.5s, show title at 1.0s, both hold then fade out.
    Plays in silence (audio ends before this segment starts).
    Returns path or None on failure.
    """
    font = get_font_path()
    tbc_path = f"{tmp_dir}/ep{episode_number}_tbc.mp4"

    if font:
        font_safe  = font.replace("\\", "/").replace(":", "\\:")
        show_safe  = story_title.upper().replace("'", "\\'").replace(":", "\\:")
        vf = (
            f"drawtext=fontfile='{font_safe}':"
            f"text='NEXT TIME ON':"
            f"fontsize=30:fontcolor=white:"
            f"alpha='min(max(t-0.5,0)/0.5,1)*min(max(3.5-t,0)/0.3,1)':"
            f"x=(w-tw)/2:y=(h-th)/2-64,"
            f"drawtext=fontfile='{font_safe}':"
            f"text='{show_safe}':"
            f"fontsize=76:fontcolor=white:"
            f"alpha='min(max(t-1.0,0)/0.6,1)*min(max(3.5-t,0)/0.3,1)':"
            f"x=(w-tw)/2:y=(h-th)/2+18:"
            f"shadowcolor=black@0.85:shadowx=3:shadowy=3"
        )
    else:
        vf = "null"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={OW}x{OH}:r={OUTPUT_FPS}:d=4.0",
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        tbc_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        logger.warning(f"TBC card generation failed: {err[-200:]}")
        return None
    logger.info(f"TBC card: {tbc_path}")
    return tbc_path


# ---------------------------------------------------------------------------
# Audio mixing (unchanged from original — numpy/Python based)
# ---------------------------------------------------------------------------

def _load_wav_as_float(path: str, target_samples: int) -> np.ndarray:
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            raw        = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        if framerate != SAMPLE_RATE:
            new_len = int(len(data) * SAMPLE_RATE / framerate)
            data = np.interp(
                np.linspace(0, len(data) - 1, new_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        if len(data) < target_samples:
            data = np.pad(data, (0, target_samples - len(data)))
        return data[:target_samples].astype(np.float32)
    except Exception as e:
        logger.warning(f"Could not load WAV {path}: {e}")
        return np.zeros(target_samples, dtype=np.float32)


def _save_mixed_wav(mixed: np.ndarray, path: str) -> None:
    mixed_clipped = np.clip(mixed, -1.0, 1.0)
    pcm    = (mixed_clipped * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo.tobytes())


def mix_audio_tracks(
    voiceover_path: str,
    music_tracks: dict,
    output_path: str,
    total_duration_s: float,
    reveal_ms_list: list = None,
    silence_hold_ms: int = 0,
    scene_arc_data: list = None,
    cliff_ms: int = None,
) -> None:
    """
    Full dynamic audio mix — Python numpy, no FFmpeg filter issues.

    Layers:
    - Voiceover:        1.0  (dominant)
    - Score (MusicGen): 0.35 (arc-aware, baked by Music Agent)
    - Impact stings:    0.40 at each reveal, 0.55 at cliffhanger freeze
    - Silence hold:     hard dropout before cliffhanger end
    """
    reveal_ms_list = reveal_ms_list or []
    total_samples = int(total_duration_s * SAMPLE_RATE)

    logger.info(f"Mixing: {total_duration_s:.1f}s, {len(reveal_ms_list)} reveals")

    # Voiceover
    vo = _load_wav_as_float(voiceover_path, total_samples)
    mixed = vo * 1.0

    # Music: prefer the arc-automated pre-mixed score from Agent 7; fall back to
    # individual layer mixing for backward compatibility with old checkpoints.
    score_path = music_tracks.get("score", "")
    if score_path and Path(score_path).exists():
        score_audio = _load_wav_as_float(score_path, total_samples)
        mixed += score_audio * 0.35
        logger.info("Using arc-automated score mix")
    else:
        # Legacy per-layer mixing (kept for backward compatibility)
        if music_tracks.get("ambient") and Path(music_tracks["ambient"]).exists():
            ambient = _load_wav_as_float(music_tracks["ambient"], total_samples)
            mixed += ambient * 0.10

        if music_tracks.get("tension") and Path(music_tracks["tension"]).exists():
            tension = _load_wav_as_float(music_tracks["tension"], total_samples)
            mixed += tension * 0.25

        if music_tracks.get("heartbeat") and Path(music_tracks["heartbeat"]).exists():
            heartbeat = _load_wav_as_float(music_tracks["heartbeat"], total_samples)
            HB_ARC_GAIN = {
                "establish":     0.02,
                "build":         0.04,
                "investigation": 0.07,
                "revelation":    0.10,
                "cliffhanger":   0.12,
            }
            hb_env = np.ones(total_samples, dtype=np.float32) * 0.05
            if scene_arc_data:
                for seg in scene_arc_data:
                    s = max(0, min(int(seg["start_ms"] * SAMPLE_RATE / 1000), total_samples))
                    e = max(0, min(int(seg["end_ms"] * SAMPLE_RATE / 1000), total_samples))
                    if s < e:
                        gain = HB_ARC_GAIN.get(seg.get("arc_position", "build"), 0.10)
                        hb_env[s:e] = gain
            from scipy.ndimage import uniform_filter1d
            hb_env = uniform_filter1d(hb_env, size=int(SAMPLE_RATE * 0.5), mode="nearest")
            mixed += heartbeat * hb_env

        if music_tracks.get("strings") and Path(music_tracks["strings"]).exists():
            strings = _load_wav_as_float(music_tracks["strings"], total_samples)
            STR_ARC_GAIN = {
                "investigation": 0.03,
                "revelation":    0.07,
                "cliffhanger":   0.10,
            }
            str_env = np.zeros(total_samples, dtype=np.float32)
            if scene_arc_data:
                for seg in scene_arc_data:
                    gain = STR_ARC_GAIN.get(seg.get("arc_position", ""), 0.0)
                    if gain > 0:
                        s = max(0, min(int(seg["start_ms"] * SAMPLE_RATE / 1000), total_samples))
                        e = max(0, min(int(seg["end_ms"] * SAMPLE_RATE / 1000), total_samples))
                        if s < e:
                            str_env[s:e] = gain
            from scipy.ndimage import uniform_filter1d
            str_env = uniform_filter1d(str_env, size=int(SAMPLE_RATE * 1.5), mode="nearest")
            mixed += strings * str_env

        if music_tracks.get("location_ambient") and Path(music_tracks["location_ambient"]).exists():
            loc_amb = _load_wav_as_float(music_tracks["location_ambient"], total_samples)
            mixed += loc_amb * 0.06

    # Impact stings — reveals at 0.40, cliffhanger freeze at 0.55 (louder hit)
    if music_tracks.get("sting") and Path(music_tracks["sting"]).exists():
        sting_data    = _load_wav_as_float(music_tracks["sting"], total_samples)
        sting_raw_len = int(SAMPLE_RATE * 2.0)  # full 2-second sting with reverb tail

        for rev_ms in reveal_ms_list:
            sting_start = int(rev_ms * SAMPLE_RATE / 1000)
            sting_len   = min(sting_raw_len, total_samples - sting_start)
            if sting_start < total_samples and sting_len > 0:
                mixed[sting_start:sting_start + sting_len] += sting_data[:sting_len] * 0.40

        # Cliffhanger freeze sting — harder than reveal stings, lands right as image freezes
        if cliff_ms is not None and cliff_ms > 0:
            cs = int(cliff_ms * SAMPLE_RATE / 1000)
            cl = min(sting_raw_len, total_samples - cs)
            if cs < total_samples and cl > 0:
                mixed[cs:cs + cl] += sting_data[:cl] * 0.55

    # Silence hold before end — hard audio dropout
    if silence_hold_ms > 0:
        silence_start = max(0, total_samples - int(silence_hold_ms * SAMPLE_RATE / 1000))
        fade_len = min(int(SAMPLE_RATE * 0.2), total_samples - silence_start)
        mixed[silence_start:silence_start + fade_len] *= np.linspace(1, 0, fade_len)
        mixed[silence_start + fade_len:] = 0.0

    # Normalise
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.92 / max_val)

    # Save and encode
    tmp_wav = output_path.replace(".aac", "_premix.wav")
    _save_mixed_wav(mixed, tmp_wav)

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_wav,
        "-acodec", "aac",
        "-b:a", "192k",
        "-ar", str(SAMPLE_RATE),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)

    try:
        if Path(tmp_wav).exists():
            os.remove(tmp_wav)
    except Exception:
        pass

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise AgentException(f"Audio encode failed: {err[-300:]}")

    logger.info(f"Dynamic audio mixed: {output_path}")


# ---------------------------------------------------------------------------
# Video + Audio mux
# ---------------------------------------------------------------------------

def combine_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_offset_s: float = 0.0,
) -> None:
    """
    Mux video and audio streams into the final MP4.

    audio_offset_s: delay the audio stream by this many seconds relative to the
    video.  Use 3.0 when a 3-second title card is prepended to the video so that
    narration begins at the first real scene rather than over the title card.
    """
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if audio_offset_s > 0.0:
        cmd += ["-itsoffset", f"{audio_offset_s:.3f}"]
    cmd += [
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v",
        "-map", "1:a",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise AgentException(f"Mux failed: {err[-300:]}")
    logger.info(f"Final: {output_path}")


# ---------------------------------------------------------------------------
# Episode renderer
# ---------------------------------------------------------------------------

def render_episode(
    edl,
    music_tracks,
    episode_number,
    reveal_moments,
    voiceover_timestamps,
    context=None,
) -> str:
    """
    Render one episode end-to-end:
      1. Render each EDL entry to a short MP4 segment (Ken Burns motion).
      2. Concatenate segments.
      4. Mix audio.
      5. Mux video + audio.
    Returns the path to the final episode MP4.
    """
    # Pre-flight: ffmpeg must be available — fail fast rather than silently
    # producing an empty episode after minutes of rendering attempts.
    import shutil
    if not shutil.which("ffmpeg"):
        raise AgentException(
            "ffmpeg not found in PATH. Install ffmpeg and ensure it is accessible "
            "from the terminal before running the compositor."
        )

    entries   = edl["entries"]
    audio_path = edl["audio_path"]

    try:
        with wave.open(audio_path, "rb") as wf:
            total_s = wf.getnframes() / wf.getframerate()
    except Exception:
        total_s = edl.get("total_duration_ms", 90000) / 1000

    tmp_dir = f"outputs/final/tmp_ep{episode_number}"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # ── Story title ──────────────────────────────────────────────────────────
    story_title = context.get("story_title", "") if context else ""

    # ── Word timestamps for subtitles ────────────────────────────────────────
    word_timestamps = []
    if context:
        ep_words = context.get("word_timestamps", [])
        if episode_number - 1 < len(ep_words):
            word_timestamps = ep_words[episode_number - 1]

    # ── Original narration chunks for subtitle text correction ────────────────
    structured_chunks = []
    if context:
        narration = context.get("structured_narration", [])
        ep_narration_idx = episode_number - 1
        if ep_narration_idx < len(narration):
            structured_chunks = narration[ep_narration_idx]

    # ── Generate subtitle file ───────────────────────────────────────────────
    # Offset by 3 s when a title card is prepended so timestamps are absolute.
    # Subtitles are burned post-concat (not per-segment) to avoid the PTS=0 bug
    # where the first subtitle event appears frozen in every segment.
    # structured_chunks corrects Whisper transcription errors while preserving timing.
    subtitle_path   = None
    title_offset_ms = 4000 if story_title else 0  # title card is 4 s
    subtitle_path   = f"{tmp_dir}/ep{episode_number}_subtitles.ass"
    try:
        # Prefer unified event stream subtitle_groups — exact WhisperX word-level ms.
        # Fall back to word_timestamps grouping, then chunk-level entries.
        unified_events = context.get("unified_events", []) if context else []
        ep_stream      = unified_events[episode_number - 1] if (episode_number - 1) < len(unified_events) else {}
        sub_groups     = ep_stream.get("subtitle_groups", [])

        if sub_groups:
            _subtitle_groups_to_ass(sub_groups, subtitle_path, offset_ms=title_offset_ms)
        elif word_timestamps:
            generate_ass_subtitles(
                word_timestamps, subtitle_path,
                offset_ms=title_offset_ms,
                structured_chunks=structured_chunks or None,
            )
        else:
            subtitle_path = None
    except Exception as e:
        logger.warning(f"Subtitle generation failed: {e}")
        subtitle_path = None

    # ── Build reveal_ms_list for audio mixing ────────────────────────────────
    reveal_ms_list = []
    for reveal in reveal_moments:
        chunk_idx = reveal.get("chunk_index", 0)
        for ts in voiceover_timestamps:
            if ts["chunk_index"] == chunk_idx:
                reveal_ms_list.append(ts["start_ms"])
                break

    # ── Arc data for heartbeat/strings mixing ─────────────────────────────────
    scene_arc_data = [
        {
            "start_ms":    e["start_ms"],
            "end_ms":      e["end_ms"],
            "arc_position": e.get("arc_position", "build"),
        }
        for e in entries
        if e.get("type") in ("scene", "cliffhanger")
    ]

    # ── Silence hold ─────────────────────────────────────────────────────────
    cliffhanger_entry = next((e for e in entries if e.get("type") == "cliffhanger"), {})
    silence_hold_ms   = cliffhanger_entry.get("silence_hold_ms", 0)

    # ── Title card segment ───────────────────────────────────────────────────
    os.makedirs("outputs/final", exist_ok=True)

    # Opening hook: use the highest-arc-position image as the title card background
    hook_image = None
    for arc_pref in ("cliffhanger", "revelation", "investigation"):
        for e in entries:
            if e.get("arc_position") == arc_pref and e.get("image_path") and Path(e["image_path"]).exists():
                hook_image = e["image_path"]
                break
        if hook_image:
            break

    title_seg = None
    if story_title:
        try:
            title_seg = generate_title_segment(
                episode_number, story_title, episode_number, tmp_dir,
                hook_image_path=hook_image,
            )
        except Exception as e:
            logger.warning(f"Title card failed: {e}")

    # ── Render each entry as a video segment ─────────────────────────────────
    seg_paths = []
    if title_seg and Path(title_seg).exists():
        seg_paths.append(title_seg)

    for idx, entry in enumerate(entries):
        img = entry.get("image_path", "")
        vp  = entry.get("video_path", "")
        has_image = bool(img) and Path(img).exists()
        has_video = bool(vp) and Path(vp).exists()
        if not has_image and not has_video:
            logger.warning(f"Entry {idx}: no image or video — skipping (type={entry.get('type','?')})")
            continue

        # Suppress text overlays on revelation/cliffhanger — labeling the discovery
        # moment ("HIDDEN ROOM") destroys the reveal before the viewer can feel it.
        if entry.get("arc_position") in ("revelation", "cliffhanger"):
            timestamp_text = None
        else:
            timestamp_text = (
                entry.get("timestamp_overlay")
                or entry.get("text_overlay")
                or None
            )

        features = {
            "subtitles_path":   None,
            "title_text":       None,
            "timestamp_text":   timestamp_text,
            "show_lens_flare":  entry.get("arc_position") in ("revelation", "cliffhanger"),
            "show_fog":         entry.get("arc_position") in ("establish", "build"),
            "entry_idx":        idx,
            "movement":         get_movement_for_entry(entry, idx),
        }

        try:
            seg = render_entry_segment(
                entry, idx, episode_number, features, tmp_dir
            )
            if seg and Path(seg).exists():
                seg_paths.append(seg)
            else:
                logger.warning(
                    f"Ep{episode_number} segment {idx} dropped — render returned no file. "
                    f"Entry type={entry.get('type','?')} arc={entry.get('arc_position','?')} "
                    f"image={entry.get('image_path','missing')}"
                )
        except Exception as e:
            logger.error(f"Entry {idx} failed: {e}")

    if not seg_paths:
        raise AgentException(f"Episode {episode_number}: no segments rendered")

    # ── TBC card (appended after cliffhanger, plays in silence) ─────────────
    # One Piece-style: music has already faded out under the cliffhanger segment;
    # this card appears in pure silence — "NEXT TIME ON / [SHOW TITLE]" on black.
    if story_title:
        try:
            tbc_seg = generate_tbc_segment(story_title, episode_number, tmp_dir)
            if tbc_seg and Path(tbc_seg).exists():
                seg_paths.append(tbc_seg)
        except Exception as e:
            logger.warning(f"TBC card failed (non-fatal): {e}")

    # ── Concatenate segments ─────────────────────────────────────────────────
    video_path = concat_segments(seg_paths, episode_number, tmp_dir)

    # ── Burn subtitles post-concat (absolute timestamps, correct PTS) ─────────
    if subtitle_path and Path(subtitle_path).exists():
        subbed_path = f"{tmp_dir}/ep{episode_number}_subbed.mp4"
        video_path = burn_subtitles_on_video(video_path, subtitle_path, subbed_path)

    # ── Post-concat shadow seal (no colour shift — grade is per-segment) ─────
    sealed_path = f"{tmp_dir}/ep{episode_number}_sealed.mp4"
    video_path = apply_shadow_seal(video_path, sealed_path)

    # ── Mix audio ────────────────────────────────────────────────────────────
    tmp_audio = f"{tmp_dir}/ep{episode_number}_audio.aac"
    cliff_entry = next((e for e in entries if e.get("type") == "cliffhanger"), {})
    mix_audio_tracks(
        voiceover_path=audio_path,
        music_tracks=music_tracks,
        output_path=tmp_audio,
        total_duration_s=total_s,
        reveal_ms_list=reveal_ms_list,
        silence_hold_ms=silence_hold_ms,
        scene_arc_data=scene_arc_data,
        cliff_ms=cliff_entry.get("start_ms"),
    )

    # ── Mux video + audio ────────────────────────────────────────────────────
    # Delay audio by 4 s (title card duration) so narration starts exactly
    # at the first real scene frame rather than playing over the title card.
    title_card_s = 4.0 if (title_seg and Path(title_seg).exists()) else 0.0
    final_path = f"outputs/final/episode_{episode_number}.mp4"
    combine_video_audio(video_path, tmp_audio, final_path, audio_offset_s=title_card_s)

    # ── Clean up tmp dir ─────────────────────────────────────────────────────
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    logger.info(f"Episode {episode_number} complete: {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def run_agent_6b(context: dict) -> dict:
    """Agent 6b: Video Compositor (FFmpeg filter_complex architecture)."""
    logger.info("Starting video composition (Agent 6b — FFmpeg)...")

    edl_list           = context.get("edl", [])
    music_tracks_all   = context.get("music_tracks", [])
    reveal_moments_all = context.get("reveal_moments", [])
    timestamps_all     = context.get("voiceover_timestamps", [])
    total_episodes     = context.get("total_episodes", 0)

    if not edl_list:
        context.setdefault("errors", []).append(
            {"agent": "video_compositor", "error": "No EDLs"}
        )
        return context

    if "final_episodes" not in context:
        context["final_episodes"] = []

    for ep_idx in range(total_episodes):
        if ep_idx >= len(edl_list):
            continue
        edl = edl_list[ep_idx]
        if not edl or not edl.get("entries"):
            continue

        music_tracks   = music_tracks_all[ep_idx]   if ep_idx < len(music_tracks_all)   else {}
        reveal_moments = reveal_moments_all[ep_idx] if ep_idx < len(reveal_moments_all) else []
        timestamps     = timestamps_all[ep_idx]      if ep_idx < len(timestamps_all)      else []

        try:
            final_path = render_episode(
                edl,
                music_tracks,
                ep_idx + 1,
                reveal_moments,
                timestamps,
                context=context,
            )
            context["final_episodes"].append(final_path)
        except Exception as e:
            logger.error(f"Episode {ep_idx + 1} failed: {e}")
            context.setdefault("errors", []).append(
                {
                    "agent":   "video_compositor",
                    "episode": ep_idx + 1,
                    "error":   str(e),
                }
            )

    logger.info(
        f"Composition: {len(context['final_episodes'])}/{total_episodes} rendered"
    )
    return context
