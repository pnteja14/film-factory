import json
import os
import re
from pathlib import Path
from utils.logger import AgentLogger
from utils.errors import AgentException
from utils.retry import retry_with_backoff, RetryableException
import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = AgentLogger("EditPlanner")

ZOOM_RATES = {
    "establish":     0.015,
    "build":         0.022,
    "investigation": 0.030,
    "revelation":    0.010,
    "cliffhanger":   0.005,
}

ZOOM_DIRECTIONS = {
    "creep": "in", "approach": "in", "fixate": "in", "descend": "in",
    "drift": "out", "sweep": "out", "linger": "none",
    "freeze": "none", "recoil": "out", "rise": "out",
}

REVEAL_MIN_MS = 1500  # reveals never fire before 1.5s — narrator must establish first


def get_scene_for_ms(scene_edl: list, time_ms: int) -> dict:
    """Find which scene is playing at a given millisecond."""
    for entry in scene_edl:
        if entry["start_ms"] <= time_ms < entry["end_ms"]:
            return entry
    if scene_edl:
        # Before first entry: use first scene (not last) to avoid showing the
        # cliffhanger image in the opening frames due to Whisper's small start offset.
        if time_ms < scene_edl[0]["start_ms"]:
            return scene_edl[0]
        return scene_edl[-1]
    return {}


def validate_reveal_chunk_indices(
    reveal_moments: list,
    structured_chunks: list,
) -> list:
    """
    Validate reveal chunk_index assignments.
    Search trigger_phrase against actual chunk text.
    Correct chunk_index if wrong.
    """
    validated = []
    for reveal in reveal_moments:
        trigger = reveal.get("trigger_phrase", "").lower().strip()
        assigned_idx = reveal.get("chunk_index", 0)

        if not trigger:
            validated.append(reveal)
            continue

        trigger_words = set(trigger.split())
        best_idx = assigned_idx
        best_score = 0

        for i, chunk in enumerate(structured_chunks):
            chunk_text = chunk.get("text", "").lower()
            chunk_words = set(chunk_text.split())
            score = len(trigger_words & chunk_words)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx != assigned_idx:
            logger.info(
                f"Reveal chunk_index corrected: {assigned_idx} → {best_idx} "
                f"(trigger: '{trigger}')"
            )
            reveal = reveal.copy()
            reveal["chunk_index"] = best_idx

        validated.append(reveal)

    return validated


def _cap_same_image_runs(scene_edl: list, max_run: int) -> None:
    """
    In-place: rotate any image that appears more than max_run times consecutively.
    Finds the nearest entry (ahead) with a different image and swaps it in.
    This prevents a single scene from monopolising the screen for the entire video.
    """
    i = 0
    while i < len(scene_edl):
        run_image = scene_edl[i].get("image_path", "")
        run_end = i
        while run_end < len(scene_edl) and scene_edl[run_end].get("image_path", "") == run_image:
            run_end += 1
        run_len = run_end - i

        if run_len > max_run:
            # Find a different-image entry later in the list to pull forward
            for swap_j in range(run_end, len(scene_edl)):
                if scene_edl[swap_j].get("image_path", "") != run_image:
                    # Swap the overrun start entry with the different-image entry
                    swap_at = i + max_run  # first position to replace
                    scene_edl[swap_at], scene_edl[swap_j] = scene_edl[swap_j], scene_edl[swap_at]
                    logger.info(
                        f"Same-image cap: rotated scene at index {swap_j} → {swap_at} "
                        f"to break {run_len}-entry run of '{Path(run_image).name}'"
                    )
                    break
        i = run_end if run_len <= max_run else i + max_run + 1


def _snap_to_cut(ms: int, cut_points: list, window_ms: int = 200) -> int:
    """Snap a timestamp to the nearest music-aligned cut point if within window_ms."""
    if not cut_points:
        return ms
    nearest = min(cut_points, key=lambda cp: abs(cp["ms"] - ms))
    return nearest["ms"] if abs(nearest["ms"] - ms) <= window_ms else ms


def build_scene_edl(
    scene_prompts: list,
    total_audio_ms: int,
    voiceover_timestamps: list = None,
    cut_points: list = None,
) -> list:
    """
    Build base scene EDL distributed across audio duration.

    When voiceover_timestamps are available: scenes are mapped proportionally
    to the actual narration time spans so visuals align with what is being said.
    Fallback: distribute by arc weight (original behaviour).
    """
    if not scene_prompts:
        return []

    n_scenes = len(scene_prompts)
    scene_edl = []

    def _build_entry(i, scene, start_ms, duration_ms):
        duration_ms = max(duration_ms, 5000)
        arc_position = scene.get("arc_position", "build")
        movement_type = scene.get("movement_type", "drift")
        zoom_direction = ZOOM_DIRECTIONS.get(movement_type, "in")
        zoom_rate = ZOOM_RATES.get(arc_position, 0.02)
        if scene.get("zoom_direction"):
            zoom_direction = scene["zoom_direction"]
        if i % 2 == 0 and zoom_direction == "in":
            zoom_direction = "out"
        image_path = scene.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            # Fall back to location anchor so the slot never shows a black frame
            location_name = scene.get("location_name") or scene.get("location", "")
            slug = re.sub(r"[^a-z0-9]+", "_", location_name.lower()).strip("_")
            anchor_candidate = f"outputs/anchors/{slug}.png"
            if Path(anchor_candidate).exists():
                image_path = anchor_candidate
                logger.info(f"Scene {i+1}: using anchor fallback → {anchor_candidate}")
            else:
                logger.warning(f"Scene {i+1}: image missing and no anchor found — {image_path}")
        image_type = scene.get("image_type", "narrative")
        scene_video = scene.get("video_path", "")
        if scene_video and not Path(scene_video).exists():
            scene_video = ""
        return {
            "type": "scene",
            "index": i,
            "chunk_index": scene.get("chunk_index", scene.get("scene_number", i+1) - 1),
            "start_ms": start_ms,
            "end_ms": start_ms + duration_ms,
            "duration_ms": duration_ms,
            "image_path": image_path,
            "video_path": scene_video,
            "location": scene.get("location_name") or scene.get("location", "unknown"),
            "arc_position": arc_position,
            "movement_type": movement_type,
            "zoom_rate": zoom_rate,
            "zoom_direction": zoom_direction,
            "cut_type": "cut",
            "text_overlay": None,
            "desaturate": False,
            "flash_frame": False,
            # Pass through treatment flags from scene director
            "photo_evidence_treatment": (
                scene.get("photo_evidence_treatment", False) or
                image_type in ("object", "evidence")
            ),
        }

    # ── NARRATION-TIMESTAMP DISTRIBUTION ─────────────────────────────────────
    # Each scene has a chunk_index set by the scene director — it says which
    # spoken chunk that scene is supposed to illustrate.  Map scenes directly
    # to those chunk timestamps so visuals match what is actually being said.
    if voiceover_timestamps and len(voiceover_timestamps) > 0:
        from collections import defaultdict

        sorted_ts = sorted(voiceover_timestamps, key=lambda x: x.get("chunk_index", 0))
        ts_by_chunk = {t["chunk_index"]: t for t in sorted_ts}
        n_chunks = len(sorted_ts)

        # Group scenes by their assigned chunk_index
        chunk_scenes: dict = defaultdict(list)
        for i, scene in enumerate(scene_prompts):
            ci = scene.get("chunk_index", scene.get("scene_number", i + 1) - 1)
            ci = max(0, min(ci, n_chunks - 1))
            chunk_scenes[ci].append((i, scene))

        scene_edl_map: dict = {}

        VISUAL_OFFSET_MS = 300  # show image ~7 frames before narration — leads the cut visually

        for ci, scenes_in_chunk in sorted(chunk_scenes.items()):
            ts = ts_by_chunk.get(ci, sorted_ts[min(ci, n_chunks - 1)])
            chunk_start = ts["start_ms"]
            chunk_end   = ts["end_ms"]
            chunk_dur   = max(chunk_end - chunk_start, 5000)

            n_in_chunk = len(scenes_in_chunk)
            for j, (orig_idx, scene) in enumerate(scenes_in_chunk):
                # First scene starts just before narration for visual-first offset
                if j == 0:
                    seg_start = max(0, chunk_start - VISUAL_OFFSET_MS)
                    seg_start = _snap_to_cut(seg_start, cut_points or [])
                else:
                    seg_start = chunk_start + int(j * chunk_dur / n_in_chunk)
                    seg_start = _snap_to_cut(seg_start, cut_points or [])
                seg_end = chunk_start + int((j + 1) * chunk_dur / n_in_chunk)
                if j == n_in_chunk - 1:
                    seg_end = chunk_end
                scene_edl_map[orig_idx] = _build_entry(
                    orig_idx, scene, seg_start, max(seg_end - seg_start, 5000)
                )

        # Preserve original scene order for get_scene_for_ms lookups
        scene_edl = [scene_edl_map[i] for i in range(n_scenes) if i in scene_edl_map]

        # Enforce consecutive non-overlapping coverage
        scene_edl.sort(key=lambda e: e["start_ms"])
        for i in range(len(scene_edl) - 1):
            scene_edl[i]["end_ms"] = scene_edl[i + 1]["start_ms"]
            scene_edl[i]["duration_ms"] = max(
                scene_edl[i]["end_ms"] - scene_edl[i]["start_ms"], 1
            )
        if scene_edl:
            scene_edl[-1]["end_ms"] = total_audio_ms
            scene_edl[-1]["duration_ms"] = max(
                total_audio_ms - scene_edl[-1]["start_ms"], 5000
            )

        # Cap same image repeating more than MAX_SAME_IMAGE_RUN consecutive entries
        MAX_SAME_IMAGE_RUN = 2
        _cap_same_image_runs(scene_edl, MAX_SAME_IMAGE_RUN)

        return scene_edl

    # ── FALLBACK: ARC-WEIGHT DISTRIBUTION ────────────────────────────────────
    arc_weights = {
        "establish": 0.8, "build": 1.0,
        "investigation": 1.1, "revelation": 1.4, "cliffhanger": 1.6,
    }
    weights = [arc_weights.get(s.get("arc_position", "build"), 1.0) for s in scene_prompts]
    total_weight = sum(weights)
    MAX_SCENE_FRACTION = 0.40
    max_single_ms = int(total_audio_ms * MAX_SCENE_FRACTION)
    raw_durations = [max(int((w / total_weight) * total_audio_ms), 5000) for w in weights]
    excess_ms = sum(max(0, d - max_single_ms) for d in raw_durations)
    durations = [min(d, max_single_ms) for d in raw_durations]
    if excess_ms > 0:
        under_cap = [i for i, d in enumerate(raw_durations) if d < max_single_ms]
        if under_cap:
            under_weights = [weights[i] for i in under_cap]
            total_under = sum(under_weights)
            for idx, uc_i in enumerate(under_cap):
                bonus = int((under_weights[idx] / total_under) * excess_ms)
                durations[uc_i] = min(durations[uc_i] + bonus, max_single_ms)

    current_ms = 0
    for i, (scene, duration_ms) in enumerate(zip(scene_prompts, durations)):
        scene_edl.append(_build_entry(i, scene, current_ms, duration_ms))
        current_ms += max(duration_ms, 5000)

    return scene_edl


def resolve_reveal_ms(reveal: dict, voiceover_timestamps: list) -> int:
    """
    Find exact millisecond to cut for a reveal.
    Fires at the midpoint of the trigger chunk — by the midpoint the narrator
    has already spoken enough to set up the cut, and the reveal lands as the
    sentence resolves rather than before it starts.
    Always >= REVEAL_MIN_MS so the opening is never interrupted.
    """
    chunk_index = reveal.get("chunk_index", 0)

    for ts in voiceover_timestamps:
        if ts["chunk_index"] == chunk_index:
            chunk_mid = ts["start_ms"] + (ts["end_ms"] - ts["start_ms"]) * 2 // 3
            return max(REVEAL_MIN_MS, chunk_mid)

    if voiceover_timestamps:
        last = voiceover_timestamps[-1]
        return max(REVEAL_MIN_MS, last["end_ms"] // 2)

    return REVEAL_MIN_MS


def _build_nonoverlap_timeline(
    scene_edl: list,
    reveals_sorted: list,
    cliffhanger_entry: dict,
) -> list:
    """
    Build a non-overlapping EDL timeline.

    Reveals occupy their exact time windows.
    Scene entries fill every gap between reveals, respecting scene boundaries.
    Total duration_ms sums exactly to total_audio_ms — no -shortest needed.
    """
    result = []
    current_ms = 0
    cliff_start = cliffhanger_entry["start_ms"]

    def fill_gap(from_ms: int, to_ms: int) -> None:
        t = from_ms
        while t < to_ms:
            scene = get_scene_for_ms(scene_edl, t)
            if not scene:
                break
            seg_end = min(scene["end_ms"], to_ms)
            if seg_end <= t:
                break
            seg = scene.copy()
            seg["start_ms"] = t
            seg["end_ms"] = seg_end
            seg["duration_ms"] = seg_end - t
            if t != scene["start_ms"]:
                seg["is_return"] = True
            result.append(seg)
            t = seg_end

    for reveal in reveals_sorted:
        rev_start = reveal["start_ms"]
        rev_end = reveal["end_ms"]

        if rev_start <= current_ms:
            logger.warning(
                f"Reveal at {rev_start}ms overlaps previous end {current_ms}ms — skipping"
            )
            continue

        fill_gap(current_ms, rev_start)
        result.append(reveal)
        current_ms = rev_end

    if current_ms < cliff_start:
        fill_gap(current_ms, cliff_start)

    result.append(cliffhanger_entry)
    return result


def build_full_edl(
    scene_prompts: list,
    reveal_moments: list,
    voiceover_timestamps: list,
    episode_number: int,
    audio_path: str,
    structured_chunks: list = None,
    unified_stream: dict = None,
) -> dict:
    """
    Build complete EDL for one episode.

    1. Distribute scenes across audio duration by arc weight
    2. Insert reveals at resolved timestamps
    3. Return-to-scene after each reveal
    4. Cliffhanger freeze + desaturate at end
    5. Sort by start_ms

    No dramatic pauses. No silence insertion. No timestamp drift.
    """
    logger.info(f"Building EDL for episode {episode_number + 1}...")

    # Total duration from real Whisper timestamps.
    # Add 3000ms buffer (not 1000ms) so the cliffhanger has 3+ seconds of silence
    # after the last narration word — the WAV already has 2s of trailing silence.
    # Total duration is ALWAYS derived from actual narration end — never from
    # scene duration sums. This eliminates audio tail gaps across all future episodes.
    if voiceover_timestamps:
        last_narration_end_ms = voiceover_timestamps[-1]["end_ms"]
        total_audio_ms = last_narration_end_ms + 2000  # 2s breathing room, not 3s
    else:
        last_narration_end_ms = 0
        total_audio_ms = sum(
            s.get("duration_seconds", 8) * 1000 for s in scene_prompts
        )

    logger.info(f"Total audio: {total_audio_ms}ms ({total_audio_ms/1000:.1f}s)")

    # Validate reveal chunk indices against actual narration
    if structured_chunks and reveal_moments:
        reveal_moments = validate_reveal_chunk_indices(reveal_moments, structured_chunks)

    # Extract cut_points and story_beats from unified stream (music-snapped timing)
    cut_points  = (unified_stream or {}).get("cut_points",  [])
    story_beats = (unified_stream or {}).get("story_beats", [])

    # Build reveal-trigger lookup: trigger phrase → resolved ms (already music-snapped)
    beat_ms_by_trigger = {}
    for b in story_beats:
        if b.get("ms") is not None and b.get("trigger"):
            beat_ms_by_trigger[b["trigger"].lower().strip()] = b["ms"]

    # Build base scene EDL — pass timestamps + music cut_points for beat-snapped cuts
    scene_edl = build_scene_edl(scene_prompts, total_audio_ms, voiceover_timestamps, cut_points)

    if not scene_edl:
        logger.warning("No scene EDL entries")
        return {
            "episode_number": episode_number + 1,
            "audio_path": audio_path,
            "total_duration_ms": total_audio_ms,
            "entry_count": 0,
            "reveal_count": 0,
            "chunk_count": len(voiceover_timestamps),
            "dramatic_pause_count": 0,
            "entries": []
        }

    # ── INSERT REVEALS ────────────────────────────────────────────────
    reveal_insertions = []

    for r_idx, reveal in enumerate(reveal_moments):
        image_path = reveal.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            logger.warning(f"Reveal {r_idx + 1}: no image — skipping")
            continue

        # Use music-snapped story beat ms when available; fall back to chunk midpoint
        trigger_key = reveal.get("trigger_phrase", "").lower().strip()
        if trigger_key and trigger_key in beat_ms_by_trigger:
            cut_ms = beat_ms_by_trigger[trigger_key]
        else:
            cut_ms = resolve_reveal_ms(reveal, voiceover_timestamps)
        hold_ms = max(int(reveal.get("hold_seconds", 3.5) * 1000), 3500)
        cut_type = reveal.get("cut_type", "hard")
        image_type = reveal.get("image_type", "evidence")

        # Extend location_detail reveals further
        if image_type == "location_detail":
            hold_ms = max(hold_ms, 4000)

        # Alternate zoom direction: even reveals pull back to show context,
        # odd reveals push in for intensity
        reveal_zoom = "out" if r_idx % 2 == 0 else "in"

        reveal_video = reveal.get("video_path", "")
        if reveal_video and not Path(reveal_video).exists():
            reveal_video = ""
        reveal_entry = {
            "type": "reveal",
            "reveal_index": r_idx,
            "chunk_index": reveal.get("chunk_index", 0),
            "start_ms": cut_ms,
            "end_ms": cut_ms + hold_ms,
            "duration_ms": hold_ms,
            "image_path": image_path,
            "video_path": reveal_video,
            "cut_type": cut_type,
            "flash_frame": cut_type in ("flash", "smash"),
            "zoom_rate": 0.006,
            "zoom_direction": reveal_zoom,
            "text_overlay": reveal.get("text_overlay", None),
            "desaturate": False,
            "arc_position": "revelation",
            "reveal_type": image_type,
            "trigger_phrase": reveal.get("trigger_phrase", ""),
            "photo_evidence_treatment": image_type in ("evidence", "document"),
        }
        reveal_insertions.append(reveal_entry)

        # Return to scene after reveal
        if reveal.get("return_to_scene", True):
            return_ms = cut_ms + hold_ms
            scene_at_return = get_scene_for_ms(scene_edl, return_ms)
            if scene_at_return and scene_at_return["end_ms"] > return_ms:
                return_entry = {
                    "type": "scene",
                    "index": scene_at_return.get("index", 0),
                    "chunk_index": scene_at_return.get("chunk_index", 0),
                    "start_ms": return_ms,
                    "end_ms": scene_at_return["end_ms"],
                    "duration_ms": max(0, scene_at_return["end_ms"] - return_ms),
                    "image_path": scene_at_return["image_path"],
                    "location": scene_at_return.get("location", ""),
                    "arc_position": scene_at_return.get("arc_position", "build"),
                    "movement_type": scene_at_return.get("movement_type", "drift"),
                    "cut_type": "cut",
                    "zoom_rate": scene_at_return.get("zoom_rate", 0.02),
                    "zoom_direction": scene_at_return.get("zoom_direction", "in"),
                    "text_overlay": None,
                    "desaturate": False,
                    "flash_frame": False,
                    "is_return": True,
                }
                reveal_insertions.append(return_entry)

    # ── CLIFFHANGER ───────────────────────────────────────────────────
    # Start the freeze when the narrator stops speaking (last word + 200ms breath),
    # but no earlier than 5s before the end so the cliffhanger has real duration.
    # This prevents the freeze cutting into active narration.
    if voiceover_timestamps:
        cliff_from_last_word = last_narration_end_ms + 3000  # 3s hold after final word before freeze
        cliff_from_end      = total_audio_ms - 5000           # at least 5s from end
        cliffhanger_start_ms = max(cliff_from_last_word, cliff_from_end)
        cliffhanger_start_ms = max(0, min(cliffhanger_start_ms, total_audio_ms - 2000))
    else:
        cliffhanger_start_ms = max(0, total_audio_ms - 4000)
    last_scene = scene_edl[-1]

    # Cliffhanger image: prefer the last reveal's image for dramatic impact.
    # A reveal image (evidence/location_detail) is far more visceral than whatever
    # scene happens to be last in the EDL.
    cliff_image = last_scene["image_path"]
    valid_reveals = [r for r in reveal_moments if r.get("image_path") and Path(r["image_path"]).exists()]
    if valid_reveals:
        cliff_image = valid_reveals[-1]["image_path"]

    # Tell the music agent exactly when to peak its swell (4s before the freeze)
    music_swell_at_ms = max(0, cliffhanger_start_ms - 4000)

    # If the cliffhanger scene has a Wan video clip, use it — animated wide shot
    # is far more visceral than a frozen still. Desaturate still applies to the video.
    cliff_scene_video = None
    for scene in scene_prompts:
        if (scene.get("arc_position") == "cliffhanger"
                and scene.get("video_path")
                and Path(scene["video_path"]).exists()):
            cliff_scene_video = scene["video_path"]
            break

    cliffhanger_entry = {
        "type": "cliffhanger",
        "start_ms": cliffhanger_start_ms,
        "end_ms": total_audio_ms,
        "duration_ms": total_audio_ms - cliffhanger_start_ms,
        "image_path": cliff_image,
        "video_path": cliff_scene_video,
        "cut_type": "cut",
        "flash_frame": False,
        "zoom_rate": 0.0,
        "zoom_direction": "none",
        "text_overlay": None,
        "desaturate": True,
        "desaturate_start_ms": cliffhanger_start_ms,
        "arc_position": "cliffhanger",
        "silence_hold_ms": 3000,
        "music_swell_at_ms": music_swell_at_ms,
    }

    # ── INSERT SHOTS (word-level visual cuts) ─────────────────────────
    # For each scene with insert_shots, find the Whisper word timestamp
    # of the trigger_word and inject a short close-up EDL entry at that ms.
    # These are woven in alongside reveals — the same non-overlap builder handles them.
    word_timestamps: dict = {}   # word → list of ms positions
    if voiceover_timestamps:
        for ts in voiceover_timestamps:
            for w in ts.get("words", []):
                word = w.get("word", "").lower().strip(".,!?;:")
                if word:
                    word_timestamps.setdefault(word, []).append(int(w.get("start", 0) * 1000))

    insert_entries = []
    for scene in scene_prompts:
        for ins in scene.get("insert_shots", []):
            ins_path = ins.get("image_path", "")
            if not ins_path or not Path(ins_path).exists():
                continue
            trigger = ins.get("trigger_word", "").lower().strip(".,!?;:")
            hold_ms = max(1500, int(ins.get("hold_seconds", 2.0) * 1000))

            # Find earliest occurrence of trigger word after chunk start
            chunk_idx = scene.get("chunk_index", 0)
            chunk_start_ms = 0
            if voiceover_timestamps and chunk_idx < len(voiceover_timestamps):
                chunk_start_ms = voiceover_timestamps[chunk_idx].get("start_ms", 0)

            candidates = [ms for ms in word_timestamps.get(trigger, []) if ms >= chunk_start_ms]
            if not candidates:
                # Fall back to chunk midpoint if word not found in timestamps
                if voiceover_timestamps and chunk_idx < len(voiceover_timestamps):
                    ts = voiceover_timestamps[chunk_idx]
                    cut_ms = ts["start_ms"] + (ts["end_ms"] - ts["start_ms"]) // 2
                else:
                    continue
            else:
                cut_ms = min(candidates)

            cut_ms = max(REVEAL_MIN_MS, cut_ms)

            insert_entries.append({
                "type": "insert",
                "start_ms": cut_ms,
                "end_ms": cut_ms + hold_ms,
                "duration_ms": hold_ms,
                "image_path": ins_path,
                "trigger_word": trigger,
                "cut_type": "cut",
                "flash_frame": False,
                "zoom_rate": 0.004,
                "zoom_direction": "in",
                "text_overlay": None,
                "desaturate": False,
                "arc_position": "investigation",
                "photo_evidence_treatment": True,
                "music_profile": scene.get("arc_position", "investigation"),
            })

    # Merge inserts with reveals for non-overlap building
    all_timed_cuts = sorted(
        [e for e in reveal_insertions if e.get("type") == "reveal"] + insert_entries,
        key=lambda x: x["start_ms"],
    )

    # Deduplicate: if an insert and a reveal fire within 500ms of each other, keep the reveal
    deduped_cuts = []
    for cut in all_timed_cuts:
        if deduped_cuts and cut["start_ms"] - deduped_cuts[-1]["end_ms"] < 200:
            if cut.get("type") == "reveal":
                deduped_cuts[-1] = cut   # prefer reveal over insert
            continue
        deduped_cuts.append(cut)

    insert_count = sum(1 for e in deduped_cuts if e.get("type") == "insert")
    logger.info(f"Insert shots scheduled: {insert_count}")

    # ── PERSIST_VISUAL — extend scene EDL across consecutive same-subject chunks ─
    # When scene N has persist_visual=True the previous scene's image continues.
    # We achieve this by replacing scene N's image_path with scene N-1's image_path.
    for i in range(1, len(scene_edl)):
        scene = scene_prompts[i] if i < len(scene_prompts) else {}
        if scene.get("persist_visual") and scene_edl[i - 1].get("image_path"):
            scene_edl[i]["image_path"] = scene_edl[i - 1]["image_path"]
            logger.info(
                f"Persist visual: scene {i+1} inherits image from scene {i} "
                f"(chunk {scene.get('chunk_index')})"
            )

    # ── BUILD NON-OVERLAPPING TIMELINE ────────────────────────────────
    reveals_only = deduped_cuts
    all_entries = _build_nonoverlap_timeline(scene_edl, reveals_only, cliffhanger_entry)
    all_entries = [e for e in all_entries if e.get("duration_ms", 0) > 10]

    edl = {
        "episode_number": episode_number + 1,
        "audio_path": audio_path,
        "total_duration_ms": total_audio_ms,
        "total_audio_ms": total_audio_ms,
        "entry_count": len(all_entries),
        "reveal_count": len([e for e in reveal_insertions if e["type"] == "reveal"]),
        "insert_count": insert_count,
        "chunk_count": len(voiceover_timestamps),
        "dramatic_pause_count": 0,
        "entries": all_entries,
    }

    logger.info(
        f"EDL built: {len(all_entries)} entries, "
        f"{edl['reveal_count']} reveals, "
        f"{edl['insert_count']} inserts, "
        f"total {total_audio_ms/1000:.1f}s"
    )

    return edl


def run_agent_6a(context: dict) -> dict:
    """Agent 6a: Edit Planner — clean, no dramatic pauses."""
    logger.info("Building edit decision lists (Agent 6a)...")

    total_episodes       = context.get("total_episodes", 0)
    scene_prompts_all    = context.get("scene_prompts", [])
    reveal_moments_all   = context.get("reveal_moments", [])
    timestamps_all       = context.get("voiceover_timestamps", [])
    audio_files          = context.get("audio_files", [])
    structured_narration = context.get("structured_narration", [])
    unified_events_all   = context.get("unified_events", [])

    context["edl"] = []

    for ep_idx in range(total_episodes):
        logger.info(f"Planning episode {ep_idx + 1}...")
        try:
            scene_prompts        = scene_prompts_all[ep_idx]    if ep_idx < len(scene_prompts_all)    else []
            reveal_moments       = reveal_moments_all[ep_idx]   if ep_idx < len(reveal_moments_all)   else []
            voiceover_timestamps = timestamps_all[ep_idx]       if ep_idx < len(timestamps_all)       else []
            audio_path           = audio_files[ep_idx]          if ep_idx < len(audio_files)          else ""
            chunks               = structured_narration[ep_idx] if ep_idx < len(structured_narration) else []
            unified_stream       = unified_events_all[ep_idx]   if ep_idx < len(unified_events_all)   else None

            if not scene_prompts:
                logger.warning(f"Episode {ep_idx + 1}: no scene prompts")
                context["edl"].append({})
                continue

            valid_scenes = sum(
                1 for s in scene_prompts
                if s.get("image_path") and Path(s["image_path"]).exists()
            )
            logger.info(
                f"Episode {ep_idx + 1}: {len(scene_prompts)} scenes "
                f"({valid_scenes} with images), {len(reveal_moments)} reveals, "
                f"{len(voiceover_timestamps)} timestamps"
            )

            edl = build_full_edl(
                scene_prompts,
                reveal_moments,
                voiceover_timestamps,
                ep_idx,
                audio_path,
                structured_chunks=chunks,
                unified_stream=unified_stream,
            )

            context["edl"].append(edl)

            os.makedirs("outputs", exist_ok=True)
            edl_path = f"outputs/episode_{ep_idx + 1}_edl.json"
            with open(edl_path, "w") as f:
                json.dump(edl, f, indent=2)
            logger.info(f"EDL saved: {edl_path}")

        except Exception as e:
            logger.error(f"Episode {ep_idx + 1} EDL failed: {e}")
            context["errors"].append({
                "agent": "edit_planner",
                "episode": ep_idx + 1,
                "error": str(e)
            })
            context["edl"].append({})

    logger.info(f"Edit planning complete: {len(context['edl'])} EDLs")
    return context