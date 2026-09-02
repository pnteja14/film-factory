"""
Agent Event Stream — unified timeline all downstream agents read from.

Three timelines reconciled into one shared clock:
  NARRATION events  → WhisperX word-level timestamps (±10ms with WhisperX large-v3)
  STORY_BEAT events → revelation/cliffhanger resolved to ms via trigger phrase matching
  MUSIC events      → librosa beat/onset/section analysis of generated music

Cut point resolution — priority table (higher wins when events collide within 80ms):
  STORY_BEAT          100   always wins: revelation and cliffhanger
  MUSIC_ONSET_STRONG   80   strong instrument attack coinciding with story boundary
  SENTENCE_END         60   spoken sentence boundary (punctuation or 400ms+ gap)
  MUSIC_ONSET          40   regular instrument attack
  CHUNK_END            30   narration chunk boundary
  MUSIC_BEAT           20   beat grid tick
  WORD_END             10   individual word end (subtitle boundaries)

Subtitle groups are built from word events at exact WhisperX milliseconds —
no proportional distribution, no character-count estimation.

Output: context['unified_events'] per episode + outputs/alignment/ep{N}_events.json
"""

import json
from pathlib import Path
from utils.logger import AgentLogger

logger = AgentLogger("EventStream")

ALIGNMENT_DIR = Path("outputs/alignment")
ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

CUT_PRIORITY = {
    "STORY_BEAT":         100,
    "MUSIC_ONSET_STRONG":  80,
    "SENTENCE_END":        60,
    "MUSIC_ONSET":         40,
    "CHUNK_END":           30,
    "MUSIC_BEAT":          20,
    "WORD_END":            10,
}

_SNAP_WINDOW_MS = 150  # snap narration/story cuts to music beat if within this window


# ── NARRATION ANALYSIS ────────────────────────────────────────────────────────

def _sentence_end_ms(words: list) -> list:
    """Detect sentence boundaries from punctuation and gaps > 400ms."""
    out = []
    for i, w in enumerate(words):
        raw = w.get("word_raw", w.get("word", "")).rstrip("\"'")
        if raw.endswith((".", "!", "?", "…", "—")):
            out.append(w["end_ms"])
        elif i + 1 < len(words):
            if words[i + 1]["start_ms"] - w["end_ms"] > 400:
                out.append(w["end_ms"])
    return out


# ── STORY BEAT RESOLUTION ─────────────────────────────────────────────────────

def _resolve_beats(narration: list, reveals: list, words: list) -> list:
    """
    Match story beat trigger phrases to exact WhisperX milliseconds.
    Forward-only scan enforces temporal order.
    """
    beats = []
    if not words:
        return beats

    def _tokens(text: str) -> list:
        return [w.strip(".,!?;:\"'—…") for w in text.lower().split()
                if w.strip(".,!?;:\"'—…")]

    for reveal in (reveals or []):
        trigger = reveal.get("trigger_phrase", "")
        if not trigger:
            continue
        tw = _tokens(trigger)
        if not tw:
            continue
        last_word = tw[-1]
        resolved_ms = None
        for i, w in enumerate(words):
            if w["word"] != last_word:
                continue
            n   = min(len(tw), i + 1)
            ctx = [words[i - n + 1 + j]["word"] for j in range(n)]
            if sum(a == b for a, b in zip(tw[-n:], ctx)) >= max(1, n // 2):
                resolved_ms = w["end_ms"]
                break
        beats.append({
            "type":        "STORY_BEAT",
            "source":      "story",
            "ms":          resolved_ms,
            "priority":    CUT_PRIORITY["STORY_BEAT"],
            "beat_type":   "revelation",
            "trigger":     trigger,
            "chunk_index": reveal.get("chunk_index", 0),
            "image_path":  reveal.get("image_path", ""),
            "data":        {"beat_type": "revelation"},
        })

    # Cliffhanger = last spoken word
    if narration and words:
        beats.append({
            "type":        "STORY_BEAT",
            "source":      "story",
            "ms":          words[-1]["end_ms"],
            "priority":    CUT_PRIORITY["STORY_BEAT"],
            "beat_type":   "cliffhanger",
            "trigger":     "end_of_narration",
            "chunk_index": len(narration) - 1,
            "image_path":  "",
            "data":        {"beat_type": "cliffhanger"},
        })

    return beats


# ── CUT POINT RESOLVER ────────────────────────────────────────────────────────

def _snap(ms: int, music_times: list, window: int = _SNAP_WINDOW_MS) -> int:
    """Snap a timestamp to the nearest music beat/onset if within window."""
    if not music_times:
        return ms
    nearest = min(music_times, key=lambda b: abs(b - ms))
    return int(nearest) if abs(nearest - ms) <= window else ms


def _resolve_cut_points(events: list, all_music_ms: list) -> list:
    """
    Merge events within 80ms → pick winner by priority → snap to music beat.
    Returns canonical cut point list sorted by ms.
    """
    candidates = [e for e in events if e["type"] in CUT_PRIORITY and e.get("ms") is not None]
    if not candidates:
        return []

    candidates.sort(key=lambda e: (e["ms"], -e.get("priority", 0)))
    merged = []
    i = 0
    while i < len(candidates):
        group = [candidates[i]]
        j = i + 1
        while j < len(candidates) and candidates[j]["ms"] - candidates[i]["ms"] <= 80:
            group.append(candidates[j])
            j += 1
        winner   = max(group, key=lambda e: e.get("priority", 0))
        snap_ms  = _snap(winner["ms"], all_music_ms) if winner["source"] != "librosa" else winner["ms"]
        merged.append({
            "ms":       snap_ms,
            "type":     winner["type"],
            "priority": winner.get("priority", 0),
            "source":   winner["source"],
            "data":     winner.get("data", {}),
        })
        i = j

    return merged


# ── SUBTITLE GROUPS ───────────────────────────────────────────────────────────

def _subtitle_groups(words: list, max_words: int = 4, max_ms: int = 3000) -> list:
    """
    Build subtitle display groups from WhisperX word timestamps.
    Fires at exact word start_ms — no proportional estimation.
    """
    groups = []
    i = 0
    while i < len(words):
        g_words = []
        g_start = words[i]["start_ms"]
        g_end   = g_start
        while i < len(words):
            w = words[i]
            if len(g_words) >= max_words or (g_words and w["end_ms"] - g_start > max_ms):
                break
            g_words.append(w.get("word_raw", w.get("word", "")))
            g_end = w["end_ms"]
            i += 1
        if g_words:
            groups.append({"start_ms": g_start, "end_ms": g_end, "text": " ".join(g_words)})
    return groups


# ── MAIN BUILDER ──────────────────────────────────────────────────────────────

def build_event_stream(
    episode_number: int,
    word_timestamps: list,
    chunk_timestamps: list,
    structured_narration: list,
    reveal_moments: list,
    music_analysis: dict,
    audio_path: str,
) -> dict:
    """Build the unified event stream for one episode."""
    events = []
    narration = structured_narration or []
    words     = word_timestamps or []

    # WORD events
    for w in words:
        ci   = w.get("chunk_index", 0)
        base = {"chunk_index": ci, "word": w.get("word", ""), "word_raw": w.get("word_raw", w.get("word", ""))}
        events.append({"ms": w["start_ms"], "type": "WORD_START", "source": "whisperx", "priority": 1,  "data": dict(base)})
        events.append({"ms": w["end_ms"],   "type": "WORD_END",   "source": "whisperx", "priority": CUT_PRIORITY["WORD_END"], "data": dict(base)})

    # SENTENCE_END events
    for ms in _sentence_end_ms(words):
        events.append({"ms": ms, "type": "SENTENCE_END", "source": "whisperx",
                        "priority": CUT_PRIORITY["SENTENCE_END"], "data": {}})

    # CHUNK boundary events
    for ct in (chunk_timestamps or []):
        ci  = ct["chunk_index"]
        arc = narration[ci].get("arc_position", "build") if ci < len(narration) else "build"
        txt = narration[ci].get("text", "")              if ci < len(narration) else ""
        events.append({"ms": ct["start_ms"], "type": "CHUNK_START", "source": "whisperx",
                        "priority": 5, "data": {"chunk_index": ci, "arc_position": arc, "text": txt}})
        events.append({"ms": ct["end_ms"],   "type": "CHUNK_END",   "source": "whisperx",
                        "priority": CUT_PRIORITY["CHUNK_END"], "data": {"chunk_index": ci, "arc_position": arc}})

    # STORY_BEAT events
    story_beats = _resolve_beats(narration, reveal_moments, words)

    # All music timestamps for snapping
    ma = music_analysis or {}
    all_music_ms = sorted(set(ma.get("beat_times_ms", []) + ma.get("onset_times_ms", [])))

    for beat in story_beats:
        if beat["ms"] is not None:
            beat["ms"] = _snap(beat["ms"], all_music_ms)
        events.append(beat)

    # MUSIC events
    for ms in ma.get("beat_times_ms", []):
        events.append({"ms": int(ms), "type": "MUSIC_BEAT",          "source": "librosa", "priority": CUT_PRIORITY["MUSIC_BEAT"],          "data": {}})
    for ms in ma.get("onset_times_ms", []):
        events.append({"ms": int(ms), "type": "MUSIC_ONSET",         "source": "librosa", "priority": CUT_PRIORITY["MUSIC_ONSET"],         "data": {}})
    for ms in ma.get("strong_onset_times_ms", []):
        events.append({"ms": int(ms), "type": "MUSIC_ONSET_STRONG",  "source": "librosa", "priority": CUT_PRIORITY["MUSIC_ONSET_STRONG"],  "data": {}})
    for ms in ma.get("section_times_ms", []):
        events.append({"ms": int(ms), "type": "MUSIC_SECTION",       "source": "librosa", "priority": 25, "data": {}})

    events.sort(key=lambda e: (e.get("ms") or 0, -e.get("priority", 0)))
    cut_points     = _resolve_cut_points(events, all_music_ms)
    sub_groups     = _subtitle_groups(words)

    stream = {
        "episode":         episode_number,
        "audio_path":      audio_path,
        "events":          events,
        "cut_points":      cut_points,
        "story_beats":     story_beats,
        "subtitle_groups": sub_groups,
        "music_tempo_bpm": ma.get("tempo_bpm", 0),
    }

    out = ALIGNMENT_DIR / f"ep{episode_number}_events.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stream, f, indent=2)

    logger.info(
        f"Episode {episode_number}: {len(events)} events, "
        f"{len(cut_points)} cut points, {len(story_beats)} story beats, "
        f"{len(sub_groups)} subtitle groups"
    )
    return stream


def run_event_stream(context: dict) -> dict:
    """Build unified event stream for all episodes."""
    logger.info("Building unified event stream...")
    context["unified_events"] = []

    for ep_idx in range(context.get("total_episodes", 0)):
        def _ep(key, default):
            lst = context.get(key, [])
            return lst[ep_idx] if ep_idx < len(lst) else default

        stream = build_event_stream(
            episode_number       = ep_idx + 1,
            word_timestamps      = _ep("word_timestamps", []),
            chunk_timestamps     = _ep("voiceover_timestamps", []),
            structured_narration = _ep("structured_narration", []),
            reveal_moments       = _ep("reveal_moments", []),
            music_analysis       = _ep("music_analysis", {}),
            audio_path           = _ep("audio_files", ""),
        )
        context["unified_events"].append(stream)

    return context
