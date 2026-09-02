"""
Agent 7 — Scene-based music generation.

Prompts are built from actual narration text for each arc segment.
No hardcoded genre labels, no hardcoded instrument names, no hardcoded
heartbeat tracks. MusicGen reads the story content and derives tone.

Pipeline: group EDL into arc segments → enrich with narration text + story
beats → build content-driven MusicGen prompt → generate → assemble with
event envelope (reveals drop, cliffhanger builds then fades).

Priority: Kaggle MusicGen-Large → local MusicGen-small → synth drone fallback.
"""

import hashlib
import io
import os
import wave
import numpy as np
from pathlib import Path
from utils.logger import AgentLogger

logger = AgentLogger("MusicAgent")

MUSIC_SOURCE     = os.environ.get("MUSIC_SOURCE", "local")
KAGGLE_MUSIC_URL = os.environ.get("KAGGLE_MUSIC_URL", "").rstrip("/")

MUSIC_DIR = Path("outputs/music")
MUSIC_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44100


def _check_musicgen():
    try:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        return True
    except ImportError:
        return False

MUSICGEN_AVAILABLE = _check_musicgen()


# ── DYNAMIC PROMPT CONSTRUCTION ───────────────────────────────────────────────

def _build_dynamic_prompt(
    narration_text: str,
    arc_position: str,
    story_beats_in_segment: list,
    duration_s: float,
    peak_at_s: float = None,
    episode_style: str = "",
) -> str:
    """
    Build a MusicGen prompt using musical descriptors for the arc position.
    episode_style is the shared sonic palette prefix — same instruments throughout.

    MusicGen responds to MUSICAL descriptors (dynamics, tempo, texture), NOT to
    narrative text like "a shocking discovery is revealed."  Narration excerpt
    is intentionally dropped; arc_music descriptors drive the generation.
    """
    # MusicGen responds to MUSICAL descriptors (dynamics, tempo, texture), not narrative text.
    # These per-arc musical descriptors drive the generation; the narration excerpt
    # provides thematic color but is less important than the instrumental directives.
    arc_music = {
        "establish":     "very quiet, pp dynamics, sparse single instrument, slow tempo, held notes, wide space between phrases",
        "build":         "mp dynamics, gradually increasing density, more instruments entering, rising tension, slow build",
        "investigation": "mf dynamics, irregular rhythm, staccato phrases, anxious, unresolved harmonics, searching",
        "revelation":    "sudden forte, full ensemble, dramatic, intense, fff dynamics, urgent, stabbing chords",
        "cliffhanger":   "climactic, maximum tension fff, sustained dissonance, crescendo to peak then abrupt cut to silence",
    }.get(arc_position, "mp dynamics, moderate tension")

    peak_context = f" Peak intensity at {peak_at_s:.0f} seconds in." if peak_at_s and peak_at_s > 2 else ""

    prompt = (
        f"{episode_style}"
        f"{arc_music}."
        f"{peak_context} "
        f"Duration {duration_s:.0f} seconds. No lyrics, no vocals, no singing."
    )
    return prompt


def _enrich_segments(segments, structured_narration, voiceover_timestamps, reveal_moments):
    """
    Add narration_text and story_beats to each arc segment so prompts are
    derived from actual story content rather than arc labels.
    """
    narration = structured_narration or []
    chunk_ts  = voiceover_timestamps or []
    reveals   = reveal_moments or []

    for seg in segments:
        seg_start = seg["start_ms"]
        seg_end   = seg["end_ms"]

        texts = []
        for ct in chunk_ts:
            if ct["start_ms"] < seg_end and ct["end_ms"] > seg_start:
                ci = ct["chunk_index"]
                if ci < len(narration):
                    t = narration[ci].get("text", "").strip()
                    if t:
                        texts.append(t)
        seg["narration_text"] = " ".join(texts)

        # Find reveal moments whose chunk falls within this segment
        seg_beats = []
        for rev in reveals:
            ci = rev.get("chunk_index", 0)
            if ci < len(chunk_ts):
                ct = chunk_ts[ci]
                if ct["start_ms"] >= seg_start and ct["end_ms"] <= seg_end:
                    seg_beats.append({"beat_type": "revelation", "trigger": rev.get("trigger_phrase", "")})
        seg["story_beats"] = seg_beats

        # If there's a story beat, find its approximate position within the segment
        if seg_beats and chunk_ts:
            first_beat_ci = reveals[0].get("chunk_index", 0) if reveals else 0
            if first_beat_ci < len(chunk_ts):
                beat_ms = chunk_ts[first_beat_ci].get("end_ms", seg_end)
                seg["peak_at_s"] = max(0.0, (beat_ms - seg_start) / 1000.0)
            else:
                seg["peak_at_s"] = None
        else:
            seg["peak_at_s"] = None

    return segments


# ── UTILS ─────────────────────────────────────────────────────────────────────

def _save_wav(samples: np.ndarray, path: str) -> None:
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.80
    pcm = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo.tobytes())
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    logger.info(f"Saved: {path} ({len(samples)/SAMPLE_RATE:.1f}s)")


def _loop_to_duration(audio: np.ndarray, target_s: float, xfade_s: float = 1.0) -> np.ndarray:
    """
    Loop audio to target duration using overlap-add at the loop seam.
    NO segment-level fade-in/out here — _assemble_score handles those.
    Double-fading caused 3.5s of silence at the start of every segment.
    """
    target = int(target_s * SAMPLE_RATE)
    if len(audio) >= target:
        return audio[:target].copy().astype(np.float32)

    xfade = min(int(SAMPLE_RATE * xfade_s), len(audio) // 4, target // 4)
    period = len(audio) - xfade  # advance per loop with overlap

    result = np.zeros(target, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, xfade).astype(np.float32)
    fade_in  = np.linspace(0.0, 1.0, xfade).astype(np.float32)

    pos = 0
    while pos < target:
        end = min(pos + len(audio), target)
        result[pos:end] += audio[:end - pos]
        # Blend outgoing tail with incoming head at next loop point
        blend_start = pos + period
        blend_end   = min(blend_start + xfade, target)
        if blend_end > blend_start:
            blen = blend_end - blend_start
            result[blend_start:blend_end] *= fade_out[:blen]
            result[blend_start:blend_end] += audio[:blen] * fade_in[:blen]
        pos += period

    return result[:target].astype(np.float32)


# ── GENERATION BACKENDS ────────────────────────────────────────────────────────

def _call_kaggle_music(prompt: str, duration_s: float, seed: int = -1) -> np.ndarray | None:
    if not KAGGLE_MUSIC_URL:
        return None
    try:
        import requests, base64 as _b64
        payload = {"prompt": prompt, "duration_seconds": min(duration_s, 30.0),
                   "guidance_scale": 4.5, "seed": seed}
        logger.info(f"Kaggle music: '{prompt[:80]}' ({duration_s:.0f}s)")
        resp = requests.post(f"{KAGGLE_MUSIC_URL}/generate_music", json=payload, timeout=360)
        if resp.status_code != 200:
            logger.warning(f"Kaggle music {resp.status_code}: {resp.text[:120]}")
            return None
        result    = resp.json()
        wav_bytes = _b64.b64decode(result["audio_base64"])
        remote_sr = result.get("sample_rate", 32000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            n_ch   = wf.getnchannels()
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch).mean(axis=1)
        if remote_sr != SAMPLE_RATE:
            try:
                from scipy.signal import resample as _rs
                pcm = _rs(pcm, int(len(pcm) * SAMPLE_RATE / remote_sr))
            except ImportError:
                target_n = int(len(pcm) * SAMPLE_RATE / remote_sr)
                pcm = np.interp(np.linspace(0, len(pcm) - 1, target_n),
                                np.arange(len(pcm)), pcm).astype(np.float32)
        return pcm.astype(np.float32)
    except Exception as e:
        logger.warning(f"Kaggle music failed: {e}")
        return None


def _musicgen_local(prompt: str, duration_s: float) -> np.ndarray | None:
    try:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        logger.info(f"Local MusicGen: '{prompt[:60]}' ({duration_s:.0f}s)")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model     = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        model.eval()
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        with torch.no_grad():
            audio_values = model.generate(
                **inputs, max_new_tokens=int(min(duration_s, 30) * 50),
                do_sample=True, guidance_scale=4.0,
            )
        audio = audio_values[0, 0].numpy()
        sr    = model.config.audio_encoder.sampling_rate
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Local MusicGen failed: {e}")
        return None


# ── SYNTH FALLBACK ─────────────────────────────────────────────────────────────

def _synth_drone(duration_s: float, arc_position: str = "build") -> np.ndarray:
    """Dark atmospheric drone — used only when MusicGen is completely unavailable."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n)
    intensity = {"establish": 0.55, "build": 0.70, "investigation": 0.80,
                 "revelation": 0.95, "cliffhanger": 1.0}.get(arc_position, 0.70)
    drone = (
        0.38 * np.sin(2 * np.pi * 55.00 * t) +
        0.20 * np.sin(2 * np.pi * 55.22 * t) +
        0.16 * np.sin(2 * np.pi * 54.80 * t) +
        0.14 * np.sin(2 * np.pi * 110.0 * t) +
        0.10 * np.sin(2 * np.pi * 82.5  * t) +
        0.08 * np.sin(2 * np.pi * 27.5  * t)
    ) * intensity * (0.80 + 0.20 * np.sin(2 * np.pi * 0.08 * t))
    rng = np.random.RandomState(42)
    shimmer = np.convolve(rng.normal(0, 1, n).astype(np.float32),
                          np.ones(max(1, int(SAMPLE_RATE * 0.001))) / max(1, int(SAMPLE_RATE * 0.001)),
                          mode="same") * 0.02 * intensity
    out = (drone + shimmer).astype(np.float32)
    fade = min(int(SAMPLE_RATE * 3), n // 4)
    out[:fade]  *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def _synth_sting() -> np.ndarray:
    dur = 2.0
    n   = int(SAMPLE_RATE * dur)
    t   = np.linspace(0, dur, n)
    body = (
        0.62 * np.sin(2 * np.pi * 40 * t) * np.exp(-t * 4.2) +
        0.42 * np.sin(2 * np.pi * 28 * t) * np.exp(-t * 2.8) +
        0.30 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 6.0)
    )
    rng = np.random.RandomState(99)
    click_len = int(SAMPLE_RATE * 0.008)
    click = rng.normal(0, 1, click_len) * np.exp(-np.linspace(0, 1, click_len) * 100)
    body[:click_len] += click.astype(np.float32) * 0.50
    max_val = np.max(np.abs(body))
    if max_val > 0:
        body = body / max_val * 0.92
    return body.astype(np.float32)


# ── ARC SEGMENT GROUPING ───────────────────────────────────────────────────────

def _group_arc_segments(edl_entries: list) -> list:
    """
    Collapse EDL into ~5 major arc blocks for MusicGen scoring.

    ONLY processes scene/cliffhanger entries.  Insert and reveal entries are
    brief B-roll/cutaway shots (1-3s) that inject foreign arc_position values
    (typically "investigation") between scene entries of the same arc, which
    shatters the grouping into 13+ micro-segments too short for MusicGen.
    """
    major_types = {"scene", "cliffhanger"}
    major = [e for e in edl_entries if e.get("type") in major_types]
    if not major:
        major = edl_entries  # last-resort fallback

    segments = []
    cur_arc = cur_loc = None
    cur_start = cur_end = 0
    for entry in major:
        arc = entry.get("arc_position", "build")
        loc = entry.get("location", cur_loc) or cur_loc or ""
        if arc != cur_arc:
            if cur_arc is not None and cur_end > cur_start:
                segments.append({"arc_position": cur_arc, "location": cur_loc,
                                  "start_ms": cur_start, "end_ms": cur_end,
                                  "duration_ms": cur_end - cur_start})
            cur_arc   = arc
            cur_start = entry.get("start_ms", cur_end)
            cur_loc   = loc
        cur_end = entry.get("end_ms", cur_end)
        if loc:
            cur_loc = loc
    if cur_arc is not None and cur_end > cur_start:
        segments.append({"arc_position": cur_arc, "location": cur_loc or "",
                          "start_ms": cur_start, "end_ms": cur_end,
                          "duration_ms": cur_end - cur_start})
    return segments


# ── EVENT ENVELOPE ─────────────────────────────────────────────────────────────

def _build_event_envelope(n: int, edl_entries: list) -> np.ndarray:
    """
    Per-sample gain envelope driven by EDL reveals and cliffhanger.
    Base: 0.65 (allows +54% headroom for swells — much more perceptible than 0.85→1.0).
    Reveals: 4.5s pre-swell to 1.0 → dip to 0.08 at cut → 2s recovery to 0.75.
    Cliffhanger: build → near-silence → sting fires → fade out.
    """
    env = np.ones(n, dtype=np.float32) * 0.65
    for rev in (e for e in edl_entries if e.get("type") == "reveal"):
        rev_s   = int(rev["start_ms"] * SAMPLE_RATE / 1000)
        swell_s = max(0, rev_s - int(4.5 * SAMPLE_RATE))
        peak_s  = max(swell_s, rev_s - int(0.6 * SAMPLE_RATE))
        if swell_s < peak_s <= n:
            env[swell_s:peak_s] = np.linspace(float(env[swell_s]), 1.0, peak_s - swell_s)
        sil_s = max(0, rev_s - int(0.6 * SAMPLE_RATE))
        sil_e = min(n, rev_s + int(1.2 * SAMPLE_RATE))
        # Floor at 0.08 — complete silence sounds like an audio cut, not a dip
        env[sil_s:sil_e] = 0.08
        rec_e = min(n, sil_e + int(1.8 * SAMPLE_RATE))
        if sil_e < rec_e:
            env[sil_e:rec_e] = np.linspace(0.08, 0.75, rec_e - sil_e)

    cliff = next((e for e in edl_entries if e.get("type") == "cliffhanger"), None)
    if cliff:
        cliff_s  = int(cliff["start_ms"] * SAMPLE_RATE / 1000)
        cliff_e  = min(n, int(cliff["end_ms"] * SAMPLE_RATE / 1000))
        swell_at = int(cliff.get("music_swell_at_ms", max(0, cliff["start_ms"] - 4000))
                       * SAMPLE_RATE / 1000)
        if swell_at < cliff_s:
            # Swell peaks 0.35s before the freeze — then near-silence creates the
            # tension dip right before the sting hits at cliff_s.
            swell_peak = max(swell_at, cliff_s - int(0.35 * SAMPLE_RATE))
            env[swell_at:swell_peak] = np.linspace(float(env[swell_at]), 1.15, swell_peak - swell_at)
            if swell_peak < cliff_s:
                env[swell_peak:cliff_s] = np.linspace(1.15, 0.08, cliff_s - swell_peak)
        if cliff_s < cliff_e:
            env[cliff_s:cliff_e] = np.linspace(0.08, 0.0, cliff_e - cliff_s)
        if cliff_e < n:
            env[cliff_e:] = 0.0

    k = max(1, int(SAMPLE_RATE * 0.04))
    return np.clip(np.convolve(env, np.ones(k) / k, mode="same"), 0.0, 1.0).astype(np.float32)


# ── SEGMENT CLIP GENERATION ───────────────────────────────────────────────────

def _generate_segment_clip(seg: dict, ep: int, seg_idx: int, episode_style: str = "") -> np.ndarray:
    """
    Generate a MusicGen clip for one arc segment using content-driven prompt.
    Cache key is a hash of the prompt text so re-runs with same content are instant.
    """
    prompt = _build_dynamic_prompt(
        narration_text         = seg.get("narration_text", ""),
        arc_position           = seg["arc_position"],
        story_beats_in_segment = seg.get("story_beats", []),
        duration_s             = seg["duration_ms"] / 1000.0,
        peak_at_s              = seg.get("peak_at_s"),
        episode_style          = episode_style,
    )

    cache_key  = hashlib.md5(prompt.encode()).hexdigest()[:8]
    cache_path = MUSIC_DIR / f"ep{ep}_seg{seg_idx}_{cache_key}.wav"

    if cache_path.exists():
        logger.info(f"Segment cache hit: {cache_path.name}")
        with wave.open(str(cache_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            n_ch   = wf.getnchannels()
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch).mean(axis=1)
        return pcm

    dur_s = min(seg["duration_ms"] / 1000.0, 30.0)
    audio = None

    if MUSIC_SOURCE == "kaggle":
        audio = _call_kaggle_music(prompt, dur_s)

    if audio is None and MUSICGEN_AVAILABLE:
        audio = _musicgen_local(prompt, dur_s)

    if audio is None:
        logger.info(f"Segment {seg_idx}: synth drone fallback")
        audio = _synth_drone(dur_s, seg["arc_position"])

    _save_wav(audio, str(cache_path))
    return audio


# ── SCORE ASSEMBLY ─────────────────────────────────────────────────────────────

def _assemble_score(
    segments: list,
    ep: int,
    total_s: float,
    edl_entries: list,
    xfade_s: float = 2.0,
    episode_style: str = "",
) -> np.ndarray:
    """
    Assemble full-episode score:
    1. Generate one MusicGen clip per arc segment (content-driven prompt)
    2. Loop clips to fill segment windows
    3. Crossfade between segments
    4. Apply reveal/cliffhanger event envelope
    """

    n_total = int(SAMPLE_RATE * total_s)
    score   = np.zeros(n_total, dtype=np.float32)
    xfade   = int(SAMPLE_RATE * xfade_s)

    for idx, seg in enumerate(segments):
        dur_s     = seg["duration_ms"] / 1000.0
        clip      = _generate_segment_clip(seg, ep, idx, episode_style=episode_style)
        seg_audio = _loop_to_duration(clip, dur_s)

        # Segment-level fade-in/out for crossfading between segments.
        # These are the ONLY fades — _loop_to_duration no longer adds its own.
        fi = min(xfade, len(seg_audio) // 3)
        fo = min(xfade, len(seg_audio) // 3)
        seg_audio[:fi]  *= np.linspace(0, 1, fi)
        seg_audio[-fo:] *= np.linspace(1, 0, fo)

        s = int(seg["start_ms"] * SAMPLE_RATE / 1000)
        e = min(s + len(seg_audio), n_total)
        if e > s:
            score[s:e] += seg_audio[:e - s]

    score = score * _build_event_envelope(n_total, edl_entries)

    max_val = np.max(np.abs(score))
    if max_val > 0:
        score = score / max_val * 0.82
    return score.astype(np.float32)


def _generate_sting(ep: int) -> str:
    path = str(MUSIC_DIR / f"ep{ep}_sting.wav")
    if not Path(path).exists():
        _save_wav(_synth_sting(), path)
    return path


# ── EPISODE STYLE DERIVATION ──────────────────────────────────────────────────

def _derive_episode_style(context: dict, ep_idx: int) -> str:
    """
    Build the shared MusicGen style prefix from story content so it's never
    hardcoded.  Every arc segment in the episode uses this prefix — one sonic
    palette throughout, not per-scene mood switching.

    Pulls from: story_title, story_genre, scene_prompts location/visual descriptions.
    Falls back to "cinematic thriller score" if nothing useful is available.
    """
    story_title = context.get("story_title", "")
    story_genre = context.get("story_genre", "")

    # Scan first few scene locations + visual briefs for atmosphere keywords
    scenes = (context.get("scene_prompts") or [[]])[ep_idx] if (
        context.get("scene_prompts") and ep_idx < len(context.get("scene_prompts", []))
    ) else []

    location_words = " ".join(
        f"{s.get('location_name', '')} {s.get('location', '')} {s.get('visual_brief', '')}"
        for s in scenes[:5]
    ).lower()

    # Map genre keywords → concrete MusicGen instrument/style palette.
    # Keywords are GENRE-level only — no story/episode-specific words.
    # Order matters: more specific genres take priority over generic ones.
    palette_rules = [
        (["sci-fi", "science fiction", "space", "future", "robot", "cyber", "android"],
         "Sci-fi thriller score. Atmospheric synthesizer pads, sparse electronic elements, minor key. Dark, cold, and vast."),
        (["horror", "supernatural", "ghost", "spirit", "creature", "demon", "occult"],
         "Horror score. Low string tremolo, dissonant sustained tones, minor key. Dark, oppressive, and suffocating."),
        (["historical", "period", "medieval", "victorian", "ancient", "renaissance", "colonial"],
         "Period thriller score. Solo cello, sparse chamber strings, low woodwinds. Minor key, dark, and archaic."),
        (["western", "frontier", "outlaw", "cowboy", "saloon", "desert"],
         "Western noir score. Sparse acoustic guitar, distant slide guitar, low strings. Desolate and suspenseful."),
        (["war", "military", "combat", "soldier", "battle", "espionage", "spy"],
         "Military thriller score. Low brass, tense strings, minor key. Urgent, heavy, and relentless."),
        (["nature", "wilderness", "survival", "forest", "jungle", "isolated"],
         "Dark nature thriller score. Solo piano, sparse ambient strings, held tones. Haunting and desolate."),
        (["psychological", "mind", "paranoia", "identity", "memory", "hallucination"],
         "Psychological thriller score. Solo piano, dissonant intervals, sparse strings. Unsettling and fragmented."),
        (["mystery", "detective", "crime", "murder", "investigation", "noir", "conspiracy"],
         "Crime noir score. Sparse solo piano in minor key, sustained cello, dark atmospheric strings. Tense and melancholic."),
        (["thriller", "suspense", "dark", "tension", "secret", "unknown"],
         "Cinematic thriller score. Solo piano, low sustained strings, sparse atmospheric texture. Dark and tense."),
    ]

    combined = f"{story_title} {story_genre} {location_words}"
    chosen = None
    for keywords, palette in palette_rules:
        if any(kw in combined for kw in keywords):
            chosen = palette
            break

    if chosen is None:
        chosen = "Cinematic thriller score. Solo piano, low strings, atmospheric texture. Minor key, dark and tense."

    return f"{chosen} Same instruments throughout — no genre shifts, no percussion beats. "


# ── MAIN ENTRY ─────────────────────────────────────────────────────────────────

def run_agent_7(context: dict) -> dict:
    """
    Generate music for each episode.
    Groups EDL into arc segments, enriches each with actual narration text
    and story beats, builds content-driven MusicGen prompts, assembles score.
    """
    logger.info(f"Music generation (source={MUSIC_SOURCE}, MusicGen={MUSICGEN_AVAILABLE})...")

    edl_list    = context.get("edl", [])
    context["music_tracks"] = []

    for ep_idx in range(context.get("total_episodes", 0)):
        ep_num = ep_idx + 1

        for suffix in ("score", "sting"):
            old = MUSIC_DIR / f"ep{ep_num}_{suffix}.wav"
            if old.exists():
                old.unlink()

        edl         = edl_list[ep_idx] if ep_idx < len(edl_list) else {}
        edl_entries = (edl or {}).get("entries", [])
        duration_s  = max((edl or {}).get("total_duration_ms", 90000) / 1000, 30.0)

        def _ep(key, default):
            lst = context.get(key, [])
            return lst[ep_idx] if ep_idx < len(lst) else default

        structured_narration = _ep("structured_narration", [])
        voiceover_timestamps = _ep("voiceover_timestamps", [])
        reveal_moments       = _ep("reveal_moments", [])

        n_reveals = sum(1 for e in edl_entries if e.get("type") == "reveal")
        cliff_at  = next((e["start_ms"] / 1000 for e in edl_entries if e.get("type") == "cliffhanger"), 0)
        logger.info(f"Episode {ep_num}: {duration_s:.1f}s, {n_reveals} reveals, cliffhanger at {cliff_at:.1f}s")

        try:
            episode_style = _derive_episode_style(context, ep_idx)
            logger.info(f"Episode {ep_num} style: {episode_style[:80]}...")

            segments = _group_arc_segments(edl_entries)
            segments = _enrich_segments(segments, structured_narration, voiceover_timestamps, reveal_moments)

            logger.info(
                "Arc segments: " +
                ", ".join(f"{s['arc_position']}({s['duration_ms']//1000}s)" for s in segments)
            )

            score_audio = _assemble_score(segments, ep_num, duration_s, edl_entries,
                                          episode_style=episode_style)

            score_path = str(MUSIC_DIR / f"ep{ep_num}_score.wav")
            _save_wav(score_audio, score_path)

            context["music_tracks"].append({
                "score": score_path,
                "sting": _generate_sting(ep_num),
            })
            logger.info(f"Episode {ep_num} music complete")

        except Exception as e:
            logger.error(f"Episode {ep_num} music failed: {e}")
            context.setdefault("warnings", []).append(
                {"agent": "music_agent", "episode": ep_num, "warning": str(e)}
            )
            context["music_tracks"].append({})

    return context
