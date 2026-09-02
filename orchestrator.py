import json
import os
import sys
from pathlib import Path

# Force UTF-8 output so box-drawing and check-mark characters don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime

from context import create_context
from agents.story_architect import plan_story
from agents.screenplay_agent import generate_script, summarize_episode
from agents.narration_director import run_narration_director
from agents.scene_director import generate_scene_prompts
from agents.voiceover_agent import generate_voiceover
from agents.agent_5a_image_generator import run_agent_5a
from agents.agent_6a_edit_planner import run_agent_6a
from agents.agent_6b_video_compositor import run_agent_6b
from agents.agent_7_music_agent import run_agent_7
from agents.agent_music_analyzer import run_music_analyzer
from agents.agent_event_stream import run_event_stream
from agents.agent_image_analyzer import run_image_analyzer
from agents.agent_video_generator import run_agent_video_generator
from utils.logger import setup_logger, AgentLogger
from utils.errors import FilmFactoryException

main_logger = setup_logger("Orchestrator")

# Set True to run only the first episode — saves API costs while fixing/testing
SINGLE_EPISODE_MODE = True


# ── PIPELINE GATE UTILITIES ───────────────────────────────────────────────────
#
# Architecture (two-machine split):
#
#   LOCAL  (Claude API)    │  GPU    (Colab A100)
#   ───────────────────────┼──────────────────────────────────────────────
#   1  Story Architect     │  4   Voiceover Agent   → WAV files
#   2  Screenplay Writer   │  WA  Whisper Alignment → word timestamps
#   4a Narration Director  │  5a  Image Generator   → FLUX scene images
#   3  Scene Director      │  VG  Video Generator   → Wan2.1-I2V clips
#                          │  7   Music Agent       → MusicGen score
#                          │  MA  Music Analyzer    → beat grid (librosa)
#                          │  ES  Event Stream      → unified timeline
#                          │  6a  Edit Planner      → beat-snapped EDL
#                          │  6b  Video Compositor  → final MP4s
#
# Gate rules:
#   HARD gate  — missing output raises PipelineGateError, stops the run
#   SOFT gate  — missing output logs a warning, stage is skipped gracefully
#
# Compositor is the strictest gate: audio + timestamps + images + EDL must
# ALL be present before a single frame is rendered.

class PipelineGateError(Exception):
    """Raised when a required upstream output is missing."""


def _count_images(context: dict) -> tuple:
    """Returns (scene_count, reveal_count, total) for images that exist on disk."""
    scene = sum(
        sum(1 for s in ep if s.get("image_path") and Path(s["image_path"]).exists())
        for ep in context.get("scene_prompts", [])
    )
    reveal = sum(
        sum(1 for r in ep if r.get("image_path") and Path(r["image_path"]).exists())
        for ep in context.get("reveal_moments", [])
    )
    return scene, reveal, scene + reveal


def _gate_narration(context: dict) -> None:
    """structured_narration must exist (Narration Director output)."""
    if not context.get("structured_narration") or not any(context["structured_narration"]):
        raise PipelineGateError(
            "structured_narration missing — Narration Director (4a) must run first"
        )


def _gate_audio(context: dict) -> None:
    """Audio WAV files must exist on disk for every episode."""
    audio_files = context.get("audio_files", [])
    if not audio_files:
        raise PipelineGateError(
            "audio_files missing from context.\n"
            "  → Run Voiceover Agent (4), or copy WAVs from Colab TTS output"
        )
    missing = [f for f in audio_files if not Path(f).exists()]
    if missing:
        raise PipelineGateError(
            f"Audio WAV not on disk: {missing[0]}\n"
            "  → If VOICE_SOURCE=colab: copy outputs/audio/episode_N_voiceover.wav from Drive"
        )


def _gate_timestamps(context: dict) -> None:
    """voiceover_timestamps must exist (Whisper alignment output)."""
    ts = context.get("voiceover_timestamps", [])
    if not ts or not any(ts):
        raise PipelineGateError(
            "voiceover_timestamps empty — run Whisper Alignment (WA) after voiceover.\n"
            "  Requires audio WAVs on disk."
        )


def _gate_images(context: dict) -> None:
    """At least one scene image must exist on disk."""
    total_slots = sum(len(ep) for ep in context.get("scene_prompts", []))
    if total_slots == 0:
        raise PipelineGateError("scene_prompts empty — run Scene Director (3) first")
    _, _, found = _count_images(context)
    if found == 0:
        raise PipelineGateError(
            f"No scene images on disk ({total_slots} slots defined).\n"
            "  → Run Image Generator (5a): set IMAGE_SOURCE=kaggle + KAGGLE_IMAGE_URL"
        )
    pct = found / total_slots
    if pct < 0.5:
        main_logger.warning(
            f"Only {found}/{total_slots} images on disk ({pct*100:.0f}%) — "
            f"compositor will Ken-Burns the missing slots"
        )


def _gate_edl(context: dict) -> None:
    """At least one EDL with entries must be built (Edit Planner output)."""
    built = sum(1 for e in context.get("edl", []) if e and e.get("entries"))
    if built == 0:
        raise PipelineGateError(
            "No EDLs built — Edit Planner (6a) must complete before compositor.\n"
            "  Needs: voiceover_timestamps + scene_prompts + unified_events"
        )


def print_readiness(context: dict) -> None:
    """
    Pre-compositor readiness checklist.
    ✓ / ✗  — required outputs (compositor blocked if any are ✗)
    ◎ / ○  — optional outputs (compositor degrades gracefully if missing)
    """
    print("\n  ┌─── Pipeline Readiness ──────────────────────────────────────┐")

    audio        = context.get("audio_files", [])
    audio_ok     = bool(audio) and all(Path(f).exists() for f in audio)
    n_audio      = sum(1 for f in audio if Path(f).exists())
    print(f"  │ {'✓' if audio_ok else '✗'} Audio WAVs:          {n_audio}/{len(audio)} episodes on disk")

    ts           = context.get("voiceover_timestamps", [])
    ts_ok        = bool(ts) and any(ts)
    n_ts         = sum(len(t) for t in ts)
    print(f"  │ {'✓' if ts_ok else '✗'} Chunk timestamps:    {n_ts} chunks (Whisper ±10 ms)")

    scene, reveal, total_imgs = _count_images(context)
    total_slots  = sum(len(ep) for ep in context.get("scene_prompts", []))
    img_ok       = total_imgs > 0
    pct          = total_imgs / total_slots * 100 if total_slots else 0
    print(f"  │ {'✓' if img_ok else '✗'} FLUX scene images:   {scene} scene + {reveal} reveal  ({pct:.0f}% of {total_slots} slots)")

    clips = sum(
        sum(1 for s in ep if s.get("video_path") and Path(s["video_path"]).exists())
        for ep in context.get("scene_prompts", [])
    )
    print(f"  │ {'◎' if clips else '○'} Wan video clips:     {clips}  (optional — Ken Burns used for remainder)")

    music        = context.get("music_tracks", [])
    music_ok     = any(music)
    print(f"  │ {'◎' if music_ok else '○'} MusicGen score:      {sum(1 for m in music if m)}/{len(music)} episodes  (optional)")

    events       = context.get("unified_events", [])
    event_ok     = any(events)
    n_ev         = sum(len(e.get("events", [])) for e in events if e)
    print(f"  │ {'◎' if event_ok else '○'} Unified event stream:{n_ev} events (subtitles + beat grid)")

    edls         = context.get("edl", [])
    edl_count    = sum(1 for e in edls if e and e.get("entries"))
    edl_ok       = edl_count > 0
    n_entries    = sum(len(e.get("entries", [])) for e in edls if e)
    print(f"  │ {'✓' if edl_ok else '✗'} EDLs built:          {edl_count}/{len(edls)} episodes  ({n_entries} entries total)")

    all_ok = audio_ok and ts_ok and img_ok and edl_ok
    if all_ok:
        print("  │")
        print("  │  ✓ COMPOSITOR CLEARED — all required outputs present")
    else:
        missing = [n for ok, n in [
            (audio_ok, "audio WAVs"),
            (ts_ok,    "Whisper timestamps"),
            (img_ok,   "scene images"),
            (edl_ok,   "EDL"),
        ] if not ok]
        print("  │")
        print(f"  │  ✗ COMPOSITOR BLOCKED — missing: {', '.join(missing)}")
    print("  └────────────────────────────────────────────────────────────┘\n")


def ensure_directories():
    """Create all required output directories."""
    dirs = [
        "outputs",
        "outputs/anchors",
        "outputs/audio",
        "outputs/scene_images",
        "outputs/reveal_images",
        "outputs/insert_images",
        "outputs/depth_maps",
        "outputs/video_clips",
        "outputs/music",
        "outputs/final",
        "outputs/alignment",
        "logs",
        "memory",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    main_logger.info("Output directories verified")


def save_context(context: dict, filename: str = "context.json"):
    """Save context to JSON for inspection and debugging."""
    try:
        output_path = Path("outputs") / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2)
        main_logger.info(f"Context saved: {output_path}")
    except Exception as e:
        main_logger.error(f"Failed to save context: {e}")


def print_summary(context: dict):
    """Print pipeline execution summary."""
    print("\n" + "=" * 65)
    print("  FILM FACTORY — PIPELINE SUMMARY")
    print("=" * 65)

    print(f"\n  Story:     {context.get('story_title', 'N/A')}")
    print(f"  Episodes:  {context.get('total_episodes', 0)}")
    print(f"  Scripts:   {len(context.get('scripts', []))}")

    # Scene stats
    scene_total = sum(len(ep) for ep in context.get("scene_prompts", []))
    print(f"  Scenes:    {scene_total} total")

    # Reveal stats
    reveal_total = sum(len(ep) for ep in context.get("reveal_moments", []))
    print(f"  Reveals:   {reveal_total} total")

    # Narration chunks
    chunk_total = sum(len(ep) for ep in context.get("structured_narration", []))
    print(f"  Narration chunks: {chunk_total}")

    # Audio
    print(f"  Audio files: {len(context.get('audio_files', []))}")

    # Images
    anchor_count = len(context.get("story_bible", {}).get("anchor_images", {}))
    scene_images = sum(
        sum(1 for s in ep if s.get("image_path"))
        for ep in context.get("scene_prompts", [])
    )
    reveal_images = sum(
        sum(1 for r in ep if r.get("image_path"))
        for ep in context.get("reveal_moments", [])
    )
    print(f"  Images: {anchor_count} anchors, {scene_images} scenes, {reveal_images} reveals")

    # EDL
    edl_count = sum(1 for e in context.get("edl", []) if e and e.get("entries"))
    print(f"  EDLs built: {edl_count}")

    # Final episodes
    final = context.get("final_episodes", [])
    print(f"  Final episodes: {len(final)}")
    for path in final:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1_000_000
            print(f"    ✓ {path} ({size_mb:.1f} MB)")
        else:
            print(f"    ✗ {path} (file not found)")

    if context.get("errors"):
        print(f"\n  Errors: {len(context['errors'])}")
        for err in context["errors"][:5]:
            print(f"    ✗ [{err.get('agent', '?')}] {err.get('error', '')[:80]}")

    if context.get("warnings"):
        print(f"\n  Warnings: {len(context['warnings'])}")
        for w in context["warnings"][:3]:
            print(f"    ! [{w.get('agent', '?')}] {w.get('warning', '')[:60]}")

    print("\n" + "=" * 65 + "\n")


def run_pipeline(story_idea: str, skip_to: str = None):
    """
    Run the complete cinematic narration pipeline.

    Agent execution order:
    ┌─────────────────────────────────────────────────────────┐
    │  1   Story Architect  → story bible, episode plan       │
    │  2   Screenplay Writer → episode scripts                │
    │  4a  Narration Director → structured narration chunks   │
    │  3   Scene Director   → image prompts + reveal moments  │
    │  4   Voiceover Agent  → audio                           │
    │  WA  Whisper Alignment → word-level timestamps (±10ms)  │
    │  5a  Image Generator  → anchors → scenes + reveals      │
    │  7   Music Agent      → dynamic MusicGen score          │
    │  MA  Music Analyzer   → beat grid + onset map (librosa) │
    │  ES  Event Stream     → unified timeline (all layers)   │
    │  6a  Edit Planner     → beat-snapped frame-accurate EDL │
    │  6b  Video Compositor → final rendered MP4s             │
    └─────────────────────────────────────────────────────────┘

    IMPORTANT: Agent 4a runs BEFORE Agent 3.
    Agent 3 needs structured_narration to design reveal_moments
    (it uses chunk_index to tag which spoken sentence triggers each cut).
    Music Agent runs BEFORE Edit Planner so the beat grid is available
    for cut-point snapping via the Event Stream.

    Args:
        story_idea: One-line story concept
        skip_to: Optional agent name to start from (for resuming)

    Returns:
        Complete context dict
    """
    main_logger.info("=" * 65)
    main_logger.info("FILM FACTORY — CINEMATIC NARRATION PIPELINE")
    main_logger.info("=" * 65)

    try:
        ensure_directories()
        context = create_context(story_idea)
        main_logger.info(f"Story idea: {story_idea}")

        # ── AGENT 1: Story Architect ───────────────────────────────────
        print("\n═══ Agent 1: Story Architect ═══")
        try:
            context, story_plan = plan_story(context)
            if SINGLE_EPISODE_MODE and context["total_episodes"] > 1:
                context["total_episodes"] = 1
                main_logger.info("SINGLE_EPISODE_MODE: running episode 1 only")
            print(f"  ✓ {context['story_title']} — {context['total_episodes']} episodes")
            print(f"  ✓ {len(context['story_bible'].get('location_visuals', {}))} locations designed")
        except FilmFactoryException as e:
            main_logger.error(f"Story Architect failed: {e}")
            raise

        # ── AGENT 2: Screenplay Writer ─────────────────────────────────
        print("\n═══ Agent 2: Screenplay Writer ═══")
        try:
            for i in range(context["total_episodes"]):
                print(f"  Writing episode {i + 1}...")
                script = generate_script(context, i)
                context["scripts"].append(script)
                print(f"  ✓ Episode {i + 1}: {len(script.split())} words")

                if i < context["total_episodes"] - 1:
                    summary = summarize_episode(script, i)
                    context["episode_summaries"].append(summary)
        except FilmFactoryException as e:
            main_logger.error(f"Screenplay Writer failed: {e}")
            raise

        # ── AGENT 4a: Narration Director ──────────────────────────────
        # MUST run before Agent 3 — scene director needs chunk indices
        # to design reveal moments synced to specific spoken words
        print("\n═══ Agent 4a: Narration Director ═══")
        try:
            context = run_narration_director(context)
            total_chunks = sum(len(ep) for ep in context.get("structured_narration", []))
            print(f"  ✓ {total_chunks} narration chunks across {context['total_episodes']} episodes")
        except Exception as e:
            main_logger.error(f"Narration Director failed: {e}")
            raise PipelineGateError(f"Narration Director failed — cannot proceed without structured narration: {e}")

        # ── COLAB EXPORT: structured chunks for GPU notebook ──────────
        # Only needed when using the Colab TTS workflow
        from agents.voiceover_agent import VOICE_SOURCE as _VS
        if _VS == "colab":
            colab_input_dir = Path("outputs/colab_input")
            colab_input_dir.mkdir(parents=True, exist_ok=True)
            for ep_idx, chunks in enumerate(context.get("structured_narration", [])):
                chunks_path = colab_input_dir / f"episode_{ep_idx + 1}_chunks.json"
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump({"episode": ep_idx + 1, "chunks": chunks}, f, indent=2)
            print(
                f"  ✓ Colab input exported: "
                f"{len(context.get('structured_narration', []))} episode(s) → "
                f"outputs/colab_input/"
            )

        # ── AGENT 3: Scene Director ────────────────────────────────────
        print("\n═══ Agent 3: Scene Director ═══")
        try:
            context["scene_prompts"] = []
            # reveal_moments initialized inside generate_scene_prompts
            for i in range(context["total_episodes"]):
                print(f"  Directing episode {i + 1}...")
                prompts = generate_scene_prompts(context, i)
                context["scene_prompts"].append(prompts)

                reveals = context.get("reveal_moments", [[]] * (i + 1))
                ep_reveals = reveals[i] if i < len(reveals) else []

                est_secs = sum(p.get("duration_seconds", 8) for p in prompts)
                print(
                    f"  ✓ Episode {i + 1}: "
                    f"{len(prompts)} scenes, "
                    f"{len(ep_reveals)} reveals, "
                    f"~{est_secs}s"
                )
        except FilmFactoryException as e:
            main_logger.error(f"Scene Director failed: {e}")
            raise

        # Save checkpoint
        save_context(context, "context_after_scene_director.json")

        # ── COLAB WAIT: pause until voice WAVs appear on disk ─────────
        from agents.voiceover_agent import VOICE_SOURCE
        if VOICE_SOURCE == "colab":
            import time as _time
            missing = []
            for i in range(context["total_episodes"]):
                wav = Path(f"outputs/audio/episode_{i + 1}_voiceover.wav")
                if not wav.exists():
                    missing.append(str(wav))

            if missing:
                print("\n═══ Waiting for Colab voice output ═══")
                print("  Missing WAV files:")
                for m in missing:
                    print(f"    {m}")
                print("\n  → Open film_factory_gpu.ipynb on Colab")
                print("  → Set RUN_VOICE=True, run all cells")
                print("  → Drive desktop app will sync the WAV back automatically")
                print("  → This window will continue once the files appear...\n")

                while missing:
                    _time.sleep(15)
                    missing = [
                        m for m in missing
                        if not Path(m).exists()
                    ]
                    if missing:
                        print(f"  Still waiting for {len(missing)} file(s)...")
                    else:
                        print("  ✓ All Colab voice files detected — resuming pipeline")

        # ── AGENT 4: Voiceover Agent ───────────────────────────────────
        print("\n═══ Agent 4: Voiceover Agent ═══")
        try:
            for i in range(context["total_episodes"]):
                print(f"  Generating voiceover episode {i + 1}...")
                audio_path = generate_voiceover(context, i)
                if audio_path:
                    ts = context.get("voiceover_timestamps", [[]] * (i + 1))
                    ep_ts = ts[i] if i < len(ts) else []
                    total_ms = ep_ts[-1]["end_ms"] if ep_ts else 0
                    print(
                        f"  ✓ Episode {i + 1}: {audio_path} "
                        f"({total_ms / 1000:.1f}s, {len(ep_ts)} chunks timed)"
                    )
        except FilmFactoryException as e:
            main_logger.error(f"Voiceover Agent failed: {e}")
            context["warnings"].append({"agent": "voiceover", "warning": str(e)})

        # ── WHISPER ALIGNMENT ─────────────────────────────────────────
        # GATE: audio files must be on disk before Whisper can align them
        print("\n═══ Whisper: Word Alignment ═══")
        _gate_audio(context)
        from agents.agent_whisper_alignment import run_whisper_alignment, derive_chunk_timestamps_from_words
        context = run_whisper_alignment(context)
        for ep_idx, words in enumerate(context.get('word_timestamps', [])):
            audio_path = context['audio_files'][ep_idx]
            n_chunks = len(context['structured_narration'][ep_idx])
            real_ts = derive_chunk_timestamps_from_words(words, n_chunks, audio_path)
            context['voiceover_timestamps'][ep_idx] = real_ts
            print(f"  ✓ Episode {ep_idx+1}: {len(words)} words aligned, {n_chunks} chunks timed")

        # ── AGENT 5a: Image Generator ──────────────────────────────────
        print("\n═══ Agent 5a: Image Generator ═══")
        try:
            context = run_agent_5a(context)
            anchors = len(context.get("story_bible", {}).get("anchor_images", {}))
            scene_imgs = sum(
                sum(1 for s in ep if s.get("image_path"))
                for ep in context.get("scene_prompts", [])
            )
            reveal_imgs = sum(
                sum(1 for r in ep if r.get("image_path"))
                for ep in context.get("reveal_moments", [])
            )
            insert_imgs = sum(
                sum(
                    1 for ins in s.get("insert_shots", [])
                    if ins.get("image_path")
                )
                for ep in context.get("scene_prompts", [])
                for s in ep
            )
            print(
                f"  ✓ {anchors} anchors, "
                f"{scene_imgs} scene images, "
                f"{reveal_imgs} reveals, "
                f"{insert_imgs} insert shots"
            )
        except FilmFactoryException as e:
            main_logger.error(f"Image Generator failed: {e}")
            context["warnings"].append({"agent": "image_generator", "warning": str(e)})

        # Save checkpoint
        save_context(context, "context_after_images.json")

        # ── IMAGE ANALYZER: per-image adaptive grade ───────────────────
        print("\n═══ Image Analyzer (adaptive grade) ═══")
        try:
            context = run_image_analyzer(context)
            adapted = sum(
                sum(1 for s in ep if s.get("adaptive_grade"))
                for ep in context.get("scene_prompts", [])
            ) + sum(
                sum(1 for r in ep if r.get("adaptive_grade"))
                for ep in context.get("reveal_moments", [])
            )
            print(f"  ✓ {adapted} images analyzed, adaptive grade attached")
        except Exception as e:
            main_logger.warning(f"Image analyzer failed (non-fatal): {e}")
            print(f"  Skipped (non-fatal): {e}")

        # ── VIDEO GENERATOR: Wan2.1-I2V clips (optional, Kaggle) ─────────
        # Only runs when VIDEO_SOURCE=kaggle and KAGGLE_VIDEO_URL is set.
        # Skips silently if not configured — Ken Burns is used instead.
        print("\n═══ Video Generator (Wan2.1-I2V) ═══")
        try:
            context = run_agent_video_generator(context)
            video_clips = sum(
                sum(1 for s in ep if s.get("video_path"))
                for ep in context.get("scene_prompts", [])
            ) + sum(
                sum(1 for r in ep if r.get("video_path"))
                for ep in context.get("reveal_moments", [])
            )
            if video_clips:
                print(f"  ✓ {video_clips} video clips generated")
                save_context(context, "context_after_video_gen.json")
            else:
                print("  Skipped — set VIDEO_SOURCE=kaggle + KAGGLE_VIDEO_URL to enable")
        except Exception as e:
            print(f"  Video generation skipped (non-fatal): {e}")

        # ── AGENT 7: Music Agent ───────────────────────────────────────
        # Runs BEFORE Edit Planner — builds the score from narration text so
        # the beat grid is ready for Event Stream snapping.
        # GATE: Whisper timestamps must exist (needed for score duration + scene timing)
        print("\n═══ Agent 7: Music Agent ═══")
        _gate_timestamps(context)
        try:
            context = run_agent_7(context)
            tracks = context.get("music_tracks", [])
            print(f"  ✓ {len(tracks)} episode music sets generated")
            for i, t in enumerate(tracks):
                if t:
                    print(f"    Episode {i + 1}: dynamic score generated")
        except Exception as e:
            main_logger.error(f"Music Agent failed: {e}")
            context["warnings"].append({"agent": "music_agent", "warning": str(e)})

        # ── MUSIC ANALYZER ─────────────────────────────────────────────
        print("\n═══ Music Analyzer (librosa) ═══")
        try:
            context = run_music_analyzer(context)
            analyses = context.get("music_analysis", [])
            print(f"  ✓ {len(analyses)} tracks analyzed")
            for i, a in enumerate(analyses):
                if a:
                    print(
                        f"    Episode {i + 1}: {a.get('tempo_bpm', 0):.1f} BPM, "
                        f"{len(a.get('beat_times_ms', []))} beats, "
                        f"{len(a.get('strong_onset_times_ms', []))} strong onsets"
                    )
        except Exception as e:
            main_logger.error(f"Music Analyzer failed: {e}")
            context["warnings"].append({"agent": "music_analyzer", "warning": str(e)})

        # ── EVENT STREAM ────────────────────────────────────────────────
        print("\n═══ Event Stream (unified timeline) ═══")
        try:
            context = run_event_stream(context)
            streams = context.get("unified_events", [])
            print(f"  ✓ {len(streams)} unified event streams built")
            for i, s in enumerate(streams):
                if s:
                    print(
                        f"    Episode {i + 1}: {len(s.get('events', []))} events, "
                        f"{len(s.get('cut_points', []))} cut points, "
                        f"{len(s.get('subtitle_groups', []))} subtitle groups"
                    )
        except Exception as e:
            main_logger.error(f"Event Stream failed: {e}")
            context["warnings"].append({"agent": "event_stream", "warning": str(e)})

        # ── AGENT 6a: Edit Planner ─────────────────────────────────────
        # Runs AFTER Event Stream — reads unified_events for beat-snapped cut points.
        # GATE: images + timestamps must be ready before EDL can be built
        print("\n═══ Agent 6a: Edit Planner ═══")
        _gate_images(context)
        _gate_timestamps(context)
        try:
            context = run_agent_6a(context)
            edl_built = sum(1 for e in context.get("edl", []) if e and e.get("entries"))
            reveal_planned = sum(
                e.get("reveal_count", 0)
                for e in context.get("edl", [])
                if e
            )
            print(f"  ✓ {edl_built} EDLs built, {reveal_planned} reveal cuts planned")
            for i, edl in enumerate(context.get("edl", [])):
                if edl and edl.get("entries"):
                    print(
                        f"    Episode {i + 1}: "
                        f"{edl['entry_count']} entries, "
                        f"{edl['total_duration_ms'] / 1000:.1f}s"
                    )
        except Exception as e:
            main_logger.error(f"Edit Planner failed: {e}")
            context["errors"].append({"agent": "edit_planner", "error": str(e)})

        # Save checkpoint after music + event stream + planner
        save_context(context, "context_after_edit_plan.json")

        # ── AGENT 6b: Video Compositor ─────────────────────────────────
        # HARD GATE: all four required outputs must be present before rendering.
        # Wan clips + music + event stream are optional — compositor degrades gracefully.
        print_readiness(context)
        _gate_audio(context)
        _gate_timestamps(context)
        _gate_images(context)
        _gate_edl(context)
        print("\n=== Agent 6b: Video Compositor ===")
        try:
            context = run_agent_6b(context)
            final = context.get("final_episodes", [])
            print(f"  ✓ {len(final)} episodes rendered")
            for path in final:
                if Path(path).exists():
                    size_mb = Path(path).stat().st_size / 1_000_000
                    print(f"    ✓ {path} ({size_mb:.1f} MB)")
                else:
                    print(f"    ✗ {path} — file not found")
        except Exception as e:
            main_logger.error(f"Video Compositor failed: {e}")
            context["errors"].append({"agent": "video_compositor", "error": str(e)})

        # ── FINAL SAVE ─────────────────────────────────────────────────
        save_context(context)
        print_summary(context)

        return context

    except PipelineGateError as e:
        main_logger.error(f"Pipeline gate blocked: {e}")
        print(f"\n  ✗ GATE: {e}\n")
        sys.exit(1)
    except FilmFactoryException as e:
        main_logger.error(f"Pipeline failed: {e}")
        print(f"\n  ✗ Pipeline failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        main_logger.error(f"Unexpected error: {e}")
        print(f"\n  ✗ Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Film Factory pipeline")
    parser.add_argument("--resume", metavar="CHECKPOINT",
                        nargs="?", const="outputs/context.json",
                        help="Load a context checkpoint and re-run from the video compositor. "
                             "Defaults to outputs/context.json if no path given.")
    parser.add_argument("--resume-from", metavar="AGENT", default="compositor",
                        help="With --resume, which agent to start from (default: compositor)")
    args = parser.parse_args()

    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.exists():
            print(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)
        with open(checkpoint, encoding="utf-8") as f:
            context = json.load(f)
        print(f"Loaded checkpoint: {checkpoint}")
        resume_from = args.resume_from.lower() if args.resume_from else "compositor"
        print(f"Story: {context.get('story_title', '?')} - resuming from: {resume_from}")

        # Always patch audio when VOICE_SOURCE=colab
        if os.getenv("VOICE_SOURCE") == "colab":
            ep_num = 1
            colab_wav = Path(f"outputs/audio/episode_{ep_num}_voiceover.wav")
            if colab_wav.exists():
                wav_str = str(colab_wav)
                context["audio_files"] = [wav_str]
                for edl_entry in context.get("edl", []):
                    if not edl_entry.get("audio_path"):
                        edl_entry["audio_path"] = wav_str
                print(f"  Patched audio_files -> {colab_wav} ({colab_wav.stat().st_size // 1024} KB)")

        # ── RESUME FROM IMAGES: re-gen missing scene images + layers + motion ──
        if resume_from in ("images", "image"):
            print("\n=== Agent 5a: Image Generator (missing images only) ===")
            try:
                context = run_agent_5a(context)
                print("  Done: image generation complete")
            except Exception as e:
                print(f"  Image Generator failed: {e}")
                sys.exit(1)

            print("\n=== Video Generator (Wan2.1-I2V) ===")
            try:
                context = run_agent_video_generator(context)
                save_context(context, "context_after_video_gen.json")
            except Exception as e:
                print(f"  Video generation skipped: {e}")

            print("\n=== Image Analyzer (adaptive grade) ===")
            try:
                context = run_image_analyzer(context)
                adapted = sum(
                    sum(1 for s in ep if s.get("adaptive_grade"))
                    for ep in context.get("scene_prompts", [])
                ) + sum(
                    sum(1 for r in ep if r.get("adaptive_grade"))
                    for ep in context.get("reveal_moments", [])
                )
                print(f"  Done: {adapted} images with adaptive grade")
            except Exception as e:
                print(f"  Image analyzer skipped (non-fatal): {e}")

            resume_from = "whisper"  # fall through to whisper → 6a → 6b

        # ── RESUME FROM WHISPER: align new voice + rebuild EDL ─────────────────
        if resume_from in ("whisper", "images"):
            # GATE: audio files must exist before Whisper can run
            try:
                _gate_audio(context)
            except PipelineGateError as e:
                print(f"\n  ✗ GATE FAIL (Whisper): {e}")
                sys.exit(1)
            print("\n=== Whisper: Word Alignment ===")
            from agents.agent_whisper_alignment import run_whisper_alignment, derive_chunk_timestamps_from_words
            context["word_timestamps"] = []
            context["voiceover_timestamps"] = []
            context = run_whisper_alignment(context)
            for ep_idx, words in enumerate(context.get("word_timestamps", [])):
                audio_path = context["audio_files"][ep_idx]
                n_chunks = len(context["structured_narration"][ep_idx])
                real_ts = derive_chunk_timestamps_from_words(words, n_chunks, audio_path)
                context["voiceover_timestamps"].append(real_ts)
                print(f"  Episode {ep_idx+1}: {len(words)} words aligned, {n_chunks} chunks timed")

            resume_from = "music"  # fall through to music → analyzer → event_stream → planner

        # ── RESUME FROM MUSIC: re-gen score + rebuild event stream + EDL ─────────
        if resume_from in ("music", "planner", "whisper"):
            print("\n=== Agent 7: Music Agent ===")
            try:
                context = run_agent_7(context)
                tracks = context.get("music_tracks", [])
                print(f"  Done: {len(tracks)} episode music sets generated")
            except Exception as e:
                print(f"  Music Agent failed (non-fatal): {e}")
                context.setdefault("warnings", []).append({"agent": "music_agent", "warning": str(e)})

            print("\n=== Music Analyzer (librosa) ===")
            try:
                context = run_music_analyzer(context)
                print(f"  Done: {len(context.get('music_analysis', []))} tracks analyzed")
            except Exception as e:
                print(f"  Music Analyzer failed (non-fatal): {e}")
                context.setdefault("warnings", []).append({"agent": "music_analyzer", "warning": str(e)})

            print("\n=== Event Stream (unified timeline) ===")
            try:
                context = run_event_stream(context)
                print(f"  Done: {len(context.get('unified_events', []))} event streams built")
            except Exception as e:
                print(f"  Event Stream failed (non-fatal): {e}")
                context.setdefault("warnings", []).append({"agent": "event_stream", "warning": str(e)})

            print("\n=== Agent 6a: Edit Planner ===")
            context["edl"] = []
            try:
                context = run_agent_6a(context)
                edl_built = sum(1 for e in context.get("edl", []) if e and e.get("entries"))
                print(f"  Done: {edl_built} EDLs built")
                # Re-patch EDL audio_path after rebuild
                if os.getenv("VOICE_SOURCE") == "colab":
                    wav_str = context["audio_files"][0] if context.get("audio_files") else ""
                    for edl_entry in context.get("edl", []):
                        edl_entry["audio_path"] = wav_str
            except Exception as e:
                print(f"  Edit Planner failed: {e}")
                sys.exit(1)

        # ── COMPOSITOR — hard gate before any rendering ──────────────────────────
        print_readiness(context)
        try:
            _gate_audio(context)
            _gate_timestamps(context)
            _gate_images(context)
            _gate_edl(context)
        except PipelineGateError as e:
            print(f"\n  ✗ COMPOSITOR BLOCKED: {e}")
            print("  Fix the missing outputs above, then re-run with --resume\n")
            save_context(context)
            sys.exit(1)

        print("\n=== Agent 6b: Video Compositor ===")
        try:
            context = run_agent_6b(context)
            final = context.get("final_episodes", [])
            print(f"  Done: {len(set(final))} episodes rendered")
            for path in set(final):
                if Path(path).exists():
                    size_mb = Path(path).stat().st_size / 1_000_000
                    print(f"    {path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Video Compositor failed: {e}")
            sys.exit(1)
        save_context(context)
        print_summary(context)
    else:
        story_idea = input("Enter your story idea: ").strip()
        if not story_idea:
            print("Story idea cannot be empty.")
            sys.exit(1)
        run_pipeline(story_idea)