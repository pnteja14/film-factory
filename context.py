from pathlib import Path
from typing import Dict, List, Any


def create_context(story_idea: str) -> Dict[str, Any]:
    """
    Create a fresh context object for the cinematic narration pipeline.

    Context is the single shared state dict passed through all agents.
    Each agent reads from and writes to specific keys.

    Pipeline flow:
    Agent 1  → story_bible, characters, episodes
    Agent 2  → scripts, episode_summaries
    Agent 4a → structured_narration (before Agent 3 needs it for reveal design)
    Agent 3  → scene_prompts, reveal_moments
    Agent 4  → audio_files, voiceover_timestamps
    WhisperX → word_timestamps (phoneme-level, ±10ms)
    Agent 5a → story_bible.anchor_images, scene_prompts[].image_path,
               reveal_moments[].image_path
    Agent 7  → music_tracks (dynamic MusicGen score from narration text)
    MusicAnalyzer → music_analysis (beat grid + onsets via librosa)
    EventStream   → unified_events (all layers on one timeline)
    Agent 6a → edl (beat-snapped Edit Decision Lists)
    Agent 6b → final_episodes
    """
    # Create full output directory structure
    dirs = [
        "outputs",
        "outputs/anchors",
        "outputs/audio",
        "outputs/scene_images",
        "outputs/reveal_images",
        "outputs/music",
        "outputs/final",
        "logs",
        "memory",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    return {
        # ── INPUT ──────────────────────────────────────────────────────
        "story_idea": story_idea,

        # ── AGENT 1: Story Architect ───────────────────────────────────
        "story_title": "",
        "story_summary": "",
        "thematic_quote": "",
        "quote_author": "",
        "quote_source": "",
        "characters": [],
        "episodes": [],

        # Story Bible — visual/narrative consistency layer
        "story_bible": {
            "locations": [],
            "location_visuals": {},      # location_name → visual_description str
            "character_arcs": [],
            "emotional_beats": [],       # [{ episode, tone, key_moment }]
            "visual_style": {
                "color_grade": "dark desaturated muted tones",
                "film_style": "cinematic grain atmospheric",
                "lighting": "dim natural light deep shadows",
                "camera_feel": "slow deliberate first person POV",
                "mood": "tense melancholic unsettling"
            },
            "anchor_images": {}          # location_name → { image_path, seed, visual_description }
        },

        # ── AGENT 2: Screenplay Writer ─────────────────────────────────
        "scripts": [],                   # one raw script string per episode
        "episode_summaries": [],         # rolling summaries for forward context

        # ── AGENT 4a: Narration Director ──────────────────────────────
        # Must run BEFORE Agent 3 so reveal_moments can reference chunk_index
        "structured_narration": [],      # [[{type, text, instructions, pause_after_chunk}]]
                                         # index 0 = episode 1

        # ── AGENT 3: Scene Director ────────────────────────────────────
        "scene_prompts": [],             # [[scene_dict]] per episode
                                         # scene_dict keys:
                                         #   scene_number, image_prompt,
                                         #   arc_position, duration, duration_seconds,
                                         #   location, zoom_direction, is_padding,
                                         #   image_path (filled by 5a),
                                         #   prompt_corrected (filled by 5a.5)

        "reveal_moments": [],            # [[reveal_dict]] per episode
                                         # reveal_dict keys:
                                         #   chunk_index, trigger_phrase,
                                         #   image_type, image_brief, flux_prompt,
                                         #   cut_type, hold_seconds,
                                         #   text_overlay, return_to_scene,
                                         #   image_path (filled by 5a)

        # ── AGENT 4: Voiceover Agent ───────────────────────────────────
        "audio_files": [],               # paths to episode WAV files
        "voiceover_timestamps": [],      # [[{chunk_index, start_ms, end_ms, duration_ms, audio_file}]]
                                         # Agent 6a uses this to sync reveal cuts
        "word_timestamps": [],           # [[{word, start_ms, end_ms, chunk_index}]] per episode (WhisperX)

        # ── AGENT 5a: Image Generator ──────────────────────────────────
        # Output written into:
        #   story_bible["anchor_images"]
        #   scene_prompts[ep][scene]["image_path"]
        #   reveal_moments[ep][reveal]["image_path"]
        "anchor_image_descriptions": {}, # location_name → Claude vision description (from 5a.5)

        # ── AGENT 6a: Edit Planner ─────────────────────────────────────
        "edl": [],                       # [edl_dict] per episode
                                         # edl_dict keys:
                                         #   episode_number, audio_path, total_duration_ms,
                                         #   entry_count, reveal_count,
                                         #   entries: [{type, start_ms, end_ms, image_path,
                                         #              cut_type, zoom_rate, zoom_direction,
                                         #              text_overlay, desaturate, flash_frame,
                                         #              arc_position}]

        # ── AGENT 7: Music Agent ───────────────────────────────────────
        "music_tracks": [],              # [{score, sting}] per episode

        # ── Music Analyzer (librosa) ────────────────────────────────────
        "music_analysis": [],            # [{tempo_bpm, beat_times_ms, onset_times_ms,
                                         #   strong_onset_times_ms, section_times_ms,
                                         #   rms_times_ms, rms_values, duration_ms}]

        # ── Event Stream (unified timeline) ────────────────────────────
        "unified_events": [],            # [{episode, events, cut_points, story_beats,
                                         #   subtitle_groups, music_tempo_bpm}]

        # ── AGENT 6b: Video Compositor ────────────────────────────────
        "final_episodes": [],            # paths to rendered MP4 files

        # ── LEGACY / UNUSED IN NEW PIPELINE ───────────────────────────
        "video_clips": [],               # deprecated — was for AI video clips
        "music_file": "",                # deprecated — replaced by music_tracks

        # ── SYSTEM TRACKING ────────────────────────────────────────────
        "current_episode": 0,
        "total_episodes": 0,
        "status": "started",
        "errors": [],                    # [{ agent, episode?, error }]
        "warnings": [],                  # [{ agent, episode?, warning }]
    }