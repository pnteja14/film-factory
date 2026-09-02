"""
Agent Music Analyzer — extracts beat grid, onsets, and energy from generated music.

Runs post-MusicGen, feeds the unified event stream so all EDL cuts can be
snapped to real music events rather than arbitrary time boundaries.

Requires: pip install librosa
"""

import numpy as np
from pathlib import Path
from utils.logger import AgentLogger

logger = AgentLogger("MusicAnalyzer")


def analyze_music_track(audio_path: str, episode_number: int) -> dict:
    """
    Analyze a generated music WAV using librosa.
    Returns beat grid, onset peaks, energy envelope, and section boundaries.
    All timestamps are milliseconds.
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed — music analysis skipped. pip install librosa")
        return {}

    try:
        logger.info(f"Ep{episode_number}: analyzing {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration_ms = int(len(y) / sr * 1000)

        # Beat grid
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times_ms = [int(t) for t in librosa.frames_to_time(beat_frames, sr=sr) * 1000]

        # Onset peaks (where instruments attack)
        onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames", backtrack=True)
        onset_times_ms = [int(t) for t in librosa.frames_to_time(onset_frames, sr=sr) * 1000]

        # Strong onsets: top 20% by onset strength
        if len(onset_frames) > 0:
            strengths = onset_env[np.minimum(onset_frames, len(onset_env) - 1)]
            threshold = np.percentile(strengths, 80) if len(strengths) > 5 else 0.0
            strong_frames = onset_frames[strengths >= threshold]
            strong_onset_times_ms = [int(t) for t in librosa.frames_to_time(strong_frames, sr=sr) * 1000]
        else:
            strong_onset_times_ms = []

        # Energy envelope (RMS, ~10 Hz resolution for segment-level analysis)
        hop = sr // 10
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_times_ms = [int(i * hop / sr * 1000) for i in range(len(rms))]
        rms_values   = [float(v) for v in rms]

        # Section boundaries via spectral clustering
        try:
            mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
            n_sections = min(8, max(2, duration_ms // 15000))
            bound_frames  = librosa.segment.agglomerative(mfcc, n_sections)
            section_times_ms = [int(t) for t in librosa.frames_to_time(bound_frames, sr=sr) * 1000]
        except Exception:
            section_times_ms = []

        analysis = {
            "tempo_bpm":             float(tempo),
            "beat_times_ms":         beat_times_ms,
            "onset_times_ms":        onset_times_ms,
            "strong_onset_times_ms": strong_onset_times_ms,
            "section_times_ms":      section_times_ms,
            "rms_times_ms":          rms_times_ms,
            "rms_values":            rms_values,
            "duration_ms":           duration_ms,
        }
        logger.info(
            f"Ep{episode_number}: {tempo:.1f} BPM, "
            f"{len(beat_times_ms)} beats, "
            f"{len(strong_onset_times_ms)} strong onsets, "
            f"{len(section_times_ms)} sections"
        )
        return analysis

    except Exception as e:
        logger.error(f"Music analysis failed: {e}")
        return {}


def run_music_analyzer(context: dict) -> dict:
    """Analyze all generated music tracks. Stores results in context['music_analysis']."""
    logger.info("Analyzing generated music tracks with librosa...")
    context["music_analysis"] = []

    for ep_idx, tracks in enumerate(context.get("music_tracks", [])):
        score_path = (tracks or {}).get("score", "")
        if not score_path or not Path(score_path).exists():
            logger.warning(f"Episode {ep_idx + 1}: no score found — skipping")
            context["music_analysis"].append({})
            continue
        context["music_analysis"].append(analyze_music_track(score_path, ep_idx + 1))

    return context
