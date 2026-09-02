"""
Agent Whisper Alignment — word-level timestamp extraction from voiceover WAV.

Uses OpenAI Whisper with forced alignment to find exactly when
each word is spoken in the audio file.

Output: word_timestamps per episode
[{ word, start_ms, end_ms, chunk_index }]

This enables:
- Mid-sentence dramatic pauses at exact word boundaries
- Word-level subtitle sync
- Precise reveal cut timing
"""

import os
import wave
import numpy as np
from pathlib import Path
from utils.logger import AgentLogger

logger = AgentLogger("WhisperAlignment")

ALIGNMENT_CACHE_DIR = Path("outputs/alignment")
ALIGNMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_whisper_model = None
_whisperx_model = None


def load_whisper_model(model_size: str = "base"):
    """Load Whisper model — cached after first load."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        logger.info(f"Loading Whisper {model_size} model...")
        _whisper_model = whisper.load_model(model_size)
        logger.info("Whisper model loaded")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        return None


def _assign_chunks_from_segments(
    segments: list,
    structured_chunks: list,
) -> list:
    """
    Assign a chunk_index to each Whisper segment using word-overlap scoring
    against the narration chunk texts.

    For each segment, find the narration chunk whose text shares the most words
    with the segment transcript. This is far more accurate than proportional
    character-count distribution because it uses actual spoken content.

    Returns raw_words list with chunk_index populated.
    """
    import re

    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-z']+", text.lower()))

    chunk_tokens = [_tokens(c.get("text", "")) for c in structured_chunks]
    n_chunks = len(structured_chunks)

    # Greedy sequential scan: once a chunk is matched, don't go back.
    # This preserves temporal order — narration is always linear.
    last_matched = 0
    raw_words = []

    for seg in segments:
        seg_text = seg.get("text", "")
        seg_tokens = _tokens(seg_text)

        # Score from last_matched forward only — narration is sequential
        best_score = -1
        best_ci    = last_matched
        for ci in range(last_matched, n_chunks):
            score = len(seg_tokens & chunk_tokens[ci])
            if score > best_score:
                best_score = score
                best_ci = ci

        # If no match at all, keep same chunk (silence / stutter segment)
        if best_score > 0:
            last_matched = best_ci

        for word_data in seg.get("words", []):
            word = word_data.get("word", "").strip()
            if not word:
                continue
            raw_words.append({
                "word":      word.lower().strip(".,!?;:\"'"),
                "word_raw":  word,
                "start_ms":  int(word_data.get("start", 0) * 1000),
                "end_ms":    int(word_data.get("end",   0) * 1000),
                "chunk_index": last_matched,
            })

    return raw_words


def _extract_with_whisperx(audio_path: str, structured_chunks: list) -> list:
    """
    Use WhisperX large-v3 with GPU forced alignment for word-level timestamps.
    Falls back to returning [] if WhisperX is not available.

    WhisperX accuracy: ±10-30ms per word vs Whisper base ±200-500ms.
    """
    global _whisperx_model
    try:
        import torch
        import whisperx

        device     = "cuda" if torch.cuda.is_available() else "cpu"
        compute    = "float16" if device == "cuda" else "int8"
        batch_size = 16

        if _whisperx_model is None:
            logger.info("Loading WhisperX large-v3...")
            _whisperx_model = whisperx.load_model(
                "large-v3", device, compute_type=compute
            )

        audio = whisperx.load_audio(audio_path)
        result = _whisperx_model.transcribe(audio, batch_size=batch_size, language="en")

        # Forced phoneme-level alignment
        align_model, metadata = whisperx.load_align_model(
            language_code="en", device=device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )

        segments = result.get("segments", [])
        raw_words = _assign_chunks_from_segments(segments, structured_chunks)
        logger.info(f"WhisperX: {len(raw_words)} words aligned on {device.upper()}")
        return raw_words

    except ImportError:
        logger.info("WhisperX not installed — falling back to Whisper")
        return []
    except Exception as e:
        logger.warning(f"WhisperX failed: {e} — falling back to Whisper")
        return []


def extract_word_timestamps(
    audio_path: str,
    episode_number: int,
    structured_chunks: list,
) -> list:
    """
    Extract word-level timestamps from voiceover WAV using Whisper.

    Returns list of word dicts:
    [{ word, start_ms, end_ms, chunk_index }]

    chunk_index is estimated by matching word position
    to structured narration chunk text.
    """
    cache_path = ALIGNMENT_CACHE_DIR / f"ep{episode_number}_words.npy"

    # Return cached only if the cache is newer than the audio file
    if cache_path.exists():
        audio_mtime = Path(audio_path).stat().st_mtime if Path(audio_path).exists() else 0
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= audio_mtime:
            try:
                data = np.load(str(cache_path), allow_pickle=True)
                result = data.tolist()
                logger.info(f"Episode {episode_number}: loaded {len(result)} cached word timestamps")
                return result
            except Exception:
                pass
        else:
            logger.info(f"Episode {episode_number}: audio newer than cache — re-running Whisper alignment")
            cache_path.unlink(missing_ok=True)

    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        return []

    # ── Attempt 1: WhisperX with GPU forced alignment (most accurate) ────────
    raw_words = _extract_with_whisperx(audio_path, structured_chunks)

    # ── Attempt 2: Whisper large-v2 on GPU / medium on CPU ───────────────────
    if not raw_words:
        try:
            import torch
            model_size = "large-v2" if torch.cuda.is_available() else "medium"
        except ImportError:
            model_size = "medium"

        model = load_whisper_model(model_size)
        if model is None:
            logger.warning("Whisper unavailable — falling back to chunk timestamps")
            return []

        try:
            logger.info(f"Running Whisper {model_size} alignment on {audio_path}...")
            import whisper

            result = model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en",
                fp16=False,
                verbose=False,
            )

            segments = result.get("segments", [])
            raw_words = _assign_chunks_from_segments(segments, structured_chunks)
            logger.info(f"Episode {episode_number}: {len(raw_words)} words aligned (Whisper {model_size})")

        except Exception as e:
            logger.error(f"Whisper alignment failed: {e}")
            return []

    if not raw_words:
        return []

    np.save(str(cache_path), raw_words)
    return raw_words




def run_whisper_alignment(context: dict) -> dict:
    """
    Run Whisper alignment on all episode voiceovers.
    Stores word_timestamps in context for Agent 6a to use.
    """
    logger.info("Running Whisper word alignment...")

    audio_files = context.get("audio_files", [])
    structured_narration = context.get("structured_narration", [])

    context["word_timestamps"] = []

    for ep_idx, audio_path in enumerate(audio_files):
        chunks = structured_narration[ep_idx] if ep_idx < len(structured_narration) else []

        words = extract_word_timestamps(audio_path, ep_idx + 1, chunks)
        context["word_timestamps"].append(words)

        if words:
            logger.info(
                f"Episode {ep_idx + 1}: {len(words)} words aligned, "
                f"span {words[0]['start_ms']}ms -> {words[-1]['end_ms']}ms"
            )

    return context


def derive_chunk_timestamps_from_words(
    word_timestamps: list,
    n_chunks: int,
    audio_path: str,
) -> list:
    import wave

    # Actual WAV duration is the authoritative ceiling — nothing can exceed this
    actual_ms = 0
    try:
        with wave.open(audio_path, 'rb') as wf:
            actual_ms = int(wf.getnframes() / wf.getframerate() * 1000)
    except Exception:
        actual_ms = 60000

    if not word_timestamps:
        return [
            {
                "chunk_index": i,
                "start_ms": int((i / n_chunks) * actual_ms),
                "end_ms": int(((i + 1) / n_chunks) * actual_ms),
                "duration_ms": int(actual_ms / n_chunks),
                "audio_file": audio_path,
            }
            for i in range(n_chunks)
        ]

    # Group words by the chunk_index Whisper assigned them
    chunk_words: dict = {i: [] for i in range(n_chunks)}
    for w in word_timestamps:
        ci = max(0, min(w.get("chunk_index", 0), n_chunks - 1))
        chunk_words[ci].append(w)

    result = []
    for i in range(n_chunks):
        words = chunk_words[i]
        if words:
            start_ms = min(w["start_ms"] for w in words)
            end_ms   = max(w["end_ms"]   for w in words)
        else:
            # Chunk got no words — place it immediately after previous chunk
            prev_end = result[-1]["end_ms"] if result else 0
            start_ms = prev_end
            end_ms   = prev_end + max(_MIN_SUBTITLE_MS, actual_ms // n_chunks)

        # Hard clamp to actual file duration — Whisper sometimes over-reports
        start_ms = min(start_ms, actual_ms)
        end_ms   = min(end_ms,   actual_ms)

        result.append({
            "chunk_index": i,
            "start_ms":    start_ms,
            "end_ms":      end_ms,
            "duration_ms": max(0, end_ms - start_ms),
            "audio_file":  audio_path,
            # word-level data in seconds for insert-shot trigger lookups in edit planner
            "words": [
                {
                    "word":  w["word"],
                    "start": w["start_ms"] / 1000.0,
                    "end":   w["end_ms"]   / 1000.0,
                }
                for w in chunk_words[i]
            ],
        })

    # Enforce strict monotone, non-overlapping boundaries
    for i in range(len(result) - 1):
        if result[i]["end_ms"] > result[i + 1]["start_ms"]:
            # Overlap: clip this chunk so next one starts cleanly
            result[i]["end_ms"]      = result[i + 1]["start_ms"]
            result[i]["duration_ms"] = max(0, result[i]["end_ms"] - result[i]["start_ms"])
        elif result[i]["end_ms"] < result[i + 1]["start_ms"]:
            # Gap between chunks: extend this chunk to fill it
            result[i]["end_ms"]      = result[i + 1]["start_ms"]
            result[i]["duration_ms"] = result[i]["end_ms"] - result[i]["start_ms"]

    # Last chunk always ends exactly at the actual audio end
    if result:
        result[-1]["end_ms"]      = actual_ms
        result[-1]["duration_ms"] = max(0, actual_ms - result[-1]["start_ms"])

    logger.info(
        f"Chunk timestamps derived: {n_chunks} chunks, "
        f"audio={actual_ms}ms, "
        f"last_chunk_end={result[-1]['end_ms']}ms"
    )
    return result