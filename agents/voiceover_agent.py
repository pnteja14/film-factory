import io
import os
import re
import wave
import struct
import requests
from dotenv import load_dotenv
from utils.logger import AgentLogger
from utils.errors import VoiceoverAgentException, RetryableException, PermanentException
from utils.retry import retry_with_backoff

load_dotenv(dotenv_path=".env", override=True)

try:
    import dashscope
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    _DASHSCOPE_AVAILABLE = True
except ImportError:
    dashscope = None
    _DASHSCOPE_AVAILABLE = False

SYNTHESIS_MODEL = "qwen3-tts-instruct-flash"

# VOICE_SOURCE controls where voiceover audio comes from.
# "local"      — generate via DashScope Qwen3-TTS (default)
# "elevenlabs" — generate via ElevenLabs with a cloned reference voice
# "colab"      — use pre-generated WAV already on disk
VOICE_SOURCE        = os.getenv("VOICE_SOURCE", "local")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

_VALID_VOICE_SOURCES = {"local", "colab", "elevenlabs"}
if VOICE_SOURCE not in _VALID_VOICE_SOURCES:
    import warnings
    warnings.warn(
        f"VOICE_SOURCE='{VOICE_SOURCE}' is not recognised "
        f"(valid: {_VALID_VOICE_SOURCES}). Falling back to 'local'.",
        stacklevel=2,
    )
    VOICE_SOURCE = "local"

NARRATOR_INSTRUCTIONS = (
    "Extremely deep bass male voice. Maximum chest resonance. Low and dark throughout. "
    "No emotional expressiveness. No drama. Just darkness delivered with absolute control. "
    "Pace: Slow and deliberate but flowing. "
    "Pause rules: Period = 0.5 second pause. Ellipsis = 1 second pause. "
    "Blank line = one breath only. Never drag out individual words unnecessarily. "
    "Bass and depth over everything. Restraint and control throughout."
)

logger = AgentLogger("VoiceoverAgent")

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # 16-bit PCM



def extract_narration(script: str) -> str:
    """Extract only spoken narration from script."""
    try:
        if not script or not script.strip():
            raise VoiceoverAgentException("Script is empty")

        cleaned = re.sub(r'\[.*?\]', '', script, flags=re.DOTALL)

        lines = cleaned.split("\n")
        narration_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                if narration_lines and narration_lines[-1] != "":
                    narration_lines.append("")
                continue

            skip_patterns = [
                "NARRATOR:", "Narrator:", "narrator:",
                "V.O.", "CUT TO", "FADE", "INT.", "EXT.",
                "Cliffhanger:", "CLIFFHANGER:", "Here is", "Note:"
            ]
            if any(line.startswith(p) for p in skip_patterns):
                continue

            narration_lines.append(line)

        narration = "\n".join(narration_lines)
        narration = re.sub(r'\n{3,}', '\n\n', narration)
        return narration.strip()

    except VoiceoverAgentException:
        raise
    except Exception as e:
        raise VoiceoverAgentException(f"Failed to extract narration: {e}")


def format_for_pacing(narration: str) -> str:
    """Format narration for dramatic pacing."""
    lines = narration.split("\n")
    formatted = []
    for line in lines:
        if not line.strip():
            formatted.append("")
            continue
        word_count = len(line.split())
        if word_count <= 4 and not line.endswith("..."):
            line = line.rstrip(".") + "..."
        formatted.append(line)
    return "\n".join(formatted)


def chunk_narration(narration: str, max_chars: int = 500) -> list:
    """Split narration into chunks under API limit."""
    if not narration or not narration.strip():
        raise VoiceoverAgentException("Narration is empty")

    paragraphs = narration.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        else:
            if len(current_chunk) + len(paragraph) + 2 <= max_chars:
                current_chunk = (current_chunk + "\n\n" + paragraph).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    if not chunks:
        raise VoiceoverAgentException("Failed to create chunks")

    logger.info(f"Narration split into {len(chunks)} chunks")
    return chunks


@retry_with_backoff(max_retries=3, base_wait=5.0, retryable_exceptions=(RetryableException,))
def call_tts_api(text: str, voice_id: str = None, instructions: str = None) -> bytes:
    """
    Call DashScope TTS API (qwen3-tts-instruct-flash). Returns raw WAV audio bytes.

    qwen3-tts-instruct-flash lives on the multimodal-generation endpoint (same
    URL as the old voice-derived model) but requires input.text format rather
    than the messages array that MultiModalConversation sends.
    The instructions are passed in parameters.system_message.
    """
    import base64

    try:
        if not text or not text.strip():
            raise PermanentException("Text is empty")
        if len(text) > 600:
            raise PermanentException(f"Text exceeds 600 char limit: {len(text)} chars")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        system_instruction = instructions or NARRATOR_INSTRUCTIONS

        payload = {
            "model": SYNTHESIS_MODEL,
            "input": {"text": text},
            "parameters": {
                "voice": "Ethan",
                "format": "wav",
                "sample_rate": SAMPLE_RATE,
                "system_message": system_instruction,
            },
        }

        resp = requests.post(
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            try:
                err_data = resp.json()
                code = err_data.get("code", "")
                msg  = err_data.get("message", resp.text[:300])
                error_msg = f"<{resp.status_code}> {code}: {msg}"
            except Exception:
                error_msg = resp.text[:300]

            if resp.status_code in (429, 500, 503):
                raise RetryableException(f"Server error: {error_msg}")
            else:
                raise PermanentException(f"API error {resp.status_code}: {error_msg}")

        # Some endpoints return audio bytes directly in the response body
        content_type = resp.headers.get("Content-Type", "")
        if "audio" in content_type or "octet-stream" in content_type:
            if len(resp.content) > 0:
                return resp.content
            raise RetryableException("Direct audio response is empty")

        data = resp.json()
        logger.debug(f"TTS response keys: {list(data.get('output', {}).keys())}")
        audio_info = data.get("output", {}).get("audio", {})

        # URL-based audio (standard DashScope TTS format)
        audio_url = audio_info.get("url")
        if audio_url:
            ar = requests.get(audio_url, timeout=60)
            ar.raise_for_status()
            if len(ar.content) == 0:
                raise RetryableException("Downloaded audio is empty")
            return ar.content

        # Inline base64 audio (some model variants return this)
        audio_b64 = audio_info.get("data") or audio_info.get("base64")
        if audio_b64:
            decoded = base64.b64decode(audio_b64)
            if len(decoded) == 0:
                raise RetryableException("Decoded audio is empty")
            return decoded

        # Log full response structure to diagnose unexpected formats
        raise RetryableException(
            f"No audio found in response. "
            f"output keys: {list(data.get('output', {}).keys())} | "
            f"audio_info: {str(audio_info)[:200]}"
        )

    except (RetryableException, PermanentException):
        raise
    except Exception as e:
        raise RetryableException(f"TTS API call failed: {e}")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a WAV header."""
    data_size = len(pcm_bytes)
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))          # chunk size
    buf.write(struct.pack('<H', 1))           # PCM
    buf.write(struct.pack('<H', 1))           # mono
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))  # byte rate
    buf.write(struct.pack('<H', 2))           # block align (16-bit mono)
    buf.write(struct.pack('<H', 16))          # bits per sample
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()


@retry_with_backoff(max_retries=3, base_wait=5.0, retryable_exceptions=(RetryableException,))
def call_elevenlabs_tts(text: str) -> bytes:
    """
    Generate speech via ElevenLabs using the cloned reference voice.
    Returns WAV bytes at 24 kHz mono 16-bit PCM.

    Requires ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env.
    Voice settings tuned for a controlled, deep narrator:
      stability=0.70    — very consistent, minimal pitch drift
      similarity_boost=0.85 — high fidelity to the reference voice
      style=0.05        — near-zero style exaggeration
      use_speaker_boost=True — maximises voice clarity
    """
    if not ELEVENLABS_API_KEY:
        raise PermanentException("ELEVENLABS_API_KEY not set in .env")
    if not ELEVENLABS_VOICE_ID:
        raise PermanentException(
            "ELEVENLABS_VOICE_ID not set in .env. "
            "Clone your voice at elevenlabs.io → Voices → Add Voice → Instant Clone, "
            "then copy the Voice ID into .env as ELEVENLABS_VOICE_ID=xxx"
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "output_format": "pcm_24000",
        "voice_settings": {
            "stability":        0.70,
            "similarity_boost": 0.85,
            "style":            0.05,
            "use_speaker_boost": True,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)

        if resp.status_code == 401:
            raise PermanentException("ElevenLabs API key is invalid or expired")
        if resp.status_code == 404:
            raise PermanentException(
                f"Voice ID '{ELEVENLABS_VOICE_ID}' not found in your ElevenLabs account"
            )
        if resp.status_code == 429:
            raise RetryableException("ElevenLabs rate limit — will retry")
        if resp.status_code not in (200, 201):
            raise RetryableException(
                f"ElevenLabs error {resp.status_code}: {resp.text[:200]}"
            )

        pcm_bytes = resp.content
        if not pcm_bytes:
            raise RetryableException("ElevenLabs returned empty audio")

        wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=24000)
        logger.debug(f"ElevenLabs TTS: {len(pcm_bytes)//1024} KB PCM → WAV")
        return wav_bytes

    except (RetryableException, PermanentException):
        raise
    except Exception as e:
        raise RetryableException(f"ElevenLabs TTS call failed: {e}")


def get_wav_duration_ms(audio_bytes: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration_ms = int((frames / rate) * 1000)
            logger.debug(
                f"WAV: {frames} frames, {rate}Hz, "
                f"{channels}ch, {sampwidth}B — {duration_ms}ms"
            )
            return duration_ms
    except Exception as e:
        logger.warning(f"Could not read WAV duration: {e}")
        return 0


def generate_silence(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate silence as WAV bytes."""
    num_samples = int(sample_rate * duration_seconds)
    silence_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(silence_data)

    return buffer.getvalue()


def stitch_audio_chunks(
    audio_bytes_list: list,
    output_path: str,
    gap_seconds: float = 0.8
) -> list:
    """
    Stitch WAV chunks with silence between each and 2 seconds at end.

    Returns a list of timestamp dicts:
    [{ chunk_index, start_ms, end_ms, audio_file }]

    This timestamp data is critical for Agent 6a to sync
    reveal cuts to the exact spoken word.
    """
    try:
        silence_between = generate_silence(gap_seconds)
        silence_end = generate_silence(2.0)

        timestamps = []

        with wave.open(output_path, 'wb') as outfile:
            params_set = False

            for i, audio_bytes in enumerate(audio_bytes_list):
                try:
                    with wave.open(io.BytesIO(audio_bytes), 'rb') as infile:
                        if not params_set:
                            outfile.setparams(infile.getparams())
                            params_set = True
                        outfile.writeframes(infile.readframes(infile.getnframes()))

                    # Add silence between chunks
                    if i < len(audio_bytes_list) - 1:
                        with wave.open(io.BytesIO(silence_between), 'rb') as sil:
                            outfile.writeframes(sil.readframes(sil.getnframes()))

                    # Placeholder — real timestamps fixed after stitching
                    timestamps.append({
                        "chunk_index": i,
                        "start_ms": 0,
                        "end_ms": 0,
                        "duration_ms": 0,
                        "audio_file": output_path
                    })

                except Exception as e:
                    logger.warning(f"Skipping chunk {i + 1}: {e}")
                    timestamps.append({
                        "chunk_index": i,
                        "start_ms": 0,
                        "end_ms": 0,
                        "duration_ms": 0,
                        "audio_file": output_path
                    })

            # Add 2 seconds silence at end
            if params_set:
                with wave.open(io.BytesIO(silence_end), 'rb') as sil:
                    outfile.writeframes(sil.readframes(sil.getnframes()))

        # Fix timestamps using actual stitched file duration
        with open(output_path, 'rb') as f:
            total_ms = get_wav_duration_ms(f.read())

        n = len(timestamps)
        for i, ts in enumerate(timestamps):
            ts["start_ms"] = int((i / n) * total_ms)
            ts["end_ms"] = int(((i + 1) / n) * total_ms)
            ts["duration_ms"] = ts["end_ms"] - ts["start_ms"]

        logger.info(f"Audio stitched: {output_path} — total {total_ms}ms")
        return timestamps

    except Exception as e:
        raise VoiceoverAgentException(f"Failed to stitch audio: {e}")


def generate_voiceover(context: dict, episode_number: int) -> str:
    """
    Agent 4: Generate voiceover for an episode.

    Outputs:
    - outputs/audio/episode_N_voiceover.wav
    - context["voiceover_timestamps"][episode_number]: chunk-level timing data

    Timestamp data format:
    [{ chunk_index, start_ms, end_ms, duration_ms, audio_file }]

    Agent 6a uses these timestamps to sync reveal cuts to
    the exact millisecond a trigger phrase is spoken.
    """
    logger.info(f"Generating voiceover for episode {episode_number + 1}...")

    try:
        # ── COLAB VOICE SOURCE: use pre-generated WAV ─────────────────
        # elevenlabs and local both fall through to the chunk generation loop below
        if VOICE_SOURCE == "colab":
            audio_path = f"outputs/audio/episode_{episode_number + 1}_voiceover.wav"
            from pathlib import Path as _Path
            if _Path(audio_path).exists():
                logger.info(f"VOICE_SOURCE=colab: using pre-generated {audio_path}")
                context["audio_files"].append(audio_path)
                # Timestamps will be derived by Whisper alignment
                if "voiceover_timestamps" not in context:
                    context["voiceover_timestamps"] = []
                while len(context["voiceover_timestamps"]) <= episode_number:
                    context["voiceover_timestamps"].append([])
                return audio_path
            else:
                logger.warning(
                    f"VOICE_SOURCE=colab but no WAV at {audio_path} — "
                    "falling back to local TTS"
                )

        if episode_number >= len(context.get("scripts", [])):
            raise VoiceoverAgentException(
                f"Episode {episode_number + 1} script not found"
            )

        # Get structured narration from Agent 4a
        structured = context.get("structured_narration", [])
        if episode_number < len(structured):
            chunks = structured[episode_number]
            logger.info(f"Using structured narration: {len(chunks)} chunks")
        else:
            logger.warning("No structured narration found, falling back to raw extraction")
            script = context["scripts"][episode_number]
            narration = extract_narration(script)
            narration = format_for_pacing(narration)
            raw_chunks = chunk_narration(narration, max_chars=500)
            chunks = [{"text": c, "instructions": NARRATOR_INSTRUCTIONS} for c in raw_chunks]

        logger.info(f"Generating {len(chunks)} audio chunks...")
        audio_bytes_list = []

        for idx, chunk in enumerate(chunks):
            logger.info(f"  Chunk {idx + 1}/{len(chunks)}: {len(chunk['text'])} chars")
            try:
                if VOICE_SOURCE == "elevenlabs":
                    audio_bytes = call_elevenlabs_tts(chunk["text"])
                else:
                    audio_bytes = call_tts_api(
                        chunk["text"],
                        instructions=chunk.get("instructions", NARRATOR_INSTRUCTIONS)
                    )
                audio_bytes_list.append(audio_bytes)
            except PermanentException as e:
                logger.error(f"Chunk {idx + 1} permanent error: {e}")
                raise VoiceoverAgentException(f"Chunk {idx + 1} failed permanently: {e}")
            except Exception as e:
                logger.error(f"Chunk {idx + 1} failed: {e}")
                context["warnings"].append({
                    "agent": "voiceover_agent",
                    "episode": episode_number + 1,
                    "warning": f"Chunk {idx + 1} failed: {e}"
                })

        if not audio_bytes_list:
            raise VoiceoverAgentException("No audio generated")

        os.makedirs("outputs/audio", exist_ok=True)
        audio_path = f"outputs/audio/episode_{episode_number + 1}_voiceover.wav"

        if len(audio_bytes_list) == 1:
            # Single chunk — save with end silence
            with wave.open(audio_path, 'wb') as outfile:
                with wave.open(io.BytesIO(audio_bytes_list[0]), 'rb') as infile:
                    outfile.setparams(infile.getparams())
                    outfile.writeframes(infile.readframes(infile.getnframes()))
                silence = generate_silence(2.0)
                with wave.open(io.BytesIO(silence), 'rb') as sil:
                    outfile.writeframes(sil.readframes(sil.getnframes()))

            chunk_duration_ms = get_wav_duration_ms(audio_bytes_list[0])
            timestamps = [{
                "chunk_index": 0,
                "start_ms": 0,
                "end_ms": chunk_duration_ms,
                "duration_ms": chunk_duration_ms,
                "audio_file": audio_path
            }]
        else:
            timestamps = stitch_audio_chunks(audio_bytes_list, audio_path, gap_seconds=0.8)

        # Store timestamps in context for Agent 6a
        if "voiceover_timestamps" not in context:
            context["voiceover_timestamps"] = []
        while len(context["voiceover_timestamps"]) <= episode_number:
            context["voiceover_timestamps"].append([])
        context["voiceover_timestamps"][episode_number] = timestamps

        context["audio_files"].append(audio_path)
        logger.info(
            f"Voiceover complete: {audio_path} | "
            f"{len(timestamps)} chunks timed"
        )

        # Log timestamp summary for debugging
        for ts in timestamps:
            logger.info(
                f"  Chunk {ts['chunk_index']}: "
                f"{ts['start_ms']}ms → {ts['end_ms']}ms"
            )

        return audio_path

    except VoiceoverAgentException as e:
        logger.error(f"Voiceover generation failed: {e}")
        context["errors"].append({
            "agent": "voiceover_agent",
            "episode": episode_number + 1,
            "error": str(e)
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        context["errors"].append({
            "agent": "voiceover_agent",
            "episode": episode_number + 1,
            "error": f"Unexpected: {str(e)}"
        })
        raise VoiceoverAgentException(f"Unexpected error: {e}")