import os
import json
import re
import anthropic
from dotenv import load_dotenv
from utils.logger import AgentLogger
from utils.errors import AgentException, RetryableException
from utils.retry import retry_with_backoff

load_dotenv()

logger = AgentLogger("NarrationDirector")

BASE_INSTRUCTIONS = (
    "Extremely deep bass. Cold, controlled, certain. "
    "Slow and deliberate. True crime documentary narration. "
    "Every word is fact. Dry, mirthless weight."
)

CHUNK_INSTRUCTIONS = {
    "opening": (
        "Slow, heavy, detached. Observational. Like narrating from distant memory. "
        "Each sentence lands separately. Cold. No warmth. Build unease quietly."
    ),
    "build": (
        "Slightly slower. Weight increasing with each line. "
        "Controlled. Darkness building underneath. "
        "The facts are starting to not add up. Let that register without showing it."
    ),
    "revelation": (
        "Slowest delivery. Each word lands like evidence. "
        "Drop register even lower on the most important words. "
        "Maximum weight. Flat affect. The horror is in the facts not the emotion."
    ),
    "cliffhanger": (
        "Flat. Final. No drama — the drama is in the words themselves. "
        "Last sentence delivered then complete silence. "
        "Cold certainty. Like a door closing forever."
    )
}

CLEAN_AND_STRUCTURE_PROMPT = """
You are a narration director for a true crime short film series.

Take this raw episode script and return ONLY a JSON array of narration sentences.

RAW SCRIPT:
{script}

STRICT CLEANING RULES — remove ALL of these:
- Lines inside square brackets [like this] — delete entirely
- Lines starting with NARRATOR, Narrator, V.O., CAMERA, CUT TO, FADE, INT., EXT.
- Lines starting with # or * or ---
- Episode titles and episode numbers
- DISCLAIMER sections entirely
- THEMATIC QUOTE sections entirely
- Any stage directions
- Markdown formatting
- Lines with only punctuation

After cleaning you have ONLY pure spoken narration sentences.

Now split into individual sentences and assign each one:
- "type": one of opening / build / revelation / cliffhanger
- "text": the sentence rewritten for maximum dramatic impact
  * Very short. Sparse.
  * Never use "..." — replace with a period and line break
  * Remove all full stops at end of sentences — do NOT include periods
  * Isolate shocking words on their own line
  * Keep it cold and factual
- "pause_after": true or false — true for sentences that need silence after them

RESPOND ONLY WITH THIS JSON — no explanation, no markdown:
[
    {{
        "type": "opening",
        "text": "The night shift feels different",
        "pause_after": false
    }},
    {{
        "type": "opening", 
        "text": "Quieter",
        "pause_after": true
    }},
    {{
        "type": "build",
        "text": "Three patients flagged as unsuitable",
        "pause_after": false
    }},
    {{
        "type": "revelation",
        "text": "All on his shift",
        "pause_after": true
    }},
    {{
        "type": "cliffhanger",
        "text": "I am the next file",
        "pause_after": true
    }}
]
"""

@retry_with_backoff(max_retries=3, base_wait=5.0, retryable_exceptions=(RetryableException,))
def call_claude(prompt: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        raise RetryableException(f"Claude call failed: {e}")

def group_sentences_into_chunks(sentences: list, max_chars: int = 400) -> list:
    """
    Group individual sentences into chunks under max_chars.
    Each chunk gets the instructions of its dominant sentence type.
    Respects pause_after markers.
    """
    chunks = []
    current_sentences = []
    current_chars = 0
    current_type = "opening"

    for sentence in sentences:
        text = sentence["text"]
        pause = sentence.get("pause_after", False)
        stype = sentence.get("type", "opening")

        # Start new chunk if adding this sentence exceeds limit
        # or previous sentence requested a pause
        if current_sentences and (
            current_chars + len(text) > max_chars or
            current_sentences[-1].get("pause_after", False)
        ):
            chunks.append({
                "type": current_type,
                "sentences": current_sentences,
                "instructions": CHUNK_INSTRUCTIONS.get(current_type, BASE_INSTRUCTIONS)
            })
            current_sentences = []
            current_chars = 0

        current_sentences.append(sentence)
        current_chars += len(text)
        current_type = stype  # last sentence type wins

    if current_sentences:
        chunks.append({
            "type": current_type,
            "sentences": current_sentences,
            "instructions": CHUNK_INSTRUCTIONS.get(current_type, BASE_INSTRUCTIONS)
        })

    return chunks

def build_chunk_text(chunk: dict) -> str:
    """
    Build final text for a chunk from its sentences.
    Each sentence on its own line for natural pausing.
    """
    lines = []
    for sentence in chunk["sentences"]:
        text = sentence["text"].strip()
        # Remove any remaining periods
        text = text.rstrip(".")
        lines.append(text)
    return "\n\n".join(lines)

def structure_narration(script: str) -> list:
    """
    Takes raw script, returns list of chunks each with text and instructions.
    """
    prompt = CLEAN_AND_STRUCTURE_PROMPT.format(script=script)
    raw = call_claude(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        sentences = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to parse sentences: {e}")
        return [{
            "type": "opening",
            "text": script,
            "instructions": BASE_INSTRUCTIONS,
            "pause_after_chunk": True
        }]

    # Group sentences into chunks
    chunks = group_sentences_into_chunks(sentences, max_chars=400)

    # Build final text per chunk
    result = []
    for chunk in chunks:
        result.append({
            "type": chunk["type"],
            "text": build_chunk_text(chunk),
            "instructions": chunk["instructions"],
            "pause_after_chunk": True
        })

    logger.info(f"Narration structured into {len(result)} chunks")
    return result

def run_narration_director(context: dict) -> dict:
    """
    Agent 4a: Structure all episode scripts for voiceover delivery.
    """
    logger.info("Structuring narration for all episodes...")

    structured_scripts = []

    for i, script in enumerate(context.get("scripts", [])):
        logger.info(f"  Structuring episode {i + 1}...")
        try:
            chunks = structure_narration(script)
            structured_scripts.append(chunks)
            logger.info(f"  Episode {i + 1}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Episode {i + 1} failed: {e}")
            structured_scripts.append([{
                "type": "opening",
                "text": script,
                "instructions": BASE_INSTRUCTIONS,
                "pause_after_chunk": True
            }])

    context["structured_narration"] = structured_scripts
    logger.info("Narration direction complete")
    return context