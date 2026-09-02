"""
Verify all 9 fixes before running the full pipeline.
No API calls, no image generation, no rendering.
Tests logic only.
"""
import sys
import json
import numpy as np
import wave
from pathlib import Path

sys.path.insert(0, '.')
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {detail}")
        failed += 1

print("=" * 60)
print("FIX 1+2: Real Whisper timestamps")
print("=" * 60)

try:
    from agents.agent_whisper_alignment import derive_chunk_timestamps_from_words

    # Simulate word timestamps for 10 chunks
    fake_words = []
    total_ms = 53760
    words_per_chunk = 8
    for chunk_idx in range(10):
        chunk_start = int((chunk_idx / 10) * total_ms)
        chunk_end = int(((chunk_idx + 1) / 10) * total_ms)
        for w in range(words_per_chunk):
            word_start = chunk_start + int((w / words_per_chunk) * (chunk_end - chunk_start))
            word_end = chunk_start + int(((w + 1) / words_per_chunk) * (chunk_end - chunk_start))
            fake_words.append({
                "word": f"word{w}",
                "start_ms": word_start,
                "end_ms": word_end,
                "chunk_index": chunk_idx,
            })

    # Create a dummy WAV for testing
    import io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(np.zeros(int(24000 * total_ms / 1000), dtype=np.int16).tobytes())
    with open('/tmp/test_verify.wav', 'wb') as f:
        f.write(buf.getvalue())

    timestamps = derive_chunk_timestamps_from_words(fake_words, 10, '/tmp/test_verify.wav')

    check("Returns 10 chunks", len(timestamps) == 10)
    check("All chunks have start_ms", all('start_ms' in t for t in timestamps))
    check("All chunks have end_ms", all('end_ms' in t for t in timestamps))
    check("Chunks sequential", all(timestamps[i]['end_ms'] <= timestamps[i+1]['start_ms'] + 100 for i in range(9)))
    check("First chunk starts near 0", timestamps[0]['start_ms'] < 1000)
    check("Last chunk ends near total", abs(timestamps[-1]['end_ms'] - total_ms) < 2000)

    import os
    os.remove('/tmp/test_verify.wav')

except Exception as e:
    print(f"  FAIL: Import error: {e}")
    failed += 2

print()
print("=" * 60)
print("FIX 3a: Reveal chunk_index validation")
print("=" * 60)

try:
    from agents.agent_6a_edit_planner import validate_reveal_chunk_indices

    chunks = [
        {"text": "ten years of silence broken by a phone call", "type": "opening"},
        {"text": "sophia says there has been a death", "type": "build"},
        {"text": "he almost hung up the phone", "type": "build"},
        {"text": "ravenshade manor stands in the fog", "type": "revelation"},
    ]
    timestamps = [
        {"chunk_index": 0, "start_ms": 0, "end_ms": 5000},
        {"chunk_index": 1, "start_ms": 5000, "end_ms": 10000},
        {"chunk_index": 2, "start_ms": 10000, "end_ms": 15000},
        {"chunk_index": 3, "start_ms": 15000, "end_ms": 20000},
    ]

    # Reveal assigned to wrong chunk (0) but trigger is in chunk 2
    reveals = [{"chunk_index": 0, "trigger_phrase": "almost hung up", "image_path": "test.png"}]
    corrected = validate_reveal_chunk_indices(reveals, chunks, timestamps)

    check("Corrects wrong chunk_index", corrected[0]["chunk_index"] == 2,
          f"got {corrected[0]['chunk_index']}")

    # Reveal already correct
    reveals2 = [{"chunk_index": 3, "trigger_phrase": "ravenshade manor fog", "image_path": "test.png"}]
    corrected2 = validate_reveal_chunk_indices(reveals2, chunks, timestamps)
    check("Keeps correct chunk_index", corrected2[0]["chunk_index"] == 3)

except Exception as e:
    print(f"  FAIL: {e}")
    failed += 2

print()
print("=" * 60)
print("FIX 3b+3c: No overlapping entries, pause separation")
print("=" * 60)

try:
    from agents.agent_6a_edit_planner import build_timestamp_driven_edl

    # Minimal test — 3 chunks, 1 pause, no reveals
    scene_prompts = [
        {"chunk_index": 0, "image_path": "outputs/anchors/ravenshade_manor.png",
         "arc_position": "establish", "zoom_direction": "out",
         "location_name": "Manor", "image_type": "location"},
        {"chunk_index": 1, "image_path": "outputs/anchors/virats_apartment.png",
         "arc_position": "build", "zoom_direction": "in",
         "location_name": "Apartment", "image_type": "location"},
        {"chunk_index": 2, "image_path": "outputs/anchors/ravenshade_manor.png",
         "arc_position": "revelation", "zoom_direction": "in",
         "location_name": "Manor", "image_type": "location"},
    ]
    timestamps = [
        {"chunk_index": 0, "start_ms": 0, "end_ms": 5000, "duration_ms": 5000},
        {"chunk_index": 1, "start_ms": 5000, "end_ms": 10000, "duration_ms": 5000},
        {"chunk_index": 2, "start_ms": 10000, "end_ms": 15000, "duration_ms": 5000},
    ]
    dramatic_pauses = [
        {"after_chunk_index": 1, "pause_after_ms": 10000,
         "pause_duration_ms": 3000, "effect": "slow_zoom_intensify",
         "music_instruction": "swell", "auto": False}
    ]

    # Use a real audio file if available, else skip audio modification
    audio = "outputs/audio/episode_1_voiceover.wav"
    if not Path(audio).exists():
        audio = "/tmp/dummy.wav"
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            wf.writeframes(np.zeros(24000*15, dtype=np.int16).tobytes())
        with open(audio, 'wb') as f:
            f.write(buf.getvalue())

    edl = build_timestamp_driven_edl(
        scene_prompts, [], timestamps, dramatic_pauses, 0, audio
    )

    entries = edl["entries"]

    # Check no two scene entries overlap
    scene_entries = [e for e in entries if e["type"] == "scene"]
    overlaps = 0
    for i in range(len(scene_entries)):
        for j in range(i+1, len(scene_entries)):
            a, b = scene_entries[i], scene_entries[j]
            if a["start_ms"] < b["end_ms"] and b["start_ms"] < a["end_ms"]:
                if a.get("image_path") == b.get("image_path"):
                    overlaps += 1

    check("No overlapping scene entries with same image", overlaps == 0,
          f"found {overlaps} overlaps")

    # Check dramatic pause exists
    pause_entries = [e for e in entries if e["type"] == "dramatic_pause"]
    check("Dramatic pause in EDL", len(pause_entries) >= 1)

    # Check pause fires after chunk 1
    if pause_entries:
        check("Pause fires at chunk 1 end", pause_entries[0]["start_ms"] >= 10000,
              f"got {pause_entries[0]['start_ms']}")

    # Check total duration is audio + pause + 500ms tail
    check("Total duration reasonable", 15000 < edl["total_duration_ms"] < 25000,
          f"got {edl['total_duration_ms']}")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()
    failed += 4

print()
print("=" * 60)
print("FIX 4: Music swell intensity")
print("=" * 60)

try:
    src = open('agents/agent_6b_video_compositor.py', encoding='utf-8').read()
    check("Swell reaches 0.65", '0.65' in src)
    check("Peak reaches 0.72", '0.72' in src)
    check("Drop to 0.03", '0.03' in src)
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 3

print()
print("=" * 60)
print("FIX 5: Typewriter speed")
print("=" * 60)

try:
    src = open('agents/agent_6b_video_compositor.py', encoding='utf-8').read()
    check("Typewriter at 18.0 chars/sec", 'chars_per_second: float = 18.0' in src)
except Exception as e:
    failed += 1

print()
print("=" * 60)
print("FIX 6: Marcus archetype")
print("=" * 60)

try:
    src = open('agents/scene_director.py', encoding='utf-8').read()
    check("Marcus override exists", '"marcus"' in src)
    check("Virat override exists", '"virat"' in src)
    check("Name overrides before loop", src.index('name_overrides') < src.index('for char in characters'))
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 3

print()
print("=" * 60)
print("FIX 7+8: Stale image deletion + Whisper cache in test.py")
print("=" * 60)

try:
    src = open('test.py', encoding='utf-8').read()
    check("Stale scene deletion in test.py", 'stale' in src and 'os.remove' in src)
    check("Whisper cache deletion in test.py", 'whisper_cache' in src and 'unlink' in src)
    check("Real timestamps used", 'derive_chunk_timestamps_from_words' in src)
    check("Fake timestamps removed", 'int((i/n_chunks)*total_ms)' not in src)
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 4

print()
print("=" * 60)
print("FIX 9: Auto pause on anchor locations")
print("=" * 60)

try:
    src = open('agents/agent_6a_edit_planner.py', encoding='utf-8').read()
    check("ANCHOR_KEYWORDS defined", 'ANCHOR_KEYWORDS' in src)
    check("add_auto_pauses_for_anchor_locations exists", 'add_auto_pauses_for_anchor_locations' in src)
    check("Called in run_agent_6a", src.count('add_auto_pauses_for_anchor_locations') >= 2)
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 3

print()
print("=" * 60)
print(f"RESULT: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL FIXES VERIFIED — safe to run")
else:
    print("ISSUES FOUND — fix before running")
print("=" * 60)