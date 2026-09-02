"""
Resume pipeline from Voiceover Agent.

Loads context_after_images.json (all scene/reveal images already linked),
then runs:
  4  - Voiceover Agent
  W  - Whisper Alignment
  6a - Edit Planner
  7  - Music Agent
  6b - Video Compositor

Run with:  python test.py
"""
import json
import sys
from pathlib import Path

import os
os.chdir(Path(__file__).parent)

from agents.voiceover_agent import generate_voiceover
from agents.agent_whisper_alignment import run_whisper_alignment, derive_chunk_timestamps_from_words
from agents.agent_6a_edit_planner import run_agent_6a
from agents.agent_7_music_agent import run_agent_7
from agents.agent_6b_video_compositor import run_agent_6b


CHECKPOINT = "outputs/context_after_images.json"


def load_context() -> dict:
    path = Path(CHECKPOINT)
    if not path.exists():
        print(f"ERROR: {CHECKPOINT} not found.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        ctx = json.load(f)
    print(f"Loaded: {CHECKPOINT}")
    print(f"  Story:         {ctx.get('story_title')}")
    print(f"  Episodes:      {ctx.get('total_episodes')}")
    print(f"  Narr. chunks:  {sum(len(e) for e in ctx.get('structured_narration', []))}")
    print(f"  Scene images:  {sum(1 for ep in ctx.get('scene_prompts',[]) for s in ep if s.get('image_path'))}")
    print(f"  Reveal images: {sum(1 for ep in ctx.get('reveal_moments',[]) for r in ep if r.get('image_path'))}")
    return ctx


def reset_downstream(ctx: dict) -> dict:
    ctx["audio_files"]          = []
    ctx["voiceover_timestamps"] = []
    ctx["word_timestamps"]      = []
    ctx["edl"]                  = []
    ctx["dramatic_pauses"]      = []
    ctx["music_tracks"]         = []
    ctx["final_episodes"]       = []
    ctx["errors"]               = []
    ctx["warnings"]             = []
    return ctx


def save_context(ctx: dict, filename: str) -> None:
    out = Path("outputs") / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2)
    print(f"  checkpoint: {out}")


def main():
    for d in ["outputs", "outputs/audio", "outputs/final", "outputs/music", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    ctx = load_context()
    ctx = reset_downstream(ctx)
    total = ctx.get("total_episodes", 1)

    # ── AGENT 4: Voiceover ────────────────────────────────────────────────────
    print("\n=== Agent 4: Voiceover ===")
    for i in range(total):
        print(f"  Episode {i+1}...")
        try:
            audio_path = generate_voiceover(ctx, i)
            if audio_path:
                print(f"  OK  {audio_path}")
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)

    # ── WHISPER ALIGNMENT ─────────────────────────────────────────────────────
    print("\n=== Whisper: Word Alignment ===")
    try:
        ctx = run_whisper_alignment(ctx)
        for ep_idx, words in enumerate(ctx.get("word_timestamps", [])):
            audio_path = ctx["audio_files"][ep_idx]
            n_chunks   = len(ctx["structured_narration"][ep_idx])
            real_ts    = derive_chunk_timestamps_from_words(words, n_chunks, audio_path)
            ctx["voiceover_timestamps"][ep_idx] = real_ts
            print(f"  OK  ep{ep_idx+1}: {len(words)} words, {n_chunks} chunks timed")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # ── AGENT 6a: Edit Planner ────────────────────────────────────────────────
    print("\n=== Agent 6a: Edit Planner ===")
    try:
        ctx = run_agent_6a(ctx)
        edl_built = sum(1 for e in ctx.get("edl", []) if e and e.get("entries"))
        print(f"  OK  {edl_built} EDL(s) built")
        for i, edl in enumerate(ctx.get("edl", [])):
            if edl and edl.get("entries"):
                print(f"      ep{i+1}: {edl['entry_count']} entries, {edl['total_duration_ms']/1000:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # ── AGENT 7: Music ────────────────────────────────────────────────────────
    print("\n=== Agent 7: Music Agent ===")
    try:
        ctx = run_agent_7(ctx)
        print(f"  OK  {len(ctx.get('music_tracks', []))} music set(s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    save_context(ctx, "context_after_edit_plan.json")

    # ── AGENT 6b: Video Compositor ────────────────────────────────────────────
    print("\n=== Agent 6b: Video Compositor ===")
    try:
        ctx = run_agent_6b(ctx)
        final = ctx.get("final_episodes", [])
        print(f"  OK  {len(final)} episode(s) rendered")
        for path in final:
            p = Path(path)
            if p.exists():
                print(f"      {path}  ({p.stat().st_size/1_000_000:.1f} MB)")
            else:
                print(f"      MISSING: {path}")
    except Exception as e:
        print(f"  FAILED: {e}")

    if ctx.get("errors"):
        print(f"\nErrors ({len(ctx['errors'])}):")
        for err in ctx["errors"]:
            print(f"  [{err.get('agent')}] {err.get('error','')[:120]}")

    save_context(ctx, "context.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
