"""
Quick re-render: load context_after_edit_plan.json and run only Agent 6b.
Use this to test LUT / grain / vignette changes without re-generating any images.
"""
import json
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from agents.agent_6b_video_compositor import run_agent_6b

CHECKPOINT = "outputs/context_after_edit_plan.json"

print("Loading context checkpoint...")
with open(CHECKPOINT, "r", encoding="utf-8") as f:
    context = json.load(f)

print(f"Story: {context.get('story_title')} — {context.get('total_episodes')} episode(s)")
print(f"EDL entries: {context['edl'][0]['entry_count'] if context.get('edl') else 'n/a'}")
print("\nRunning Agent 6b (video compositor only)...")

context = run_agent_6b(context)

finals = context.get("final_episodes", [])
if finals:
    for path in finals:
        print(f"\nOutput: {path}")
else:
    print("\n⚠ No final episodes in context — check errors below")

errors = context.get("errors", [])
if errors:
    for e in errors:
        print(f"  ERROR: {e}")
