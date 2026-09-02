import json
from pathlib import Path

with open('outputs/episode_1_edl.json') as f:
    edl = json.load(f)

print(f"Total duration: {edl['total_duration_ms']/1000:.1f}s")
print(f"Entries: {edl['entry_count']}")
print(f"Reveals: {edl['reveal_count']}")
print(f"Chunks: {edl['chunk_count']}")
print(f"Dramatic pauses: {edl['dramatic_pause_count']}")
print()

for i, e in enumerate(edl['entries']):
    img = Path(e.get('image_path', '')).name if e.get('image_path') else 'MISSING'
    img_exists = '✓' if e.get('image_path') and Path(e['image_path']).exists() else '✗'
    print(
        f"[{i:02d}] {e['type']:12s} | "
        f"{e['start_ms']/1000:6.1f}s → {e['end_ms']/1000:6.1f}s | "
        f"{e['duration_ms']/1000:5.1f}s | "
        f"{e.get('arc_position',''):12s} | "
        f"{img_exists} {img[:40]}"
    )