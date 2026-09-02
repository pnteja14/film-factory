"""
Drive Sync Utility — film_factory
==================================
Syncs files between your local project and Google Drive.

Usage
-----
  python sync.py push          Push code + reference files to Drive
  python sync.py pull          Pull Colab/Kaggle outputs from Drive to local
  python sync.py push-outputs  Push local outputs to Drive (before a Colab run)
  python sync.py status        Show which files differ between local and Drive

Drive root : G:\\My Drive\\film_factory\\film_factory\\
Local root : (directory containing this script)
"""

import sys
import subprocess
from pathlib import Path

LOCAL = Path(__file__).parent.resolve()
DRIVE = Path(r"G:\My Drive\film_factory\film_factory")

_SEP = "-" * 60


def _robocopy(src: str, dst: str, *flags: str, files: list[str] | None = None) -> int:
    """
    Run robocopy and return the exit code.
    Robocopy exit codes 0-3 indicate success (no error, some copies, extra files, etc.).
    Exit code >= 8 indicates an error.
    """
    cmd = ["robocopy", src, dst]
    if files:
        cmd.extend(files)
    cmd.extend(flags)
    cmd += ["/R:2", "/W:2", "/NP"]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def _ok(code: int) -> bool:
    return code < 8


def push():
    """Push code and reference files from local -> Drive."""
    print(f"\n{_SEP}")
    print("  PUSH  local -> Drive")
    print(_SEP)

    if not DRIVE.exists():
        print(f"\n  ERROR: Drive not found at {DRIVE}")
        print("  Make sure Google Drive desktop app is running.\n")
        sys.exit(1)

    results = []

    # Root Python + notebook files
    print("\n[1/4] Root files (.py, .ipynb, .gitignore)...")
    code = _robocopy(
        str(LOCAL), str(DRIVE),
        "/XF", "*.onnx", "*.bin", "*.mp4", "*.wav", "*.png", "*.jpg",
        files=["*.py", "*.ipynb", "*.gitignore"],
    )
    results.append(("Root files", code))

    # agents/
    print("[2/4] agents/")
    code = _robocopy(str(LOCAL / "agents"), str(DRIVE / "agents"), files=["*.py"])
    results.append(("agents/", code))

    # colab/ and utils/
    print("[3/4] colab/ and utils/")
    code = _robocopy(str(LOCAL / "colab"), str(DRIVE / "colab"),
                     files=["*.ipynb", "*.py", "*.sh"])
    results.append(("colab/", code))
    code = _robocopy(str(LOCAL / "utils"), str(DRIVE / "utils"), files=["*.py"])
    results.append(("utils/", code))

    # Reference audio (voice cloning source)
    print("[4/4] Reference audio...")
    ref_mp3 = LOCAL / "ElevenLabs_Text_to_Speech_audio.mp3"
    if ref_mp3.exists():
        import shutil
        dst_path = DRIVE / ref_mp3.name
        shutil.copy2(ref_mp3, dst_path)
        print(f"  Copied: {ref_mp3.name} ({ref_mp3.stat().st_size // 1024} KB)")
        results.append(("Reference audio", 0))
    else:
        print(f"  Skipped: {ref_mp3.name} not found locally")

    _print_summary(results)


def pull():
    """Pull Colab/Kaggle outputs from Drive -> local."""
    print(f"\n{_SEP}")
    print("  PULL  Drive -> local")
    print(_SEP)

    if not DRIVE.exists():
        print(f"\n  ERROR: Drive not found at {DRIVE}")
        sys.exit(1)

    drive_outputs = DRIVE / "outputs"           # G:\My Drive\film_factory\film_factory\outputs
    if not drive_outputs.exists():
        drive_outputs = DRIVE.parent / "outputs"  # fallback: G:\My Drive\film_factory\outputs

    if not drive_outputs.exists():
        print(f"\n  No outputs folder found on Drive (checked {drive_outputs})\n")
        return

    local_outputs = LOCAL / "outputs"
    local_outputs.mkdir(exist_ok=True)

    results = []

    # Voice cloning outputs (Colab Section A)
    print("\n[1/6] outputs/audio/ (Colab voice cloning)...")
    code = _robocopy(
        str(drive_outputs / "audio"), str(local_outputs / "audio"),
        "/E", "/XO",    # /E = include subdirs, /XO = skip older (Drive copy is newer)
    )
    results.append(("outputs/audio/", code))

    # SAM2 layer masks (Colab Section B)
    print("[2/6] outputs/layers/ (Colab SAM2 masks)...")
    code = _robocopy(
        str(drive_outputs / "layers"), str(local_outputs / "layers"),
        "/E", "/XO",
    )
    results.append(("outputs/layers/", code))

    # Kaggle music (if generated via Kaggle server)
    print("[3/6] outputs/music/ (Kaggle music)...")
    code = _robocopy(
        str(drive_outputs / "music"), str(local_outputs / "music"),
        "/E", "/XO",
    )
    results.append(("outputs/music/", code))

    # Alignment cache (word timestamps)
    print("[4/6] outputs/alignment/ (Whisper word timestamps)...")
    code = _robocopy(
        str(drive_outputs / "alignment"), str(local_outputs / "alignment"),
        "/E", "/XO",
    )
    results.append(("outputs/alignment/", code))

    # Wan video clips
    print("[5/6] outputs/video_clips/ (Wan motion clips)...")
    code = _robocopy(
        str(drive_outputs / "video_clips"), str(local_outputs / "video_clips"),
        "/E", "/XO",
    )
    results.append(("outputs/video_clips/", code))

    # Final rendered episodes + context.json
    print("[6/6] outputs/final/ + context.json...")
    code = _robocopy(
        str(drive_outputs / "final"), str(local_outputs / "final"),
        "/E", "/XO",
    )
    results.append(("outputs/final/", code))
    import shutil as _sh
    for fname in ("context.json", "context_after_video_gen.json", "context_after_edit_plan.json"):
        src = drive_outputs / fname
        if src.exists():
            _sh.copy2(src, local_outputs / fname)

    _print_summary(results)


def push_outputs():
    """Push local outputs to Drive so Colab can access them."""
    print(f"\n{_SEP}")
    print("  PUSH OUTPUTS  local -> Drive")
    print(_SEP)

    if not DRIVE.exists():
        print(f"\n  ERROR: Drive not found at {DRIVE}")
        sys.exit(1)

    drive_outputs = DRIVE / "outputs"
    local_outputs = LOCAL / "outputs"

    if not local_outputs.exists():
        print(f"\n  No local outputs/ folder found.\n")
        return

    results = []

    # Scene and reveal images (for Colab SAM2 segmentation)
    for subdir in ["scene_images", "reveal_images", "anchors", "colab_input"]:
        src = local_outputs / subdir
        if src.exists():
            print(f"  Pushing outputs/{subdir}/...")
            code = _robocopy(str(src), str(drive_outputs / subdir), "/E")
            results.append((f"outputs/{subdir}/", code))
        else:
            print(f"  Skipping outputs/{subdir}/ (not found locally)")

    _print_summary(results)


def status():
    """Compare local vs Drive — show files that differ."""
    print(f"\n{_SEP}")
    print("  STATUS  local vs Drive")
    print(_SEP)

    if not DRIVE.exists():
        print(f"\n  ERROR: Drive not found at {DRIVE}\n")
        sys.exit(1)

    checks = [
        ("Root *.py",     LOCAL,          DRIVE,          "*.py"),
        ("agents/ *.py",  LOCAL / "agents", DRIVE / "agents", "*.py"),
        ("colab/ *.ipynb", LOCAL / "colab", DRIVE / "colab", "*.ipynb"),
    ]

    any_diff = False
    for label, src_dir, dst_dir, pattern in checks:
        if not src_dir.exists():
            continue
        for local_f in sorted(src_dir.glob(pattern)):
            drive_f = dst_dir / local_f.name
            local_mt  = local_f.stat().st_mtime
            local_sz  = local_f.stat().st_size
            if not drive_f.exists():
                print(f"  MISSING on Drive : {local_f.relative_to(LOCAL)}")
                any_diff = True
            else:
                drive_mt = drive_f.stat().st_mtime
                drive_sz = drive_f.stat().st_size
                if local_sz != drive_sz or abs(local_mt - drive_mt) > 5:
                    newer = "LOCAL" if local_mt > drive_mt else "DRIVE"
                    print(f"  DIFFERS [{newer} newer] : {local_f.relative_to(LOCAL)}")
                    any_diff = True

    # Reference audio
    ref = LOCAL / "ElevenLabs_Text_to_Speech_audio.mp3"
    if ref.exists():
        drive_ref = DRIVE / ref.name
        if not drive_ref.exists():
            print(f"  MISSING on Drive : {ref.name}")
            any_diff = True
        elif ref.stat().st_size != drive_ref.stat().st_size:
            print(f"  DIFFERS          : {ref.name}")
            any_diff = True

    if not any_diff:
        print("\n  All tracked files match between local and Drive.\n")
    else:
        print(f"\n  Run 'python sync.py push' to update Drive.\n")


def _print_summary(results: list[tuple[str, int]]):
    print(f"\n{_SEP}")
    all_ok = True
    for label, code in results:
        status = "OK" if _ok(code) else f"ERROR (code {code})"
        if not _ok(code):
            all_ok = False
        print(f"  {status:20s} {label}")
    print(_SEP)
    if all_ok:
        print("  Done.\n")
    else:
        print("  Some transfers had errors — check output above.\n")


def _usage():
    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()

    cmd = sys.argv[1].lower()
    if cmd == "push":
        push()
    elif cmd == "pull":
        pull()
    elif cmd == "push-outputs":
        push_outputs()
    elif cmd == "status":
        status()
    else:
        print(f"\n  Unknown command: {cmd}")
        _usage()
