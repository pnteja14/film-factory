#!/usr/bin/env python3
"""
Sync local outputs/ to Google Drive for Colab consumption.

Usage:
    python sync_to_drive.py             # auto-detect Drive path
    python sync_to_drive.py --dry-run   # show what would be synced
    python sync_to_drive.py --path "G:\\My Drive\\film_factory"

Set GDRIVE_PATH in .env to skip auto-detection permanently.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# ── Locate project root ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = ROOT / "outputs"
COLAB_DIR   = ROOT / "colab"
REF_AUDIO   = ROOT / "reference_voice.wav"

# ── Load .env ─────────────────────────────────────────────────────────────────
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def find_gdrive_root() -> Path | None:
    """Auto-detect the Google Drive for Desktop local root on Windows."""
    candidates = []

    # 1. Registry lookup (most reliable)
    try:
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for subkey in (
                r"Software\Google\DriveFS",
                r"Software\Google\Drive",
            ):
                try:
                    key = winreg.OpenKey(hive, subkey)
                    for value_name in ("MountPoint", "Path", "InstallLocation"):
                        try:
                            val, _ = winreg.QueryValueEx(key, value_name)
                            candidates.append(Path(val))
                        except FileNotFoundError:
                            pass
                    winreg.CloseKey(key)
                except FileNotFoundError:
                    pass
    except ImportError:
        pass

    # 2. Common virtual drive letters (Drive for Desktop default)
    for letter in "GHIJKLMNOPQRSTUVWXYZ":
        for suffix in ("My Drive", ""):
            p = Path(f"{letter}:/") / suffix
            candidates.append(p)

    # 3. Common folder paths
    home = Path.home()
    candidates += [
        home / "Google Drive" / "My Drive",
        home / "Google Drive",
        home / "My Drive",
        Path("C:/Users") / home.name / "Google Drive" / "My Drive",
    ]

    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except (PermissionError, OSError):
            pass

    return None


def find_film_factory_on_drive(gdrive_root: Path) -> Path | None:
    """
    Locate the film_factory folder inside the Drive root.
    Handles both single-nesting (film_factory/) and
    double-nesting (film_factory/film_factory/).
    """
    # Direct child named film_factory
    direct = gdrive_root / "film_factory"
    if direct.exists():
        # Check for double-nesting
        nested = direct / "film_factory"
        if nested.exists():
            print(f"  Detected double-nesting: {nested}")
            return nested
        print(f"  Detected single-nesting: {direct}")
        return direct

    # Recursive search one level deep (handles unexpected parent folders)
    for child in gdrive_root.iterdir():
        if child.is_dir() and child.name == "film_factory":
            return child

    return None


def robocopy_sync(src: Path, dst: Path, dry_run: bool = False, mirror: bool = False) -> int:
    """
    Use robocopy to sync src → dst (Windows built-in, handles large trees).
    Default: /E  = copy new/updated files only, keep extras in dst
    --mirror:    /MIR = also delete files in dst that don't exist in src
    /R:2  = 2 retries on failure
    /W:3  = 3s wait between retries
    /NP   = no progress percentage
    /NFL  = no file list (cleaner output)
    /NDL  = no directory list
    """
    cmd = [
        "robocopy",
        str(src), str(dst),
        "/MIR" if mirror else "/E",
        "/R:2", "/W:3",
        "/NP", "/NFL", "/NDL",
    ]
    if dry_run:
        cmd.append("/L")  # /L = list only, no actual copy

    result = subprocess.run(cmd)
    # robocopy exit codes 0-7 are success (0=no change, 1=copied, etc.)
    return result.returncode


def _pull(dry: bool = False):
    """Pull Drive → local: copy Colab-generated outputs back to project outputs/."""
    print("Auto-detecting Google Drive location...")
    gdrive_root = find_gdrive_root()
    if not gdrive_root:
        print("✗ Could not find Google Drive. Set GDRIVE_PATH in .env and retry.")
        sys.exit(1)

    drive_root = find_film_factory_on_drive(gdrive_root)
    if not drive_root:
        print("✗ film_factory folder not found on Drive.")
        sys.exit(1)

    src_outputs = drive_root / "outputs"
    dst_outputs = OUTPUTS_DIR

    print(f"\nPulling outputs/")
    print(f"  FROM: {src_outputs}")
    print(f"    TO: {dst_outputs}")

    if not dst_outputs.exists() and not dry:
        dst_outputs.mkdir(parents=True, exist_ok=True)

    rc = robocopy_sync(src_outputs, dst_outputs, dry_run=dry)
    if rc <= 7:
        print(f"  ✓ outputs/ pulled (robocopy exit {rc})")
    else:
        print(f"  ✗ robocopy error (exit {rc})")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("Pull complete. New files are now in local outputs/.")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(description="Sync outputs/ to/from Google Drive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be synced without copying")
    parser.add_argument("--mirror", action="store_true",
                        help="Also delete files on Drive that no longer exist locally (use with care)")
    parser.add_argument("--pull", action="store_true",
                        help="Pull Drive → local (copy Colab outputs back to project)")
    parser.add_argument("--path", default="",
                        help="Override Google Drive film_factory path")
    args = parser.parse_args()

    dry = args.dry_run
    mirror = args.mirror
    if dry:
        print("DRY RUN — no files will be copied\n")

    if args.pull:
        _pull(dry)
        return

    # ── Resolve target path ───────────────────────────────────────────────────
    override = args.path or os.getenv("GDRIVE_PATH", "")
    if override:
        target_root = Path(override)
        print(f"Using GDRIVE_PATH: {target_root}")
    else:
        print("Auto-detecting Google Drive location...")
        gdrive_root = find_gdrive_root()
        if not gdrive_root:
            print(
                "\n✗ Could not find Google Drive on this machine.\n"
                "  Set GDRIVE_PATH in .env, e.g.:\n"
                '  GDRIVE_PATH=G:\\My Drive\\film_factory\n'
                "  Then re-run: python sync_to_drive.py"
            )
            sys.exit(1)
        print(f"  Google Drive root: {gdrive_root}")

        target_root = find_film_factory_on_drive(gdrive_root)
        if not target_root:
            # Create the folder
            target_root = gdrive_root / "film_factory"
            print(f"  Creating folder: {target_root}")
            if not dry:
                target_root.mkdir(parents=True, exist_ok=True)

    # ── Sync outputs/ ─────────────────────────────────────────────────────────
    src_outputs = OUTPUTS_DIR
    dst_outputs = target_root / "outputs"

    if not src_outputs.exists():
        print(f"✗ Local outputs/ not found at {src_outputs}")
        sys.exit(1)

    print(f"\nSyncing outputs/")
    print(f"  FROM: {src_outputs}")
    print(f"    TO: {dst_outputs}")
    if not dry:
        dst_outputs.mkdir(parents=True, exist_ok=True)

    rc = robocopy_sync(src_outputs, dst_outputs, dry_run=dry, mirror=mirror)
    if rc <= 7:
        print(f"  ✓ outputs/ synced (robocopy exit {rc})")
    else:
        print(f"  ✗ robocopy error (exit {rc})")
        sys.exit(1)

    # ── Sync colab/ notebook ──────────────────────────────────────────────────
    if COLAB_DIR.exists():
        dst_colab = target_root / "colab"
        print(f"\nSyncing colab/")
        print(f"  FROM: {COLAB_DIR}")
        print(f"    TO: {dst_colab}")
        if not dry:
            dst_colab.mkdir(parents=True, exist_ok=True)
        rc2 = robocopy_sync(COLAB_DIR, dst_colab, dry_run=dry)
        if rc2 <= 7:
            print(f"  ✓ colab/ synced (robocopy exit {rc2})")
        else:
            print(f"  ✗ robocopy error on colab/ (exit {rc2})")

    # ── Sync reference_voice.wav if present ───────────────────────────────────
    if REF_AUDIO.exists():
        dst_ref = target_root / "reference_voice.wav"
        print(f"\nSyncing reference_voice.wav → {dst_ref}")
        if not dry:
            import shutil
            shutil.copy2(REF_AUDIO, dst_ref)
            print("  ✓ reference_voice.wav copied")
        else:
            print("  (dry-run) reference_voice.wav would be copied")
    else:
        print(f"\n  reference_voice.wav not found at {REF_AUDIO} — skipping")

    # ── Print Colab paths ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Sync complete. Use these paths in Colab cell 2:")
    # Convert local path to Drive path hint
    drive_outputs_hint = str(target_root / "outputs").replace("\\", "/")
    ref_hint = str(target_root / "reference_voice.wav").replace("\\", "/")
    # Strip local drive prefix, show relative from My Drive
    for prefix in ("G:/My Drive", "H:/My Drive", str(Path.home() / "Google Drive" / "My Drive").replace("\\", "/")):
        if drive_outputs_hint.startswith(prefix):
            rel = drive_outputs_hint[len(prefix):]
            print(f"  DRIVE_OUTPUTS = '/content/drive/MyDrive{rel}'")
            ref_rel = ref_hint[len(prefix):]
            print(f"  REFERENCE_AUDIO_PATH = '/content/drive/MyDrive{ref_rel}'")
            break
    else:
        print(f"  Target: {target_root}")
        print("  Update DRIVE_OUTPUTS in Colab cell 2 to match the path above.")
    print("─" * 60)


if __name__ == "__main__":
    main()
