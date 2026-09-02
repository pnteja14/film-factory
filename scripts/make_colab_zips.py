"""
Creates film_factory_code.zip and film_factory_outputs.zip using Python's zipfile
so archive entries always use forward-slash separators — required for Linux extraction.
PowerShell's Compress-Archive stores Windows backslash paths which break on Colab.

Run from the project root:
    python scripts/make_colab_zips.py
"""
import zipfile
import sys
from pathlib import Path

ROOT    = Path("C:/Users/tejap/OneDrive/Desktop/film_factory")
OUT_DIR = Path("C:/Users/tejap/OneDrive/Desktop")

EXCLUDE = {"__pycache__", "venv", ".git", "node_modules", ".env"}
EXCLUDE_EXTS = {".pyc", ".pyo"}


def _skip(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE:
            return True
    return path.suffix in EXCLUDE_EXTS


# ── Code zip ──────────────────────────────────────────────────────────────────
code_zip = OUT_DIR / "film_factory_code.zip"
code_zip.unlink(missing_ok=True)
print(f"Creating {code_zip} ...")

with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for src_dir in ["agents", "utils", "colab"]:
        for f in sorted((ROOT / src_dir).rglob("*")):
            if _skip(f) or not f.is_file():
                continue
            arcname = f.relative_to(ROOT).as_posix()   # always forward slashes
            zf.write(f, arcname)
    for fname in ["context.py", "orchestrator.py"]:
        f = ROOT / fname
        if f.exists():
            zf.write(f, fname)

mb = code_zip.stat().st_size / 1024 / 1024
print(f"  Done: {mb:.1f} MB")

# ── Outputs zip ───────────────────────────────────────────────────────────────
outputs_zip = OUT_DIR / "film_factory_outputs.zip"
outputs_zip.unlink(missing_ok=True)
print(f"Creating {outputs_zip} ...")

SUBDIRS = ["audio", "anchors", "scene_images", "reveal_images", "insert_images", "alignment"]

with zipfile.ZipFile(outputs_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    ctx = ROOT / "outputs" / "context.json"
    if ctx.exists():
        zf.write(ctx, "outputs/context.json")
    for sub in SUBDIRS:
        subdir = ROOT / "outputs" / sub
        if not subdir.exists():
            continue
        for f in sorted(subdir.rglob("*")):
            if _skip(f) or not f.is_file():
                continue
            arcname = f.relative_to(ROOT).as_posix()
            zf.write(f, arcname)

mb = outputs_zip.stat().st_size / 1024 / 1024
print(f"  Done: {mb:.1f} MB")

print()
print("Upload both files to Google Drive at:")
print("  My Drive/film_factory/film_factory/film_factory_code.zip")
print("  My Drive/film_factory/film_factory/film_factory_outputs.zip")
