"""
Dataset Cleaner & Normalizer
==============================
Re-saves all images in a folder tree through PIL to:
  - Strip bad EXIF metadata / extra JPEG bytes (fixes libpng/libjpeg warnings)
  - Remove broken iCCP/sRGB profiles from PNG files
  - Remove truly unreadable corrupt images
  - Skip images that are too small (< MIN_PIXELS)

Run from project root BEFORE training:
    python scripts/clean_dataset.py --root raw_data
    python scripts/clean_dataset.py --root data
"""

import sys
# Force UTF-8 output on Windows to avoid cp1252 emoji errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import io
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageFile
    # Allow PIL to read truncated images (instead of raising errors)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    raise ImportError("Pillow not found. Install with: pip install Pillow")

MIN_PIXELS = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def normalize_image(img_path: Path) -> tuple:
    """
    Open the image and re-save it through PIL to produce a clean file.
    - Strips bad EXIF, bad ICC profiles, extraneous bytes
    - Returns (action: str, detail: str)
      action: 'normalized', 'removed', 'skipped'
    """
    try:
        with Image.open(img_path) as img:
            img.load()
            w, h = img.size
    except Exception as e:
        return "removed", f"cannot open: {e}"

    if w < MIN_PIXELS or h < MIN_PIXELS:
        return "removed", f"too small ({w}x{h})"

    # Re-save through PIL, stripping metadata
    ext = img_path.suffix.lower()
    try:
        with Image.open(img_path) as img:
            img.load()
            # Convert to RGB to drop alpha / palette + ICC
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            save_kwargs = {}
            if ext in (".jpg", ".jpeg"):
                save_kwargs = {"format": "JPEG", "quality": 95, "optimize": True}
            elif ext == ".png":
                save_kwargs = {"format": "PNG", "optimize": True}
                # Explicitly drop icc_profile
                img.info.pop("icc_profile", None)
            else:
                save_kwargs = {"format": img.format or "JPEG"}

            # Write to a buffer first, then back to disk (atomic-ish)
            buf = io.BytesIO()
            img.save(buf, **save_kwargs)
            buf.seek(0)
            with open(img_path, "wb") as f:
                f.write(buf.read())

        return "normalized", "ok"
    except Exception as e:
        return "removed", f"re-save failed: {e}"


def clean_directory(root: Path, dry_run: bool = False) -> dict:
    stats = {
        "scanned": 0,
        "normalized": 0,
        "removed": 0,
        "removed_files": [],
    }

    image_paths = sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(f"\nFound {len(image_paths)} images under '{root}'.")
    if dry_run:
        print("  DRY RUN mode — no files will be changed or deleted.\n")
    else:
        print("  Live mode — images will be re-saved and bad ones deleted.\n")

    for i, img_path in enumerate(image_paths, 1):
        stats["scanned"] += 1

        if dry_run:
            # In dry-run just validate
            try:
                with Image.open(img_path) as img:
                    img.load()
                    w, h = img.size
                if w < MIN_PIXELS or h < MIN_PIXELS:
                    action, detail = "removed", f"too small ({w}x{h})"
                else:
                    action, detail = "ok", "readable"
            except Exception as e:
                action, detail = "removed", str(e)
        else:
            action, detail = normalize_image(img_path)

        if action == "removed":
            rel = str(img_path.relative_to(root))
            stats["removed"] += 1
            stats["removed_files"].append(rel)
            print(f"  [CORRUPT]  {rel}  [{detail}]")
            if not dry_run:
                try:
                    img_path.unlink()
                except Exception:
                    pass
        elif action == "normalized":
            stats["normalized"] += 1

        if i % 300 == 0 or i == len(image_paths):
            pct = 100 * i // len(image_paths)
            print(f"  [{i:>4}/{len(image_paths)}] {pct:3}%  "
                  f"re-saved={stats['normalized']}  "
                  f"removed={stats['removed']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normalize images (strip bad metadata) and remove corrupt ones."
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory to scan (e.g. raw_data or data)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report issues without changing any files"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: '{root}' does not exist.")
        return

    print("=" * 62)
    print("  Solar Panel Dataset Normalizer")
    print(f"  Root : {root.resolve()}")
    print(f"  Mode : {'DRY RUN (read-only)' if args.dry_run else 'LIVE (re-saving all images)'}")
    print("=" * 62)

    stats = clean_directory(root, dry_run=args.dry_run)

    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Images scanned   : {stats['scanned']}")
    print(f"  Re-saved (clean) : {stats['normalized']}")
    print(f"  Removed (corrupt): {stats['removed']}")

    if stats["removed_files"]:
        print("\n  Removed files:")
        for f in stats["removed_files"]:
            print(f"    ✗ {f}")

    if not args.dry_run:
        print("\n  ✅ Done! Now re-run prepare_dataset.py then main.py for clean training.")
    else:
        print("\n  Dry-run complete. Run without --dry-run to apply fixes.")


if __name__ == "__main__":
    main()
