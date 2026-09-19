"""
Prepare Dataset: verify, de-duplicate, and split
=================================================
Turns the raw Kaggle "Faulty solar panel" folder (one sub-folder per class,
possibly with nested sub-folders such as Bird-drop/New/) into a clean,
de-duplicated, stratified train/val/test split.

Pipeline
--------
1. Recursively collect every image under <raw_dir>/<Class>/...
2. Drop files PIL cannot open.
3. Exact de-duplication (MD5 of file bytes).
4. Near-duplicate de-duplication (64-bit difference hash, Hamming distance
   <= --phash_thr). Within a class the highest-resolution copy is kept.
   Groups that span more than one class are ambiguous and are dropped
   entirely.
5. Stratified 70/15/15 split per class (deterministic with --seed).
6. Images are re-saved as RGB JPEG with EXIF orientation applied, which also
   strips broken metadata that used to trigger libjpeg/libpng warnings.
7. A JSON report of every stage is written to --report.

Run from the project root:
    python scripts/prepare_dataset.py --raw_dir Faulty_solar_panel --out_dir data
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASS_NAMES, normalize_class_name  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_bits(img: Image.Image, size: int = 8) -> np.ndarray:
    """64-bit difference hash: robust to resizing / re-encoding."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def collect_images(raw_dir: Path):
    """Return list of dicts {path, cls} for every image under raw_dir."""
    records = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        cls = normalize_class_name(class_dir.name)
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"path": p, "cls": cls})
    return records


def verify_and_hash(records):
    """Open each image once; attach md5, dhash and pixel count. Drop unreadable."""
    good, broken = [], []
    for r in records:
        try:
            with Image.open(r["path"]) as im:
                im.load()
                im = ImageOps.exif_transpose(im)
                r["pixels"] = im.width * im.height
                r["dhash"] = dhash_bits(im)
        except Exception as e:  # noqa: BLE001
            broken.append({"path": str(r["path"]), "error": str(e)})
            continue
        r["md5"] = md5_of_file(r["path"])
        good.append(r)
    return good, broken


def exact_dedup(records):
    seen = {}
    kept, removed = [], []
    for r in records:
        if r["md5"] in seen:
            removed.append({"path": str(r["path"]), "cls": r["cls"],
                            "duplicate_of": str(seen[r["md5"]]["path"])})
        else:
            seen[r["md5"]] = r
            kept.append(r)
    return kept, removed


def near_dedup(records, thr: int):
    """Group by dHash Hamming distance <= thr. Keep best per single-class group,
    drop whole group when it spans multiple classes (label conflict)."""
    n = len(records)
    H = np.stack([r["dhash"] for r in records])          # (n, 64)
    uf = UnionFind(n)
    # pairwise Hamming distance, vectorised per row to keep memory small
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.where(d <= thr)[0]:
            uf.union(i, i + 1 + int(j))

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    kept, removed_near, removed_conflict = [], [], []
    for members in groups.values():
        if len(members) == 1:
            kept.append(records[members[0]])
            continue
        classes = {records[i]["cls"] for i in members}
        if len(classes) > 1:
            for i in members:
                removed_conflict.append({"path": str(records[i]["path"]),
                                         "cls": records[i]["cls"],
                                         "conflicting_classes": sorted(classes)})
            continue
        best = max(members, key=lambda i: records[i]["pixels"])
        kept.append(records[best])
        for i in members:
            if i != best:
                removed_near.append({"path": str(records[i]["path"]),
                                     "cls": records[i]["cls"],
                                     "near_duplicate_of": str(records[best]["path"])})
    return kept, removed_near, removed_conflict


def stratified_split(records, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for r in records:
        by_cls[r["cls"]].append(r)

    splits = {"train": [], "val": [], "test": []}
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda r: str(r["path"]))
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]
    return splits


def write_split(splits, out_dir: Path, jpeg_quality: int):
    counters = Counter()
    for split, items in splits.items():
        for r in items:
            cls = r["cls"]
            dst_dir = out_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            counters[(split, cls)] += 1
            dst = dst_dir / f"{cls}_{counters[(split, cls)]:04d}.jpg"
            with Image.open(r["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.save(dst, format="JPEG", quality=jpeg_quality, optimize=True)
            r["dst"] = str(dst.relative_to(out_dir))


def counts_by_class(records):
    return dict(sorted(Counter(r["cls"] for r in records).items()))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Verify, de-duplicate and split the raw dataset.")
    ap.add_argument("--raw_dir", default="Faulty_solar_panel", help="Raw dataset with one folder per class")
    ap.add_argument("--out_dir", default="data", help="Destination for train/val/test folders")
    ap.add_argument("--report", default="outputs/dataset_report.json", help="Where to write the JSON report")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--phash_thr", type=int, default=4, help="Max Hamming distance (of 64 bits) to call two images near-duplicates")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite existing train/val/test folders in out_dir")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    if not raw_dir.is_dir():
        sys.exit(f"ERROR: raw_dir '{raw_dir}' not found.")
    existing = [d for d in ("train", "val", "test") if (out_dir / d).exists()]
    if existing and not args.force:
        sys.exit(f"ERROR: {out_dir}/{{{','.join(existing)}}} already exist. Re-run with --force to overwrite.")
    for d in existing:
        shutil.rmtree(out_dir / d)

    t0 = time.time()
    print(f"[1/6] Collecting images under {raw_dir} ...")
    records = collect_images(raw_dir)
    stage_counts = {"collected": counts_by_class(records)}
    unknown = sorted({r["cls"] for r in records} - set(CLASS_NAMES))
    if unknown:
        sys.exit(f"ERROR: unexpected class folders {unknown}. Expected {CLASS_NAMES}.")
    print(f"      {len(records)} files in {len(stage_counts['collected'])} classes")

    print("[2/6] Verifying and hashing ...")
    records, broken = verify_and_hash(records)
    stage_counts["readable"] = counts_by_class(records)
    print(f"      {len(records)} readable, {len(broken)} broken")

    print("[3/6] Exact de-duplication (MD5) ...")
    records, removed_exact = exact_dedup(records)
    stage_counts["after_exact_dedup"] = counts_by_class(records)
    print(f"      removed {len(removed_exact)} exact duplicates -> {len(records)}")

    print(f"[4/6] Near-duplicate de-duplication (dHash, Hamming <= {args.phash_thr}) ...")
    records, removed_near, removed_conflict = near_dedup(records, args.phash_thr)
    stage_counts["after_near_dedup"] = counts_by_class(records)
    print(f"      removed {len(removed_near)} near-duplicates, "
          f"{len(removed_conflict)} label-conflict images -> {len(records)}")

    print(f"[5/6] Stratified split {args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1 - args.train_ratio - args.val_ratio:.0%} (seed={args.seed}) ...")
    splits = stratified_split(records, args.train_ratio, args.val_ratio, args.seed)

    print(f"[6/6] Writing JPEGs to {out_dir} ...")
    write_split(splits, out_dir, args.jpeg_quality)
    split_counts = {s: counts_by_class(items) for s, items in splits.items()}
    for s, c in split_counts.items():
        print(f"      {s:5s}: total={sum(c.values()):4d}  {c}")

    report = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "class_names": CLASS_NAMES,
        "params": {"train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
                   "phash_thr": args.phash_thr, "seed": args.seed},
        "stage_counts": stage_counts,
        "totals": {k: sum(v.values()) for k, v in stage_counts.items()},
        "split_counts": split_counts,
        "removed": {
            "broken": broken,
            "exact_duplicates": removed_exact,
            "near_duplicates": removed_near,
            "label_conflicts": removed_conflict,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone in {report['elapsed_sec']}s. Report -> {report_path}")


if __name__ == "__main__":
    main()
