"""
Prepare Dataset
===============
Splits a raw, already-labeled dataset (two folders: clean/ and dusty/)
into train/val/test folders (70/15/15) under data/.

Expected input structure (create this yourself after downloading from
Kaggle, or after thresholding soiling-loss values into two classes):

    raw_data/
    ├── clean/
    │   ├── img1.jpg
    │   └── ...
    └── dusty/
        ├── img1.jpg
        └── ...

Run from the project root:
    python scripts/prepare_dataset.py --raw_dir raw_data --out_dir data
"""

import os
import shutil
import random
import argparse


def split_dataset(raw_dir, out_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    if not classes:
        raise ValueError(f"No class subfolders found in {raw_dir}")

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(files)

        n = len(files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_files in splits.items():
            split_dir = os.path.join(out_dir, split, cls.lower())
            os.makedirs(split_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(split_dir, f))

        print(f"[{cls}] total={n} -> train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

    print(f"\nDone. Split dataset written to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split raw dataset into train/val/test")
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Path to raw dataset with class subfolders (e.g. clean/, dusty/)")
    parser.add_argument("--out_dir", type=str, default="data",
                         help="Output directory for the split dataset (default: data)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.raw_dir, args.out_dir, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
