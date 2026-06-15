#!/usr/bin/env python
"""Split a data_build.py output (per-patient subdirs) into flat train/val NPZ dirs.

Test patients (``data/test_ids.txt``) are held out; the remaining patients are
split train/val by a seeded shuffle. Output layout matches what ``data/data.py``
expects: ``<out-dir>/{train,val}/{label}_{patient}_{file}.npz`` (flattened).

For the R30 tighter-span ablation we evaluate on the *canonical* ``data/dataset/test``
for an apples-to-apples comparison, so only train/val are produced here.

Usage:
    python data/split_dataset.py --build-dir data/datasets/<tightspan_build> \
        --out-dir data/dataset_tightspan
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def patient_id(dirname):
    # per-patient dir is "{label}_{patient}", e.g. "positive_UF179".
    return dirname.split("_", 1)[1] if "_" in dirname else dirname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, help="data_build.py output (per-patient subdirs)")
    ap.add_argument("--out-dir", required=True, help="dest root; creates {out}/{train,val}")
    ap.add_argument("--test-ids", default="data/test_ids.txt")
    ap.add_argument("--val-frac", type=float, default=0.23,
                    help="fraction of non-test patients -> val (default ~ the 60/20/20 split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (default: symlink, to save space)")
    args = ap.parse_args()

    with open(args.test_ids) as f:
        test_ids = {line.strip() for line in f if line.strip()}

    build = Path(args.build_dir)
    patient_dirs = sorted(d for d in build.iterdir() if d.is_dir())
    non_test = [d for d in patient_dirs if patient_id(d.name) not in test_ids]
    n_test_present = len(patient_dirs) - len(non_test)

    random.Random(args.seed).shuffle(non_test)
    n_val = int(round(len(non_test) * args.val_frac))
    val, train = non_test[:n_val], non_test[n_val:]

    out = Path(args.out_dir)
    counts = {}
    for split, dirs in [("train", train), ("val", val)]:
        sd = out / split
        sd.mkdir(parents=True, exist_ok=True)
        n_files = 0
        for d in dirs:
            for f in sorted(d.glob("*.npz")):
                dest = sd / f"{d.name}_{f.name}"
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                if args.copy:
                    shutil.copy(f, dest)
                else:
                    os.symlink(f.resolve(), dest)
                n_files += 1
        counts[split] = (len(dirs), n_files)

    print(f"non-test patients: {len(non_test)} "
          f"(test patients present in build, held out: {n_test_present})")
    for split, (npat, nfile) in counts.items():
        print(f"  {split}: {npat} patients, {nfile} npz")
    print(f"-> {out}/{{train,val}}  (test = canonical data/dataset/test, by design)")


if __name__ == "__main__":
    main()
