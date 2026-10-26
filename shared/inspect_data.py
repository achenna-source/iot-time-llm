# -*- coding: utf-8 -*-
"""List the files of the attached real datasets and print a few lines of each (Kaggle CPU)."""
import glob
import os

for root in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/datasets/*/*")):
    print("=" * 30, root)
    files = sorted(glob.glob(root + "/**/*", recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    print(len(files), "files")
    for f in files[:40]:
        print("  %10d  %s" % (os.path.getsize(f), f))
    shown = 0
    for f in files:
        if shown >= 6:
            break
        if f.endswith((".csv", ".txt", ".dat", ".tsv")) or "." not in os.path.basename(f):
            try:
                with open(f, "r", errors="replace") as fh:
                    head = [next(fh).rstrip() for _ in range(4)]
                print("--", f)
                for h in head:
                    print("   ", h[:300])
                shown += 1
            except Exception as e:
                print("   (unreadable)", e)
