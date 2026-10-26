# -*- coding: utf-8 -*-
"""Forced final answers for the reasoning arms (local, light).

Both reasoning arms give the model a budget of 512 new tokens. When a generation
ends without a parsable "Answer: X", the model's own reasoning is appended to the assistant turn followed by
"\n\nAnswer:", and the next-token distribution over the option letters is read (one forward pass, run on Kaggle
with the plan "force"). Writes shared/force/force_{age,order}__<model>.jsonl.
python make_force.py <runs_dir>
"""
import glob
import json
import os
import re
import sys

RUNS = sys.argv[1]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force")
os.makedirs(OUT, exist_ok=True)
ANS = re.compile(r"answer\s*[:\-]?\s*\**\s*\(?([A-C])\b", re.I)
for paper in ("age", "order"):
    REV = {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, paper, "results", "rev_reason.jsonl"), encoding="utf-8"))}
    for fn in sorted(glob.glob(os.path.join(RUNS, "**", "%s_reason__*.jsonl" % paper), recursive=True)):
        short = os.path.basename(fn).split("__")[1][:-6]
        rows = {}
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            rows[r["id"]] = r
        out = []
        for i, r in rows.items():
            if i not in REV or ANS.findall(r["text"]):
                continue
            b = REV[i]
            out.append({"id": "force-" + i, "base": i, "prompt": b["prompt"], "letters": sorted(b["roles"].keys()),
                        "prefix": r["text"].rstrip() + "\n\nAnswer:"})
        with open(os.path.join(OUT, "force_%s__%s.jsonl" % (paper, short)), "w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(paper, short, "generations", len(rows), "to force", len(out))
