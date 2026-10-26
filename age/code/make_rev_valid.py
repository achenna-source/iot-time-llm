# -*- coding: utf-8 -*-
"""Stratified extraction subset for AGE (replaces the subset restricted to r = 0.1, condition true, rotation 0).
Strata: price (4) x remembered state (2) x volatility (0.25, 4, 64 changes per day) = 24 cells, 5 items each."""
import json
import os
import random
import sys

RES = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(RES, "probes.jsonl"), encoding="utf-8")]
main = [r for r in rows if r["part"] == "main" and r["q_day"] in (0.25, 4, 64)]
rnd = random.Random(8080)
cells = {}
for r in main:
    cells.setdefault((r["r"], r["cond_true"], r["q_day"]), []).append(r)
out = []
for key in sorted(cells):
    for r in rnd.sample(cells[key], 5):
        head, rest = r["prompt"].split("Device sheet:", 1)
        situ = "Device sheet:" + rest.split("\n\nWhat do you do next?")[0]
        out.append({"id": "agev-ext-%s" % r["id"], "base": r["id"], "max_new": 140, "messages": [{"role": "user", "content": head.strip() + "\n" + situ.strip() + (
            "\n\nDo not decide yet. Extract the facts the decision depends on and reply with only a JSON object with these "
            "keys: \"as_of\" (time of the remembered reading, ISO 8601), \"current_time\" (ISO 8601), \"changes_per_day\" "
            "(number), \"read_cost\" (number), \"wrong_command_cost\" (number), \"remembered_state\" (string).")}]})
with open(os.path.join(RES, "rev_ext.jsonl"), "w", encoding="utf-8") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("rev_ext", len(out))
