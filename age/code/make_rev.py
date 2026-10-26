# -*- coding: utf-8 -*-
"""Revision arms for AGE (after the simulated review), built from probes.jsonl without changing its ids.

reason  : one probe per (volatility, age, price) cell plus 24 fresh-reading probes, with step-by-step reasoning
          before a final "Answer: X" (generation)
rule    : the same cells in three option rotations, with the optimal decision rule stated in the prompt (letters)
pgiven  : the same, with the staleness probability itself stated (letters)
share   : the same, with the time share of each state stated (letters); removes the hidden psi = 1/2
half    : main grid restricted to six scenarios (all rotations), for the larger scale-check model
"""
import json
import os
import random
import sys

RES = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(RES, "probes.jsonl"), encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]
rnd = random.Random(4242)
cells = {}
for r in main:
    cells.setdefault((r["q_day"], r["age_s"], r["r"]), []).append(r)
COT = ("Think step by step about the expected cost of each option, then give your final answer on the last line "
       "in the form \"Answer: X\".")


def with_text(r, insert=None, cot=False):
    p = r["prompt"]
    if insert:
        p = p.replace("\nAgent memory:", " " + insert + "\nAgent memory:", 1)
    if cot:
        p = p.replace("Answer with a single letter.", COT)
    return p


out = {"reason": [], "rule": [], "pgiven": [], "share": []}
for key in sorted(cells):
    pick = rnd.choice(cells[key])
    base = (pick["scen"], pick["cond_true"])
    trio = [r for r in cells[key] if (r["scen"], r["cond_true"]) == base]
    out["reason"].append(dict(pick, id="ager-reason-%s" % pick["id"], base=pick["id"], max_new=512,
                              prompt=with_text(pick, cot=True)))
    for r in trio:
        rule = ("Decision rule: with q the number of state changes per day and D the age of the reading in days, the "
                "probability that the remembered reading is outdated is p = (1 - exp(-2 q D)) / 2; call read_state first "
                "if and only if p is larger than the read cost divided by the wrong-command cost.")
        out["rule"].append(dict(r, id="ager-rule-%s" % r["id"], base=r["id"], prompt=with_text(r, rule)))
        pv = ("%.3f" % r["p_change"]) if r["p_change"] >= 0.001 else "less than 0.001"
        pg = "The device registry estimates that this reading is now outdated with probability %s." % pv
        out["pgiven"].append(dict(r, id="ager-pgiven-%s" % r["id"], base=r["id"], prompt=with_text(r, pg)))
        out["share"].append(dict(r, id="ager-share-%s" % r["id"], base=r["id"],
                                 prompt=with_text(r, "It spends about half of the time in each state.")))
fresh = [r for r in rows if r["part"] == "gate_fresh"]
for r in rnd.sample(fresh, 24):
    out["reason"].append(dict(r, id="ager-reason-%s" % r["id"], base=r["id"], max_new=512, prompt=with_text(r, cot=True)))
half = [r for r in main if r["scen"] < 6]
for arm, data in list(out.items()) + [("half", half)]:
    with open(os.path.join(RES, "rev_%s.jsonl" % arm), "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(arm, len(data))
print(out["pgiven"][0]["prompt"][:900])
