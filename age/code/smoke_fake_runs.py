# -*- coding: utf-8 -*-
"""Synthetic runs with known behaviour, to test analysis_age.py end to end (not results).

fake-rational : P(read) = sigmoid(2 (ln p - ln r)), letters rotated as in the corpus
fake-ageonly  : P(read) depends on age only (volatility ignored), so ratio_q_age should be ~ 0
fake-always   : always reads, so J = 0
"""
import json
import math
import os
import sys

res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
P = [json.loads(l) for l in open(os.path.join(res, "probes.jsonl"), encoding="utf-8")]
T = [json.loads(l) for l in open(os.path.join(res, "toolcall.jsonl"), encoding="utf-8")]
E = [json.loads(l) for l in open(os.path.join(res, "extract.jsonl"), encoding="utf-8")]
byid = {p["id"]: p for p in P}
sig = lambda z: 1 / (1 + math.exp(-z))


def policy(name, p):
    if name == "fake-always":
        return 0.999
    if name == "fake-ageonly":
        return sig(1.5 * (math.log(max(p["age_s"], 60)) - math.log(10800)))
    pc = max(p["p_change"], 1e-6)
    return sig(2 * (math.log(pc) - math.log(p["r"])))


for name in ("fake-rational", "fake-ageonly", "fake-always"):
    with open(os.path.join(out, "age_all__%s.jsonl" % name), "w") as f:
        for p in P:
            pr = policy(name, p)
            probs = {}
            for L, role in p["roles"].items():
                probs[L] = pr if role == "read" else ((1 - pr) * 0.95 if role == p["mem_role"] else (1 - pr) * 0.05)
            f.write(json.dumps({"id": p["id"], "model": name, "mass": 0.99, "top": "A", "probs": probs}) + "\n")
    with open(os.path.join(out, "age_tool__%s.jsonl" % name), "w") as f:
        for t in T:
            b = byid[t["base"]]
            txt = '{"name": "read_state"}' if policy(name, b) > 0.5 else ('{"name": "execute_command"}' if b["cond_true"] else "No action needed.")
            f.write(json.dumps({"id": t["id"], "model": name, "text": txt}) + "\n")
    with open(os.path.join(out, "age_ext__%s.jsonl" % name), "w") as f:
        for e in E:
            b = byid[e["base"]]
            pr = b["prompt"]
            asof = pr.split('"as_of": "')[1].split('"')[0]
            now = pr.split("Current time: ")[1].split("\n")[0]
            j = {"as_of": asof, "current_time": now, "changes_per_day": b["q_day"], "read_cost": b["c"],
                 "wrong_command_cost": b["L"], "remembered_state": "x"}
            f.write(json.dumps({"id": e["id"], "model": name, "text": json.dumps(j)}) + "\n")
print("fake runs written")
