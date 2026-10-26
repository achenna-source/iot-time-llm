# -*- coding: utf-8 -*-
"""Synthetic ORDER runs with known behaviour, to test analysis_order.py end to end (not results)."""
import json
import math
import os
import sys

res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
P = [json.loads(l) for l in open(os.path.join(res, "probes.jsonl"), encoding="utf-8")]
F = [json.loads(l) for l in open(os.path.join(res, "free.jsonl"), encoding="utf-8")]
E = [json.loads(l) for l in open(os.path.join(res, "extract.jsonl"), encoding="utf-8")]
byid = {p["id"]: p for p in P}
sig = lambda z: 1 / (1 + math.exp(-max(min(z, 30), -30)))


def pA(name, p):
    if p["part"] == "cert":
        return None
    if name == "fake-bayes":
        return sig(math.log(p["prior_A"] / (1 - p["prior_A"])) + p["g"] * p["D"] / p["sigma"] ** 2 if p["part"] != "gate_mech" else math.log(p["prior_A"] / (1 - p["prior_A"])))
    if name == "fake-history":
        return 0.9 if p["prior_A"] > 0.5 else (0.1 if p["prior_A"] < 0.5 else 0.5)
    return 0.8 if p["list_a_first"] else 0.2          # fake-position


for name in ("fake-bayes", "fake-history", "fake-position"):
    with open(os.path.join(out, "order_all__%s.jsonl" % name), "w") as f:
        for p in P:
            if p["part"] == "cert":
                cert = {"A_first": 0.1, "B_first": 0.1, "undetermined": 0.8} if abs(p["D"]) <= 2 * p["delta"] else (
                    {"A_first": 0.8, "B_first": 0.1, "undetermined": 0.1} if p["D"] > 0 else {"A_first": 0.1, "B_first": 0.8, "undetermined": 0.1})
                probs = {L: cert[role] for L, role in p["roles"].items()}
            else:
                a = pA(name, p)
                probs = {L: (a if role == "A" else 1 - a) for L, role in p["roles"].items()}
            f.write(json.dumps({"id": p["id"], "model": name, "mass": 0.99, "top": "A", "probs": probs}) + "\n")
    with open(os.path.join(out, "order_free__%s.jsonl" % name), "w") as f:
        for x in F:
            b = byid[x["base"]]
            ev = b["prompt"].split("Question:")[1]
            opts = {b["roles"][l[0]]: l[3:].strip() for l in ev.splitlines() if len(l) > 3 and l[1] == ")"}
            f.write(json.dumps({"id": x["id"], "model": name, "text": '"%s" started this incident.' % opts["A" if pA(name, b) > 0.5 else "B"]}) + "\n")
    with open(os.path.join(out, "order_ext__%s.jsonl" % name), "w") as f:
        for x in E:
            b = byid[x["base"]]
            lines = [l for l in b["prompt"].split("Log:\n")[1].split("\n\n")[0].splitlines()]
            ev = b["prompt"].split("Question:")[1]
            opts = {b["roles"][l[0]]: l[3:].strip() for l in ev.splitlines() if len(l) > 3 and l[1] == ")"}
            tA = [l[:23] for l in lines if opts["A"] in l][0]
            tB = [l[:23] for l in lines if opts["B"] in l][0]
            j = {"timestamp_A": tA, "timestamp_B": tB, "offset_sd_s": b["sigma"], "latency_s": round(b["g"], 2),
                 "history_first": "A" if b["fav_fwd"] else "B", "history_percent": round(100 * b["pi"])}
            f.write(json.dumps({"id": x["id"], "model": name, "text": json.dumps(j)}) + "\n")
print("fake runs written")
