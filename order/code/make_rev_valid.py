# -*- coding: utf-8 -*-
"""Stratified random validation subsets for ORDER (replaces the systematic k%8 / k%16 selections, which picked
fixed design cells). Strata: prior (2) x favoured direction (2) x line order (2) x latency ratio (3) = 24 cells.
free : 9 items per cell (216); the question names no event, the answer must copy the event name from the log
ext  : 5 items per cell (120); same extraction instruction as before
"""
import json
import os
import random
import sys

RES = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(RES, "probes.jsonl"), encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]
rnd = random.Random(9090)
cells = {}
for r in main:
    cells.setdefault((r["pi"], r["fav_fwd"], r["list_a_first"], r["gamma"]), []).append(r)


def body(p):
    return p.split("\n\nQuestion:")[0]


def events(r):
    txt = r["prompt"].split("Question:")[1]
    return {r["roles"][l[0]]: l[3:].strip() for l in txt.splitlines() if len(l) > 3 and l[1] == ")"}


free, ext = [], []
for key in sorted(cells):
    pool = cells[key][:]
    rnd.shuffle(pool)
    for r in pool[:9]:
        free.append({"id": "ordv-free-%s" % r["id"], "base": r["id"], "max_new": 60, "messages": [{"role": "user", "content": body(r["prompt"]) + (
            "\n\nWhich event started this incident? Reply with the name of that event copied exactly as it appears in "
            "the log line, and nothing else.")}]})
    for r in pool[9:14]:
        ev = events(r)
        ext.append({"id": "ordv-ext-%s" % r["id"], "base": r["id"], "max_new": 160, "messages": [{"role": "user", "content": body(r["prompt"]) + (
            "\n\nDo not decide yet. Event A is \"%s\" and event B is \"%s\". Extract the facts a root-cause analysis "
            "depends on and reply with only a JSON object with these keys: \"timestamp_A\" and \"timestamp_B\" (the device "
            "timestamps of the two log lines, copied exactly), \"offset_sd_s\" (clock offset standard deviation in seconds), "
            "\"latency_s\" (typical delay between the two events in seconds), \"history_first\" (\"A\", \"B\" or \"none\": "
            "the event the maintenance history says usually comes first), \"history_percent\" (number, 50 if none)." % (ev["A"], ev["B"]))}]})
for name, data in (("rev_free", free), ("rev_ext", ext)):
    with open(os.path.join(RES, name + ".jsonl"), "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(name, len(data))
