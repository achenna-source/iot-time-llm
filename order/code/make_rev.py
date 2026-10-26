# -*- coding: utf-8 -*-
"""Revision arms for ORDER (after the simulated review), built from probes.jsonl without changing its ids.

reason  : every 9th main item and every 5th certification item, with step-by-step reasoning before a final
          "Answer: X" (generation)
dstated : every second main item with the timestamp difference stated in words after the log, which removes the
          datetime arithmetic (letters)
"""
import json
import os
import sys

RES = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(RES, "probes.jsonl"), encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]
cert = [r for r in rows if r["part"] == "cert"]
COT = ("Think step by step, weighing the timestamps against the clock note, the process note and the maintenance "
       "history, then give your final answer on the last line in the form \"Answer: X\".")


def events(r):
    txt = r["prompt"].split("Question:")[1]
    opts = {}
    for line in txt.splitlines():
        if len(line) > 3 and line[1] == ")":
            opts[r["roles"][line[0]]] = line[3:].strip()
    return opts


reason, dstated = [], []
for k, r in enumerate(main):
    if k % 9 == 0:
        reason.append(dict(r, id="ordr-reason-%s" % r["id"], base=r["id"], max_new=512,
                           prompt=r["prompt"].replace("Answer with a single letter.", COT)))
    if k % 2 == 0:
        ev = events(r)
        D = r["D"]
        first, second = (ev["A"], ev["B"]) if D > 0 else (ev["B"], ev["A"])
        note = ("\nNote: according to the device timestamps, \"%s\" was recorded %.3f s before \"%s\"." % (first, abs(D), second))
        p = r["prompt"].replace("\n\nQuestion:", note + "\n\nQuestion:", 1)
        dstated.append(dict(r, id="ordr-dstated-%s" % r["id"], base=r["id"], prompt=p))
for k, r in enumerate(cert):
    if k % 5 == 0:
        reason.append(dict(r, id="ordr-reason-%s" % r["id"], base=r["id"], max_new=512,
                           prompt=r["prompt"].replace("Answer with a single letter.", COT)))
for arm, data in (("reason", reason), ("dstated", dstated)):
    with open(os.path.join(RES, "rev_%s.jsonl" % arm), "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(arm, len(data))
print(dstated[0]["prompt"][-900:])
