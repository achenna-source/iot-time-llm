# -*- coding: utf-8 -*-
"""Comprehension gate for AGE: the same situations with re-reading unavailable (the device is offline), so the
model must choose between executing the command and doing nothing. Acting on the remembered state is correct
whenever the staleness probability is below 1/2, which holds for every probe here. Separate file and id space,
so that the main corpus keeps its ids.
"""
import json
import random
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from make_corpus import SCEN, units, fmt_q, iso  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

OUT = sys.argv[1]
rnd = random.Random(777)
rows = []
for si, s in enumerate(SCEN):
    site, dev, sheet, s_cond, s_other, instr, act, harm = s
    for cond in (True, False):
        for age_txt, age_s in (("0 minutes", 0), ("10 minutes", 600), ("3 hours", 10800)):
            for order in (0, 1):
                now = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rnd.randint(0, 300 * 86400))).replace(second=0)
                state = s_cond if cond else s_other
                mem = 'read_state("%s") returned {"device": "%s", "state": "%s", "as_of": "%s"}' % (dev, dev, state, iso(now - timedelta(seconds=age_s)))
                roles = ["act", "none"] if order == 0 else ["none", "act"]
                text_of = {"act": act, "none": "Do nothing"}
                opts = "\n".join("%s) %s" % ("AB"[i], text_of[r]) for i, r in enumerate(roles))
                body = ("You are the control agent of %s. You can issue commands. The device %s is offline at the moment, "
                        "so read_state cannot be called; you must decide from the information below.\n"
                        "Executing the wrong command costs %s (%s).\n"
                        "Device sheet: %s is a %s. Its logged history shows %s state changes per day on average.\n"
                        "Agent memory: %s\nCurrent time: %s\nOperator instruction: \"%s\"\n\n"
                        "What do you do now? Choose the option with the lowest expected cost.\n%s\nAnswer with a single letter."
                        ) % (site, dev, units(10), harm, dev, sheet, fmt_q(4), mem, iso(now), instr, opts)
                rows.append({"id": "agenr-%04d" % len(rows), "part": "gate_noread", "scen": si, "cond_true": cond,
                             "age_s": age_s, "roles": {"AB"[i]: r for i, r in enumerate(roles)},
                             "mem_role": "act" if cond else "none", "prompt": body})
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows))
print(rows[1]["prompt"])
