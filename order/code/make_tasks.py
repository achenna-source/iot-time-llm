# -*- coding: utf-8 -*-
"""Generation tasks for the ORDER audit, built from probes.jsonl.

free    : the same incidents without an option list; the model names the initiating event in one sentence.
extract : the model only extracts timestamps and declared parameters as JSON; an external Bayes attributor
          then decides (end-to-end evaluation of the relocated decision).
"""
import json
import sys

src, out_free, out_ext = sys.argv[1:4]
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]


def body(prompt):
    return prompt.split("\n\nQuestion:")[0]


def events(r):
    roles = r["roles"]
    txt = r["prompt"].split("Question:")[1]
    opts = {}
    for line in txt.splitlines():
        if len(line) > 3 and line[1] == ")":
            opts[roles[line[0]]] = line[3:].strip()
    return opts["A"], opts["B"]


free, ext = [], []
for k, r in enumerate(main):
    ea, eb = events(r)
    if k % 8 == 0:
        free.append({"id": r["id"] + "-free", "base": r["id"], "max_new": 90,
                     "messages": [{"role": "user", "content": body(r["prompt"]) + (
                         "\n\nWhich event started this incident: \"%s\" or \"%s\"? Answer in one sentence that names "
                         "the event." % (ea, eb))}]})
    if k % 16 == 4:
        ext.append({"id": r["id"] + "-ext", "base": r["id"], "max_new": 160,
                    "messages": [{"role": "user", "content": body(r["prompt"]) + (
                        "\n\nDo not decide yet. Event A is \"%s\" and event B is \"%s\". Extract the facts a root-cause "
                        "analysis depends on and reply with only a JSON object with these keys: \"timestamp_A\" and "
                        "\"timestamp_B\" (the device timestamps of the two log lines, copied exactly), \"offset_sd_s\" "
                        "(clock offset standard deviation in seconds), \"latency_s\" (typical delay between the two events "
                        "in seconds), \"history_first\" (\"A\", \"B\" or \"none\": the event the maintenance history "
                        "says usually comes first), \"history_percent\" (number, 50 if none)." % (ea, eb))}]})
for fn, data in ((out_free, free), (out_ext, ext)):
    with open(fn, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("free", len(free), "extract", len(ext))
