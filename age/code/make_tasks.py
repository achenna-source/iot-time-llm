# -*- coding: utf-8 -*-
"""Generation tasks for the AGE audit, built from probes.jsonl.

toolcall : the same situations with native tools (read_state, execute_command) and no option list;
           the first tool the model calls is the behavioural ground truth for the letter probe.
extract  : the model only extracts the decision-relevant facts as JSON; an external freshness gate then
           decides (end-to-end evaluation of the relocated decision).
"""
import json
import re
import sys

src, out_tool, out_ext = sys.argv[1:4]
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
main = [r for r in rows if r["part"] == "main" and r["rot"] == 0]

TOOLS = [
    {"type": "function", "function": {
        "name": "read_state", "description": "Return the current state of a device. Costs energy (see cost model).",
        "parameters": {"type": "object", "properties": {"device": {"type": "string"}}, "required": ["device"]}}},
    {"type": "function", "function": {
        "name": "execute_command", "description": "Execute a command on the site (actuation, dispatch, notification).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def split(prompt):
    head, rest = prompt.split("Device sheet:", 1)
    situ = "Device sheet:" + rest.split("\n\nWhat do you do next?")[0]
    return head.strip(), situ.strip()


tool, ext = [], []
for r in main:
    head, situ = split(r["prompt"])
    if r["scen"] < 6 and r["q_day"] in (0.25, 4, 64) and r["age_s"] in (600, 10800, 172800) and r["r"] in (0.1, 0.6):
        sys_txt = head.replace("You can call read_state(device), which returns the current state of a device, and you "
                               "can issue commands.", "Use the tools to read device states or to execute commands; if "
                               "no command is needed, reply without calling a tool.")
        tool.append({"id": r["id"] + "-tool", "base": r["id"], "max_new": 160,
                     "messages": [{"role": "system", "content": sys_txt},
                                  {"role": "user", "content": situ + "\n\nAct now, choosing the action with the lowest expected cost."}],
                     "tools": TOOLS})
    if r["scen"] < 6 and r["q_day"] in (0.25, 4, 64) and r["r"] == 0.1 and r["cond_true"]:
        ext.append({"id": r["id"] + "-ext", "base": r["id"], "max_new": 140,
                    "messages": [{"role": "user", "content": head + "\n" + situ + (
                        "\n\nDo not decide yet. Extract the facts the decision depends on and reply with only a JSON object "
                        "with these keys: \"as_of\" (time of the remembered reading, ISO 8601), \"current_time\" (ISO 8601), "
                        "\"changes_per_day\" (number), \"read_cost\" (number), \"wrong_command_cost\" (number), "
                        "\"remembered_state\" (string).")}]})
for fn, data in ((out_tool, tool), (out_ext, ext)):
    with open(fn, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("toolcall", len(tool), "extract", len(ext))
print(json.dumps(tool[0]["messages"], indent=1)[:1500])
