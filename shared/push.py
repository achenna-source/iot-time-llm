# -*- coding: utf-8 -*-
"""Stage the probe dataset and push one private Kaggle kernel per model.

python push.py dataset                      create or version the private dataset iot-time-probes
python push.py run <slug> <model_key> <plan> [gpu]   push probe_runner.py with an injected configuration
python push.py status <slug> | pull <slug> <dir>
Plans: pilot, age, order (see PLANS).
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
USER = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))["username"]
DS = "%s/iot-time-probes" % USER
BUILD = ROOT / "kaggle_build"

MODELS = {
    "qwen25": ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
    "qwen3": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "llama31": ("unsloth/Llama-3.1-8B-Instruct", "llama3.1-8b"),
    "granite": ("ibm-granite/granite-3.3-8b-instruct", "granite3.3-8b"),
    "olmo2": ("allenai/OLMo-2-1124-7B-Instruct", "olmo2-7b"),
    "qwen14": ("Qwen/Qwen2.5-14B-Instruct", "qwen2.5-14b"),
}
I = "/kaggle/input/**/"
PLANS = {
    "pilot": [[I + "age_probes.jsonl", "letters", ["main"], 300, "age_main"],
              [I + "order_probes.jsonl", "letters", ["main"], 240, "order_main"],
              [I + "age_toolcall.jsonl", "generate", None, 12, "age_tool"],
              [I + "age_extract.jsonl", "generate", None, 8, "age_ext"],
              [I + "order_free.jsonl", "generate", None, 8, "order_free"],
              [I + "order_extract.jsonl", "generate", None, 8, "order_ext"]],
    "age": [[I + "age_probes.jsonl", "letters", None, None, "age_all"],
            [I + "age_toolcall.jsonl", "generate", None, None, "age_tool"],
            [I + "age_extract.jsonl", "generate", None, None, "age_ext"]],
    "full": [[I + "age_probes.jsonl", "letters", None, None, "age_all"],
             [I + "order_probes.jsonl", "letters", None, None, "order_all"],
             [I + "order_probes.jsonl", "letters", None, None, "order_px", "Answer:"],
             [I + "age_noread.jsonl", "letters", None, None, "age_noread"],
             [I + "age_toolcall.jsonl", "generate", None, None, "age_tool"],
             [I + "order_free.jsonl", "generate", None, None, "order_free"],
             [I + "age_extract.jsonl", "generate", None, None, "age_ext"],
             [I + "order_extract.jsonl", "generate", None, None, "order_ext"]],
    "prefix": [[I + "order_probes.jsonl", "letters", None, None, "order_px", "Answer:"],
               [I + "age_noread.jsonl", "letters", None, None, "age_noread"]],
    "noread": [[I + "age_noread.jsonl", "letters", None, None, "age_noread"]],
    "rev": [[I + "age_rev_reason.jsonl", "generate_batched", None, None, "age_reason", "8"],
            [I + "order_rev_reason.jsonl", "generate_batched", None, None, "order_reason", "8"],
            [I + "age_rev_rule.jsonl", "letters", None, None, "age_rule"],
            [I + "age_rev_pgiven.jsonl", "letters", None, None, "age_pgiven"],
            [I + "age_rev_share.jsonl", "letters", None, None, "age_share"],
            [I + "order_rev_dstated.jsonl", "letters", None, None, "order_dstated", "Answer:"],
            [I + "order_rev_free.jsonl", "generate_batched", None, None, "order_vfree", "8"],
            [I + "order_rev_ext.jsonl", "generate_batched", None, None, "order_vext", "8"],
            [I + "age_rev_ext.jsonl", "generate_batched", None, None, "age_vext", "8"]],
    "force": [[I + "force_age__{short}.jsonl", "letters", None, None, "age_force"],
              [I + "force_order__{short}.jsonl", "letters", None, None, "order_force"]],
    "big": [[I + "age_rev_half.jsonl", "letters", None, None, "age_all"],
            [I + "age_noread.jsonl", "letters", None, None, "age_noread"],
            [I + "order_probes.jsonl", "letters", None, None, "order_px", "Answer:"],
            [I + "age_rev_reason.jsonl", "generate_batched", None, None, "age_reason", "6"],
            [I + "order_rev_reason.jsonl", "generate_batched", None, None, "order_reason", "6"],
            [I + "age_rev_rule.jsonl", "letters", None, None, "age_rule"],
            [I + "age_rev_pgiven.jsonl", "letters", None, None, "age_pgiven"],
            [I + "age_rev_share.jsonl", "letters", None, None, "age_share"],
            [I + "order_rev_dstated.jsonl", "letters", None, None, "order_dstated", "Answer:"],
            [I + "order_rev_free.jsonl", "generate_batched", None, None, "order_vfree", "6"],
            [I + "order_rev_ext.jsonl", "generate_batched", None, None, "order_vext", "6"],
            [I + "age_rev_ext.jsonl", "generate_batched", None, None, "age_vext", "6"],
            [I + "age_toolcall.jsonl", "generate", None, None, "age_tool"]],
    "order": [[I + "order_probes.jsonl", "letters", None, None, "order_all"],
              [I + "order_free.jsonl", "generate", None, None, "order_free"],
              [I + "order_extract.jsonl", "generate", None, None, "order_ext"]],
}


def sh(cmd):
    print("  $", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    print("   " + out.strip().replace("\n", "\n   "))
    return out


def dataset():
    d = BUILD / "dataset"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for paper, files in (("age", ("probes", "toolcall", "extract", "noread", "rev_reason", "rev_rule", "rev_pgiven", "rev_share", "rev_half", "rev_ext")),
                         ("order", ("probes", "free", "extract", "rev_reason", "rev_dstated", "rev_free", "rev_ext"))):
        for f in files:
            shutil.copy(ROOT / paper / "results" / ("%s.jsonl" % f), d / ("%s_%s.jsonl" % (paper, f)))
    for f in sorted((HERE / "force").glob("*.jsonl")) if (HERE / "force").exists() else []:
        shutil.copy(f, d / f.name)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "iot-time-probes", "id": DS, "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    out = sh(["kaggle", "datasets", "status", DS])
    if "ready" in out.lower() or "complete" in out.lower():
        sh(["kaggle", "datasets", "version", "-p", str(d), "-m", "update", "-r", "zip"])
    else:
        sh(["kaggle", "datasets", "create", "-p", str(d), "-r", "zip"])


def run(slug, key, plan, gpu=True):
    mid, short = MODELS[key]
    d = BUILD / slug
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = (HERE / "probe_runner.py").read_text(encoding="utf-8-sig")
    plan_inputs = [[x.replace("{short}", short) if isinstance(x, str) else x for x in spec] for spec in PLANS[plan]]
    env = {"MODEL_ID": mid, "SHORT": short, "LOAD_4BIT": True, "INPUTS": plan_inputs}
    for k in env:
        src = "\n".join(("# [overridden] " + ln) if re.match(r"^%s\s*=" % k, ln) else ln for ln in src.splitlines())
    header = "".join("%s = %r\n" % (k, v) for k, v in env.items())
    (d / "probe_runner.py").write_text("# configuration injected by push.py\n" + header + "\n" + src, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": "%s/%s" % (USER, slug), "title": slug, "code_file": "probe_runner.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu,
        "machine_shape": "NvidiaTeslaT4" if gpu else "", "enable_internet": True,
        "dataset_sources": [DS], "kernel_sources": [], "competition_sources": []}, indent=2), encoding="utf-8")
    sh(["kaggle", "kernels", "push", "-p", str(d)])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "dataset":
        dataset()
    elif a[0] == "run":
        run(a[1], a[2], a[3], gpu=(len(a) < 5 or a[4] != "cpu"))
    elif a[0] == "status":
        sh(["kaggle", "kernels", "status", "%s/%s" % (USER, a[1])])
    elif a[0] == "pull":
        Path(a[2]).mkdir(parents=True, exist_ok=True)
        sh(["kaggle", "kernels", "output", "%s/%s" % (USER, a[1]), "-p", a[2]])
