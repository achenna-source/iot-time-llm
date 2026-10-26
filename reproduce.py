# -*- coding: utf-8 -*-
"""Recompute every analysis, table and manuscript value of both papers from the released per-probe records.

No GPU is needed: the language-model outputs are in runs/, and everything downstream of them is recomputed here,
in a temporary copy of the results folders. The script then compares the regenerated tables and manuscript values
with the released ones and exits with a non-zero status at the first difference.

    python reproduce.py            # full check (about 10 to 20 minutes on a laptop CPU)
    python reproduce.py --quick    # skips the theory checks and the two Monte Carlo simulations
"""
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
PY = sys.executable
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def run(*args):
    print("  $", " ".join(os.path.relpath(a, ROOT) if os.path.isabs(a) and a.startswith(ROOT) else a for a in args), flush=True)
    r = subprocess.run([PY] + list(args), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("FAILED: %s" % " ".join(args))
    return r.stdout


tmp = tempfile.mkdtemp(prefix="iot_time_repro_")
T = {}
for p in ("age", "order"):
    T[p] = {k: os.path.join(tmp, p, k) for k in ("results", "tables", "manuscript")}
    shutil.copytree(os.path.join(ROOT, p, "results"), T[p]["results"])
    os.makedirs(T[p]["tables"])
    os.makedirs(T[p]["manuscript"])
RUNS, REAL = os.path.join(ROOT, "runs"), os.path.join(ROOT, "realdata")
AC, OC = os.path.join(ROOT, "age", "code"), os.path.join(ROOT, "order", "code")
A, O = T["age"], T["order"]

print("AGE")
if not QUICK:
    out = run(os.path.join(AC, "verify_theory.py"), os.path.join(A["results"], "theory_checks.json"))
    out += run(os.path.join(AC, "verify_theory_v2.py"))
    if "ALL PASS" not in out:
        sys.exit("FAILED: AGE theory checks")
run(os.path.join(AC, "analysis_age.py"), RUNS, A["results"])
run(os.path.join(AC, "analysis_age_v2.py"), RUNS, A["results"])
run(os.path.join(AC, "reference_bounds.py"), os.path.join(A["results"], "probes.jsonl"), os.path.join(A["results"], "reference_bounds.json"))
run(os.path.join(AC, "gate_misspec.py"), os.path.join(A["results"], "probes.jsonl"), os.path.join(A["results"], "gate_misspec.json"))
run(os.path.join(AC, "make_tables_ref.py"), os.path.join(A["results"], "reference_bounds.json"), os.path.join(A["tables"], "table_refbounds.md"))
run(os.path.join(AC, "make_tables_res.py"), A["results"], A["tables"], A["manuscript"], os.path.join(REAL, "realdata_age_v2.json"))

print("ORDER")
if not QUICK:
    out = run(os.path.join(OC, "verify_theory.py"), os.path.join(O["results"], "theory_checks.json"))
    out += run(os.path.join(OC, "verify_theory_v2.py"), os.path.join(O["results"], "theory_checks_v2.json"))
    out += run(os.path.join(OC, "verify_theory_v3.py"), os.path.join(O["results"], "theory_checks_v3.json"))
    if out.count("ALL PASS") < 3:
        sys.exit("FAILED: ORDER theory checks")
    run(os.path.join(OC, "misspec_stress.py"), os.path.join(O["results"], "misspec_stress.json"))
    run(os.path.join(OC, "realdata_order.py"), os.path.join(REAL, "realdata_order.json"), os.path.join(REAL, "hai_latency.json"),
        os.path.join(O["results"], "realdata_order_summary.json"))
run(os.path.join(OC, "analysis_order_v2.py"), RUNS, O["results"])
run(os.path.join(OC, "reference_bounds.py"), os.path.join(O["results"], "probes.jsonl"), os.path.join(O["results"], "reference_bounds.json"))
run(os.path.join(OC, "make_tables_ref.py"), os.path.join(O["results"], "reference_bounds.json"), os.path.join(O["tables"], "table_refbounds.md"))
run(os.path.join(OC, "make_tables_res.py"), O["results"], O["tables"], O["manuscript"])

print("COMPARISON")
bad = []
for p in ("age", "order"):
    ref_dir = os.path.join(ROOT, p, "tables")
    for f in sorted(os.listdir(ref_dir)):
        new = os.path.join(T[p]["tables"], f)
        if not os.path.exists(new) or not filecmp.cmp(new, os.path.join(ref_dir, f), shallow=False):
            bad.append("%s/tables/%s" % (p, f))
    exp = json.load(open(os.path.join(ROOT, "expected", "%s_fills.json" % p), encoding="utf-8"))
    got = json.load(open(os.path.join(T[p]["manuscript"], "fills_auto.json"), encoding="utf-8"))
    for k, v in exp.items():
        if got.get(k) != v:
            bad.append("%s manuscript value %s: released %r, recomputed %r" % (p, k, v, got.get(k)))
    n_tab, n_val = len(os.listdir(ref_dir)), len(exp)
    print("  %s: %d tables and %d manuscript values compared" % (p, n_tab, n_val))
shutil.rmtree(tmp, ignore_errors=True)
if bad:
    print("DIFFERENCES:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("OK: every table and every manuscript value is reproduced exactly.")
