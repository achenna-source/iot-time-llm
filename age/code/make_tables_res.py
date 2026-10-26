# -*- coding: utf-8 -*-
"""AGE result tables and manuscript fills (revision after the simulated review).

Reads analysis_age.json (letter-position and cue fits), analysis_age_v2.json (per-price regret, psi-robust set,
recalibration, tool calls with the structured parser, comprehension gate, revision arms, stratified extraction),
reference_bounds.json, gate_misspec.json and the real-trace re-analysis (temporal split).
python make_tables_res.py <results_dir> <tables_dir> <manuscript_dir> <realdata_age_v2.json>
"""
import json
import os
import re
import sys

import numpy as np

RES, TAB, MAN, REAL = sys.argv[1:5]
NL = chr(10)
A1 = json.load(open(os.path.join(RES, "analysis_age.json")))
A2 = json.load(open(os.path.join(RES, "analysis_age_v2.json")))
ORDER = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "granite3.3-8b", "olmo2-7b", "qwen2.5-14b"]
LAB = {"qwen2.5-7b": "Qwen2.5-7B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B",
       "granite3.3-8b": "Granite-3.3-8B", "olmo2-7b": "OLMo-2-7B", "qwen2.5-14b": "Qwen2.5-14B"}
models = [m for m in ORDER if m in A2]
core = [m for m in models if m != "qwen2.5-14b"]
PR = ("0.02", "0.1", "0.3")
PRICES = ("0.02", "0.1", "0.3", "0.6")


def sgn(x, d=3):
    return ("%+." + str(d) + "f") % x if x is not None and x == x else "n/a"


def m_(x):
    return re.sub(r"(^|[\s\[(|/])-(?=\d)", r"\1−", x)


def f2(v):
    return "n/a" if v is None or v != v else ("%.2f" % v if abs(v) < 10 else "%.0f" % v)


def pct(v):
    return "n/a" if v is None or v != v else "%.0f" % (100 * v)




def shorten(rows, title, lead=""):
    """one-line title; the definitions of the long caption move to the table note."""
    import re as _re
    cap = rows[0][len("Table: "):]
    m = _re.search(r"[.?] ", cap)
    rest = cap[m.end():] if m else ""
    details = (lead + " " + rest).strip()
    rows = list(rows)
    rows[0] = "Table: " + title
    ni = next((i for i, r in enumerate(rows) if r.startswith("Note:")), None)
    if details:
        if ni is None:
            rows.append("Note: " + details)
        else:
            rows[ni] = "Note: " + details + " " + rows[ni][len("Note: "):]
    return rows


CAPS = {'reread': ('Re-read decisions on the main grid', '8,640 probes per model; 4,320 for the Qwen2.5-14B scale check.'), 'cost': ('Cost of the re-read decisions at each read price', ''), 'arms': ('Re-read decisions when the model is helped', ''), 'tool': ('First turn with native tools', '216 situations, no option list.'), 'relocate': ('Comprehension and extraction', ''), 'gatemis': ('Freshness gate with misestimated fields', 'Main grid, no language model.'), 'fleet': ('Re-read rules on real home sensors', 'Normalised regret on 139 CASAS devices.')}


def write(name, rows):
    if name in CAPS:
        rows = shorten(rows, *CAPS[name])
    open(os.path.join(TAB, "table_%s.md" % name), "w", encoding="utf-8").write(NL.join(rows) + NL)


def lab(m):
    return LAB[m] + (" (6 scenarios)" if m == "qwen2.5-14b" else "")


def read_rate(m):
    return float(np.mean([A2[m]["per_price"][k]["read"] for k in PRICES]))


def J_all(m):
    if m in A1 and "main_argmax" in A1[m]:
        return A1[m]["main_argmax"]["J"]
    return float(np.mean([A2[m]["per_price"][k]["J"] for k in PRICES]))


def excess(pp, k):
    return 100 * (pp[k]["cost"] - pp[k]["cost_opt"])


# ---------------- Table 2: decisions
rows = ["Table: Re-read decisions on the main grid (8,640 probes per model; 4,320 for the Qwen2.5-14B scale check). Decisions are the most probable option; rates in percent. 95% intervals for $J$ from a bootstrap over the twelve scenarios.",
        "| Model | Letter mass | Re-read | Fresh reading | Acting optimal for any time share | At $r = 0.6$ | Condition holds / fails | $J$ [95% CI] |",
        "|---|---|---|---|---|---|---|---|"]
for m in models:
    a = A2[m]
    mass = A1.get(m, {}).get("format_mass_median")
    rows.append(m_("| %s | %s | %s | %s | %s | %s | %s / %s | %s [%s, %s] |" % (
        lab(m), "%.2f" % mass if mass is not None else "n/a", pct(read_rate(m)), pct(a.get("gate_fresh", {}).get("read")),
        pct(a["psi_robust"]["read_rate"]), pct(a["per_price"]["0.6"]["read"]),
        pct(a["by_state"]["condition_holds"]), pct(a["by_state"]["condition_fails"]),
        sgn(J_all(m)), sgn(a["cluster_ci"]["J"][0]), sgn(a["cluster_ci"]["J"][1]))))
rows.append("| Optimal policy | n/a | 47 | 0 | 0 | 0 | 47 / 47 | +0.314 |")
rows.append("Note: Letter mass: median probability that the first generated token is an option letter. Fresh reading: gate whose reading was taken in the current minute (288 probes). Acting optimal for any time share: the 3,744 probes on which acting on the reading is optimal whatever share of time the device spends in each state (Proposition 4). At $r = 0.6$ a read costs more than the expected loss of acting at any age in the corpus. Condition holds / fails: remembered state in which the instruction asks for the command, or for nothing.")
write("reread", rows)

# ---------------- Table 3: cost per price
rows = ["Table: Cost of the models' re-read decisions at each read price. $ρ$: normalised regret (0 = optimal, 1 = the better fixed rule at that price); in brackets, the excess expected cost over the optimum in percent of the wrong-command cost $L$. At $r = 0.6$ never re-reading is optimal and only the excess cost is defined. Best bias: the lowest regret reachable by adding one constant, chosen with hindsight, to the log-probability of the read option.",
        "| Model | $r = 0.02$ | $r = 0.1$ | $r = 0.3$ | $r = 0.6$ (excess) | Best bias, $r = 0.1$ | Best bias, $r = 0.3$ |",
        "|---|---|---|---|---|---|---|"]
for m in models:
    pp = A2[m]["per_price"]
    rc = A2[m]["recalibrated"].get("per_price", {})
    cell = lambda k: "%s (%.1f)" % (f2(pp[k]["rho"]), excess(pp, k))
    rows.append("| %s | %s | %s | %s | %.1f | %s | %s |" % (lab(m), cell("0.02"), cell("0.1"), cell("0.3"), excess(pp, "0.6"),
                                                          f2(rc.get("0.1", {}).get("rho")), f2(rc.get("0.3", {}).get("rho"))))
REF = json.load(open(os.path.join(RES, "reference_bounds.json")))
for key, name in (("always read", "Always re-read"), ("never read", "Never re-read"), ("best single TTL", "Best single time-to-live")):
    k = next(k for k in REF if k.startswith(key))
    pp = REF[k]["per_price"]
    rows.append("| %s | %s (%.1f) | %s (%.1f) | %s (%.1f) | %.1f | n/a | n/a |" % (
        name, f2(pp["0.02"]["rho"]), 100 * pp["0.02"]["excess"], f2(pp["0.1"]["rho"]), 100 * pp["0.1"]["excess"],
        f2(pp["0.3"]["rho"]), 100 * pp["0.3"]["excess"], 100 * pp["0.6"]["excess"]))
rows.append("Note: At $r = 0.02$ the optimum and always re-reading differ by only 0.3% of $L$, so small excess costs give large $ρ$; the excess cost is the quantity to compare across prices.")
write("cost", rows)

# ---------------- Table 4: helped arms
ARMS = (("baseline", "As in the corpus"), ("arm_share", "Time share stated"), ("arm_rule", "Decision rule stated"),
        ("arm_pgiven", "Staleness probability stated"), ("arm_reason", "Step-by-step reasoning"))
have_arms = [m for m in models if any(k in A2[m] for k in ("arm_rule", "arm_pgiven", "arm_share", "arm_reason"))]
rows = ["Table: Does helping the model change the decision? One probe per cell of the main grid (120 cells; three option rotations, except for reasoning, which uses one rotation and free generation within a budget of 512 new tokens, parsed for a final answer or, when the budget ran out, forced by appending \"Answer:\" to the model's own reasoning). Re-read rate at $r = 0.6$, where re-reading is never optimal, and normalised regret at $r = 0.1$ and $r = 0.3$.",
        "| Model | Arm | Re-read, $r = 0.6$ | $ρ$, $r = 0.1$ | $ρ$, $r = 0.3$ | $J$ |", "|---|---|---|---|---|---|"]
for m in have_arms:
    a = A2[m]
    first = True
    base = next((a[k]["baseline_same_cells"] for k in ("arm_rule", "arm_pgiven", "arm_share") if k in a and "baseline_same_cells" in a[k]), None)
    for k, name in ARMS:
        v = base if k == "baseline" else a.get(k)
        if v is None:
            continue
        pp = v.get("per_price", {})
        r06 = pp.get("0.6", {}).get("read") if pp else v.get("read_r06")
        nm = name + (" (%.0f%% answered within budget)" % (100 * v.get("self_answered", v["parsed"])) if k == "arm_reason" else "")
        rows.append(m_("| %s | %s | %s | %s | %s | %s |" % (lab(m) if first else "", nm, pct(r06), f2(pp.get("0.1", {}).get("rho")),
                                                        f2(pp.get("0.3", {}).get("rho")), sgn(v.get("J")))))
        first = False
rows.append("| Optimal policy | | 0 | 0.00 | 0.00 | +0.314 |")
write("arms", rows)

# ---------------- Table 5: native tool calls
rows = ["Table: Native tool calls on 216 situations (no option list). Responses are parsed with each model's structured tool-call format, with a fallback for calls written as text. Read and command: both calls emitted in the same turn, so that the command does not wait for the read. Agreement and Cohen's $κ$ compare the first turn, collapsed to read, act on the reading or other, with the letter decision in the same situation.",
        "| Model | Structured | Read only | Command only | Read and command | No call | Agreement | $κ$ |", "|---|---|---|---|---|---|---|---|"]
for m in models:
    t = A2[m].get("toolcall_v2")
    if not t:
        continue
    c = t["classes"]
    rows.append(m_("| %s | %s | %s | %s | %s | %s | %s | %s |" % (lab(m), pct(t["structured_share"]), pct(c["read"]), pct(c["act"]), pct(c["both"]),
                                                              pct(c["none"]), pct(t["agree"]), ("%.2f" % t["kappa"]) if t["kappa"] == t["kappa"] else "n/a")))
rows.append("Note: Rates in percent. $κ$ is undefined when one of the two classifications is constant.")
write("tool", rows)

# ---------------- Table 6: comprehension and extraction
rows = ["Table: What the models can do when the decision is taken away from them. Comprehension: accuracy on the gate where re-reading is unavailable (144 probes, chance 50%). Extraction: share of 120 outputs, stratified by price, remembered state and volatility, in which each field is exact. Gate $ρ$: pooled normalised regret of the freshness gate fed with the extracted fields; flips: share of its decisions that differ from the gate fed with the declared fields.",
        "| Model | Comprehension | Parsed | Age | Change rate | Read cost | Wrong-command cost | Gate $ρ$ | Flips |",
        "|---|---|---|---|---|---|---|---|---|"]
for m in models:
    a = A2[m]
    nr = a.get("noread")
    v = a.get("vext")
    fa = v["field_acc"] if v else {}
    rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        lab(m), (pct(nr["argmax_acc"]) + "%") if nr else "n/a", pct(fa.get("parsed")), pct(fa.get("age")), pct(fa.get("changes_per_day")),
        pct(fa.get("read_cost")), pct(fa.get("wrong_command_cost")), f2(v["gate_rho"]) if v else "pending",
        pct(v["decision_flip_vs_declared"]) if v else "pending"))
rows.append("Note: Field accuracies in percent.")
write("relocate", rows)

# ---------------- Table 7: gate under misestimated fields
G = json.load(open(os.path.join(RES, "gate_misspec.json")))
rows = ["Table: The freshness gate when its fields are wrong (main grid, no language model). Normalised regret against the optimum of the true process, with the excess cost in percent of $L$ in brackets; n/a where a fixed rule is optimal at that price.",
        "| Departure from the declared fields | $r = 0.02$ | $r = 0.1$ | $r = 0.3$ | $r = 0.6$ |", "|---|---|---|---|---|"]
for name, v in G.items():
    rows.append("| %s | %s |" % (name, " | ".join("%s (%.1f)" % (f2(v[k]["gate_rho"]), 100 * (v[k]["gate_cost"] - v[k]["opt_cost"])) for k in PRICES)))
write("gatemis", rows)

# ---------------- Table 8: real fleets (temporal split, requests weighted by time share)
real = json.load(open(REAL))
rows = ["Table: Normalised regret of re-read rules on real device fleets (CASAS, %d devices). Every rule and every time-to-live is fitted on the first half of each device record and evaluated on the second half; a request finds the device in each state in proportion to the time it spends there, at ages from one minute to three days. 0 is the empirical optimum of the test half and 1 the better of the two fixed rules." % real["n_devices"],
        "| Fleet | Price $r$ | Single TTL | TTL per class | TTL per device state | Exit-rate rule | Two-parameter gate | Always re-read | Never re-read |",
        "|---|---|---|---|---|---|---|---|---|"]
names = {"door": "Door contacts", "motion": "Motion sensors", "all": "All devices"}
ren = []
for k, w in (("door", "share"), ("motion", "share"), ("all", "share"), ("all", "uniform")):
    f = real["by"][k]
    for r in PR:
        v = f["%s_r%s" % (w, r)]["rho"]
        lb = (names[k] + " (%d)" % f["n_devices"] + (", uniform requests" if w == "uniform" else "")) if r == "0.02" else ""
        rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            lb, r, f2(v["ttl_single"]), f2(v["ttl_class"]), f2(v["ttl_state"]), f2(v["exit_rate_rule"]),
            f2(v["two_param_gate"]), f2(v["always_read"]), f2(v["never_read"])))
        ren.append(abs(v["renewal_rule"] - v["exit_rate_rule"]) / max(v["exit_rate_rule"], 1e-9))
rows.append("Note: The two-parameter gate applies Theorem 1 with the exit rate and time share fitted on the first half. The renewal rule of Proposition 4(ii), re-read when the age exceeds $r$ mean sojourns, is within %.0f%% of the exit-rate rule in every cell. The last block weights every device, state and age equally. Office occupancy (UCI) is omitted because its recordings are too short to split." % (100 * max(ren)))
write("fleet", rows)

# ---------------- fills
F = {"N_MODELS": {4: "four", 5: "five", 6: "six"}.get(len(core), str(len(core)))}


def join(n):
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]


F["J_MAX"] = m_(sgn(max(J_all(m) for m in core)))
F["J_MAXABS"] = "%.3f" % max(abs(J_all(m)) for m in core)
blanket = [m for m in core if read_rate(m) >= 0.85]
others = [m for m in core if m not in blanket]
F["N_BLANKET"] = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(len(blanket), str(len(blanket)))
F["BLANKET_LIST"] = join([LAB[m] for m in blanket]) if blanket else "none"
F["BLANKET_MIN"] = pct(min(read_rate(m) for m in blanket)) if blanket else "n/a"
F["OTHER_LIST"] = join([LAB[m] for m in others]) if others else "none"
F["N_OTHER"] = {0: "No", 1: "One", 2: "Two"}.get(len(others), str(len(others)))
F["ROBUST_BLANKET_MIN"] = pct(min(A2[m]["psi_robust"]["read_rate"] for m in blanket)) if blanket else "n/a"
F["FRESH_BLANKET_MIN"] = pct(min(A2[m]["gate_fresh"]["read"] for m in blanket)) if blanket else "n/a"
F["R06_BLANKET_MIN"] = pct(min(A2[m]["per_price"]["0.6"]["read"] for m in blanket)) if blanket else "n/a"
for k in PRICES:
    ex = [excess(A2[m]["per_price"], k) for m in core]
    F["EXCESS_%s_MIN" % k.replace(".", "")], F["EXCESS_%s_MAX" % k.replace(".", "")] = "%.1f" % min(ex), "%.1f" % max(ex)
for k in PR:
    rh = [A2[m]["per_price"][k]["rho"] for m in core]
    F["RHO_%s_MIN" % k.replace(".", "")], F["RHO_%s_MAX" % k.replace(".", "")] = f2(min(rh)), f2(max(rh))
rec = [A2[m]["recalibrated"]["per_price"][k]["rho"] for m in core for k in ("0.1", "0.3")
       if A2[m]["recalibrated"].get("per_price", {}).get(k, {}).get("rho") is not None]
F["RECAL_MIN"] = f2(min(rec)) if rec else "n/a"
spread = {m: max(A1[m]["P_read_by_letter"].values()) - min(A1[m]["P_read_by_letter"].values()) for m in core if m in A1}
mx = max(spread, key=spread.get)
F["LETTER_MODEL"] = LAB[mx]
F["LETTER_HI"] = "%.0f" % (100 * max(A1[mx]["P_read_by_letter"].values()))
F["LETTER_LO"] = "%.0f" % (100 * min(A1[mx]["P_read_by_letter"].values()))
syms = [abs(A1[m]["sym"].get(k, 0) or 0) for m in core if m in A1 for k in ("b_age", "b_q", "b_r") if "error" not in A1[m]["sym"]]
F["SYM_MAXABS"] = "%.3f" % max(syms) if syms else "n/a"
wrong = [m for m in core if m in A1 and "error" not in A1[m]["sym"] and A1[m]["sym"].get("b_r", 0) > max(0.05, 2 * A1[m]["sym"].get("se_r", 1))]
F["WRONG_PRICE"] = join([LAB[m] for m in wrong]) if wrong else "none"
vj = [abs(A1[m]["variants"][v]["J"]) for m in core if m in A1 for v in ("rel", "qual", "none") if v in A1[m]["variants"]]
F["VAR_J_MAXABS"] = "%.3f" % max(vj)
nrs = [A2[m]["noread"]["argmax_acc"] for m in core if "noread" in A2[m]]
F["NOREAD_MIN"], F["NOREAD_MAX"] = pct(min(nrs)), pct(max(nrs))
gst = {m: A2[m]["gate_stale"]["act_on_memory"] for m in core if "gate_stale" in A2[m]}
F["STALE_ACT_MAX"], F["STALE_ACT_MODEL"] = pct(max(gst.values())), LAB[max(gst, key=gst.get)]
tc = {m: A2[m]["toolcall_v2"] for m in core if "toolcall_v2" in A2[m]}
F["TOOL_STRUCT_ZERO"] = join([LAB[m] for m in tc if tc[m]["structured_share"] < 0.05]) or "none"
F["TOOL_BOTH_MODEL"] = LAB[max(tc, key=lambda m: tc[m]["both_rate"])]
F["TOOL_BOTH_MAX"] = pct(max(t["both_rate"] for t in tc.values()))
kap = [t["kappa"] for t in tc.values() if t["kappa"] == t["kappa"]]
F["KAPPA_MIN"], F["KAPPA_MAX"] = (m_("%.2f" % min(kap)), m_("%.2f" % max(kap))) if kap else ("n/a", "n/a")
vx = {m: A2[m]["vext"] for m in core if "vext" in A2[m]}
if vx:
    F["VEXT_FIELD_MIN"] = pct(min(min(x for x in v["field_acc"].values() if x is not None) for v in vx.values()))
    F["VEXT_GATE_RHO_MAX"] = f2(max(v["gate_rho"] for v in vx.values()))
NUMW = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def rng_(vals, f=f2):
    vals = [v for v in vals if v is not None and v == v]
    return (m_(f(min(vals))), m_(f(max(vals)))) if vals else ("n/a", "n/a")


arm = lambda m, k: A2[m].get(k)
st = [m for m in core if all(arm(m, k) for k in ("arm_rule", "arm_pgiven", "arm_share"))]
F["ARM_N"] = NUMW[len(st)]
bl_st = [m for m in st if m in blanket]
F["ARM_R06_BLANKET_MIN"] = pct(min(arm(m, k)["per_price"]["0.6"]["read"] for m in bl_st for k in ("arm_rule", "arm_pgiven", "arm_share"))) if bl_st else "n/a"
F["ARM_J_MAX"] = m_(sgn(max(arm(m, k)["J"] for m in st for k in ("arm_rule", "arm_pgiven", "arm_share")))) if st else "n/a"
F["ARM_RHO01_MIN"], F["ARM_RHO01_MAX"] = rng_([arm(m, k)["per_price"]["0.1"]["rho"] for m in st for k in ("arm_rule", "arm_pgiven", "arm_share")])
F["ARM_RHO03_MIN"], F["ARM_RHO03_MAX"] = rng_([arm(m, k)["per_price"]["0.3"]["rho"] for m in st for k in ("arm_rule", "arm_pgiven", "arm_share")])
rs = [m for m in core if arm(m, "arm_reason")]
F["RS_N"] = NUMW[len(rs)]
F["RS_R06_MIN"], F["RS_R06_MAX"] = rng_([arm(m, "arm_reason")["per_price"]["0.6"]["read"] for m in rs], pct)
F["RS_RHO01_MIN"], F["RS_RHO01_MAX"] = rng_([arm(m, "arm_reason")["per_price"]["0.1"]["rho"] for m in rs])
F["RS_RHO03_MIN"], F["RS_RHO03_MAX"] = rng_([arm(m, "arm_reason")["per_price"]["0.3"]["rho"] for m in rs])
F["RS_EX01_MIN"], F["RS_EX01_MAX"] = rng_([100 * arm(m, "arm_reason")["per_price"]["0.1"]["excess"] for m in rs], lambda x: "%.0f" % x)
F["RS_J_MAX"] = m_(sgn(max(arm(m, "arm_reason")["J"] for m in rs))) if rs else "n/a"
F["RS_SELF_MIN"], F["RS_SELF_MAX"] = rng_([arm(m, "arm_reason").get("self_answered") for m in rs], pct)
F["RS_FRESH_MIN"], F["RS_FRESH_MAX"] = rng_([arm(m, "arm_reason").get("fresh_read") for m in rs], pct)
vx = {m: A2[m]["vext"] for m in core if "vext" in A2[m]}
exact = [LAB[m] for m, v in vx.items() if min(x for x in v["field_acc"].values() if x is not None) >= 0.995]
F["VEXT_N"] = NUMW[len(vx)]
F["VEXT_EXACT_N"] = NUMW[len(exact)]
F["VEXT_EXACT_LIST"] = join(exact) if exact else "none"
F["VEXT_RHO_MAX"] = f2(max(v["gate_rho"] for v in vx.values())) if vx else "n/a"
F["VEXT_FLIP_MAX"] = pct(max(v["decision_flip_vs_declared"] for v in vx.values())) if vx else "n/a"
others_v = [(m, v) for m, v in vx.items() if LAB[m] not in exact]
FN = {"age": "age", "changes_per_day": "change rate", "read_cost": "read cost", "wrong_command_cost": "wrong-command cost"}


def vext_other(m, v):
    fa = v["field_acc"]
    bad = [(FN[k], fa[k]) for k in FN if fa.get(k) is not None and fa[k] < 0.995]
    txt = "%s returns valid JSON in %s%% of the outputs and copies %s" % (LAB[m], pct(fa.get("parsed")), join(["the %s correctly in %s%%" % (n, pct(x)) for n, x in bad]) if bad else "every field correctly")
    return txt + "; the gate fed with its fields has a regret of %s and changes %s%% of the decisions." % (f2(v["gate_rho"]), pct(v["decision_flip_vs_declared"]))


F["VEXT_OTHER"] = " ".join(vext_other(m, v) for m, v in others_v)
q = A2.get("qwen2.5-14b")
if q:
    good = J_all("qwen2.5-14b") < 0.05 and all((q["per_price"][k]["rho"] or 0) >= 0.9 for k in ("0.1", "0.3"))
    F["Q14_PHRASE"] = "much like the smaller models" if good else ("somewhat better, but no better than a fixed rule at some prices" if any((q["per_price"][k]["rho"] or 0) >= 0.9 for k in ("0.1", "0.3")) else "better than any of the smaller models")
    F["Q14_READ"] = pct(read_rate("qwen2.5-14b"))
    F["Q14_J"] = m_(sgn(J_all("qwen2.5-14b")))
    F["Q14_RHO01"] = f2(q["per_price"]["0.1"]["rho"])
    F["Q14_RHO03"] = f2(q["per_price"]["0.3"]["rho"])
    F["Q14_R06"] = pct(q["per_price"]["0.6"]["read"])
    F["Q14_FRESH"] = pct(q.get("gate_fresh", {}).get("read"))
    F["Q14_NOREAD"] = pct(q.get("noread", {}).get("argmax_acc"))
    qa = [q[k] for k in ("arm_share", "arm_rule", "arm_pgiven", "arm_reason") if k in q]
    F["Q14_ARM01_MIN"], F["Q14_ARM01_MAX"] = rng_([v["per_price"]["0.1"]["rho"] for v in qa])
    F["Q14_ARM03_MIN"], F["Q14_ARM03_MAX"] = rng_([v["per_price"]["0.3"]["rho"] for v in qa])
    F["Q14_ARMR06_MIN"] = pct(min(v["per_price"]["0.6"]["read"] for v in qa)) if qa else "n/a"
if "Q14_PHRASE" in F:
    F["Q14_SENT"] = "The scale check behaves %s. On six scenarios, Qwen2.5-14B re-reads on %s%% of the probes and on %s%% at $r = 0.6$, with $J = %s$ and a regret of %s at $r = 0.1$ and %s at $r = 0.3$ (Tables 2 to 4), although it passes the comprehension gate on %s%% of the probes and copies every field exactly (Table 6). With the time share, the rule or the staleness probability stated, or with reasoning, it is the only model whose regret at $r = 0.1$ falls below that of the better fixed rule, to between %s and %s; at $r = 0.3$ it stays between %s and %s, and the model still re-reads on at least %s%% of the probes at $r = 0.6$." % (F["Q14_PHRASE"], F["Q14_READ"], F["Q14_R06"], F["Q14_J"], F["Q14_RHO01"], F["Q14_RHO03"], F["Q14_NOREAD"], F["Q14_ARM01_MIN"], F["Q14_ARM01_MAX"], F["Q14_ARM03_MIN"], F["Q14_ARM03_MAX"], F["Q14_ARMR06_MIN"])
else:
    F["Q14_PHRASE"] = '`[to be completed: Qwen2.5-14B scale check, run still in progress]`'
    F["Q14_SENT"] = "`[To be completed: Qwen2.5-14B scale check, run still in progress.]`"
json.dump(F, open(os.path.join(MAN, "fills_auto.json"), "w"), indent=1)
print(json.dumps(F, indent=1))
