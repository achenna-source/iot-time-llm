# -*- coding: utf-8 -*-
"""AGE analysis, revision 2 (after the simulated review). Adds to analysis_age.json:

tool calls   parsed with a structured-call parser (Qwen <tool_call>, Llama python_tag JSON, any {"name": ...} JSON)
             and an explicit-syntax fallback (read_state(...) / execute_command(...)); classes read / act / none /
             both (read and command emitted in one turn: the command does not wait for the read); Cohen's kappa
per price    argmax J, rho and absolute cost per read price, and pooled without r = 0.6
psi-robust   probes on which acting is optimal for ANY time share (1 - exp(-qD) <= r, Proposition 4)
recalibrate  best expected cost reachable by adding one constant to the read-option log-probability
cluster CI   bootstrap over scenarios (the 12 scenarios are the exchangeable units)
by state     argmax re-read rate by remembered state (condition holds / does not hold)
stale gate   reading declared outdated: acting on it is the error (p = 1); report P(act on memory)
rev arms     reasoning (parsed "Answer: X"), rule stated, staleness stated, time share stated, stratified extraction
python analysis_age_v2.py <runs_dir> <results_dir>
"""
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np

from metrics import regret_summary, youden

RUNS, RES = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(1)
P = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "probes.jsonl"), encoding="utf-8"))}
TOOL = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "toolcall.jsonl"), encoding="utf-8"))}
REV = {}
for arm in ("reason", "rule", "pgiven", "share", "half", "ext"):
    fn = os.path.join(RES, "rev_%s.jsonl" % arm)
    if os.path.exists(fn):
        for r in map(json.loads, open(fn, encoding="utf-8")):
            REV[r["id"]] = r
MODELS = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "granite3.3-8b", "olmo2-7b", "qwen2.5-14b"]


def load(prefix, model):
    seen, out = set(), []
    for fn in glob.glob(os.path.join(RUNS, "**", "%s__%s.jsonl" % (prefix, model)), recursive=True):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
    return out


def roles_of(rec, probe):
    d = {"read": 0.0, "mem": 0.0, "opp": 0.0}
    for L, v in rec["probs"].items():
        role = probe["roles"][L]
        d["read" if role == "read" else ("mem" if role == probe["mem_role"] else "opp")] += v
    return d


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    cats = set(a) | set(b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


CALL = re.compile(r'"name"\s*:\s*"(read_state|execute_command)"')
TEXTCALL = re.compile(r'\b(read_state|execute_command)\s*\(')


def parse_tool(text):
    calls = CALL.findall(text)
    how = "structured"
    if not calls:
        calls = TEXTCALL.findall(text)
        how = "textual" if calls else "none"
    has_r, has_e = "read_state" in calls, "execute_command" in calls
    if has_r and has_e:
        cls = "both"
    elif has_r:
        cls = "read"
    elif has_e:
        cls = "act"
    else:
        cls = "none"
    return cls, how


ANS = re.compile(r"answer\s*[:\-]?\s*\**\s*\(?([A-C])\b", re.I)


def parse_answer(text):
    m = ANS.findall(text)
    return m[-1].upper() if m else None


def per_price(dr, dm, do, p, rr):
    out = {}
    for v in sorted(set(rr)):
        m = rr == v
        s = regret_summary(dr[m], dm[m], do[m], p[m], rr[m])
        out[str(v)] = {"read": float(dr[m].mean()), "J": float(s["J"]), "cost": float(s["cost"]),
                       "cost_opt": float(s["cost_opt"]), "cost_fix": float(s["cost_best_fixed"]),
                       "rho": float(s["rho"]) if s["cost_best_fixed"] - s["cost_opt"] > 1e-6 else None}
    m = rr < 0.5
    s = regret_summary(dr[m], dm[m], do[m], p[m], rr[m])
    out["without_0.6"] = {"read": float(dr[m].mean()), "J": float(s["J"]), "rho": float(s["rho"]), "cost": float(s["cost"])}
    return out


def analyse(model):
    recs = [r for r in load("age_all", model) if r["id"] in P]
    if not recs:
        return None
    R = {}
    main = [r for r in recs if P[r["id"]]["part"] == "main"]
    pr_ = [P[r["id"]] for r in main]
    rp = [roles_of(r, P[r["id"]]) for r in main]
    am = np.array([max(x, key=x.get) for x in rp])
    dr, dm, do = (am == "read").astype(float), (am == "mem").astype(float), (am == "opp").astype(float)
    pread = np.array([x["read"] for x in rp])
    p = np.array([x["p_change"] for x in pr_])
    rr = np.array([x["r"] for x in pr_])
    q = np.array([x["q_day"] for x in pr_]) / 86400
    age = np.array([x["age_s"] for x in pr_], float)
    scen = np.array([x["scen"] for x in pr_])
    cond = np.array([x["cond_true"] for x in pr_])
    R["n_main"] = len(main)
    R["per_price"] = per_price(dr, dm, do, p, rr)
    # psi-robust acting set (Proposition 4): acting optimal whatever the time share
    rob = (1 - np.exp(-q * age)) <= rr
    R["psi_robust"] = {"n": int(rob.sum()), "read_rate": float(dr[rob].mean()), "prob_read": float(pread[rob].mean())}
    # recalibration: one constant added to the read log-probability
    lpr = np.log(np.maximum(pread, 1e-12))
    lalt = np.log(np.maximum(np.array([max(x["mem"], x["opp"]) for x in rp]), 1e-12))
    altmem = np.array([x["mem"] >= x["opp"] for x in rp])
    best = None
    for b in np.linspace(-30, 30, 241):
        rd = (lpr + b > lalt).astype(float)
        mem = (1 - rd) * altmem
        opp = (1 - rd) * (~altmem)
        s = regret_summary(rd, mem, opp, p, rr)
        if best is None or s["cost"] < best[1]["cost"]:
            best = (b, s, rd, mem, opp)
    R["recalibrated"] = {"bias": float(best[0]), "rho": float(best[1]["rho"]), "J": float(best[1]["J"]),
                         "read": float(best[1]["read_rate"]), "per_price": per_price(best[2], best[3], best[4], p, rr)}
    # scenario-cluster bootstrap
    sc = sorted(set(scen))
    Jb, Rb = [], []
    for _ in range(1000):
        pick = rng.choice(sc, len(sc), replace=True)
        idx = np.concatenate([np.where(scen == s)[0] for s in pick])
        s = regret_summary(dr[idx], dm[idx], do[idx], p[idx], rr[idx])
        Jb.append(s["J"])
        Rb.append(s["rho"])
    R["cluster_ci"] = {"J": [float(np.percentile(Jb, 2.5)), float(np.percentile(Jb, 97.5))],
                       "rho": [float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))]}
    s_prob = regret_summary(pread, np.array([x["mem"] for x in rp]), np.array([x["opp"] for x in rp]), p, rr)
    R["prob_weighted"] = {"J": float(s_prob["J"]), "rho": float(s_prob["rho"]), "read": float(pread.mean())}
    R["by_state"] = {"condition_holds": float(dr[cond].mean()), "condition_fails": float(dr[~cond].mean())}
    # gates
    for part in ("gate_fresh", "gate_stale"):
        g = [r for r in recs if P[r["id"]]["part"] == part]
        if g:
            gp = [roles_of(r, P[r["id"]]) for r in g]
            ga = np.array([max(x, key=x.get) for x in gp])
            R[part] = {"read": float(np.mean(ga == "read")), "act_on_memory": float(np.mean(ga == "mem")),
                       "opposite": float(np.mean(ga == "opp"))}
    # tool calls with the proper parser
    tl = [t for t in load("age_tool", model) if t["id"] in TOOL]
    if tl:
        cls, how, base_ids = [], [], []
        for t in tl:
            c, h = parse_tool(t["text"])
            cls.append(c)
            how.append(h)
            base_ids.append(TOOL[t["id"]]["base"])
        cls = np.array(cls)
        bp = [P[b] for b in base_ids]
        p_t = np.array([b["p_change"] for b in bp])
        r_t = np.array([b["r"] for b in bp])
        mem_act = np.array(["act" if b["cond_true"] else "none" for b in bp])
        # cost per class: read r; both = command on memory plus the read (r + p); act or none per memory
        is_read = (cls == "read").astype(float)
        both = (cls == "both")
        on_mem = ((cls == mem_act) | both).astype(float) * (1 - is_read)
        opp = 1 - is_read - on_mem
        cost = is_read * r_t + on_mem * p_t + opp * (1 - p_t) + both * r_t
        copt = np.minimum(p_t, r_t).mean()
        cfix = min(r_t.mean(), p_t.mean())
        # letter-probe decision on the same base situations, collapsed to read / act-on-memory / other
        idx_by_base = {}
        for i, rec in enumerate(main):
            pb = P[rec["id"]]
            idx_by_base.setdefault((pb["scen"], pb["q_day"], pb["age_s"], pb["r"], pb["cond_true"]), []).append(i)
        probe_cls = []
        for b in bp:
            ii = idx_by_base.get((b["scen"], b["q_day"], b["age_s"], b["r"], b["cond_true"]), [])
            if ii:   # decision from the role probabilities averaged over the option rotations (deterministic)
                avg = {k: float(np.mean([rp[i][k] for i in ii])) for k in ("read", "mem", "opp")}
                probe_cls.append(max(("read", "mem", "opp"), key=lambda k: avg[k]))
            else:
                probe_cls.append("na")
        tool_cls3 = np.where(cls == "read", "read", np.where((cls == mem_act) | both, "mem", "opp"))
        R["toolcall_v2"] = {"n": len(tl), "structured_share": float(np.mean(np.array(how) == "structured")),
                            "classes": {c: float(np.mean(cls == c)) for c in ("read", "act", "none", "both")},
                            "read_only_rate": float(is_read.mean()), "both_rate": float(both.mean()),
                            "rho": float((cost.mean() - copt) / (cfix - copt)),
                            "J_read_only": float(youden(is_read, p_t)),
                            "agree": float(np.mean(np.array(probe_cls) == tool_cls3)),
                            "kappa": kappa(probe_cls, tool_cls3)}
    # comprehension gate (offline)
    nr = load("age_noread", model)
    if nr:
        NR = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "noread.jsonl"), encoding="utf-8"))}
        pm = np.array([sum(v for L, v in r["probs"].items() if NR[r["id"]]["roles"][L] == NR[r["id"]]["mem_role"]) for r in nr if r["id"] in NR])
        R["noread"] = {"n": int(len(pm)), "argmax_acc": float(np.mean(pm > 0.5)), "P_mem": float(pm.mean())}
    # revision arms (letters)
    for arm in ("rule", "pgiven", "share"):
        rs = [r for r in load("age_%s" % arm, model) if r["id"] in REV]
        if not rs:
            continue
        pp = [REV[r["id"]] for r in rs]
        rp2 = [roles_of(r, b) for r, b in zip(rs, pp)]
        a2 = np.array([max(x, key=x.get) for x in rp2])
        d1, d2, d3 = (a2 == "read").astype(float), (a2 == "mem").astype(float), (a2 == "opp").astype(float)
        p2 = np.array([b["p_change"] for b in pp])
        r2 = np.array([b["r"] for b in pp])
        s = regret_summary(d1, d2, d3, p2, r2)
        # the same cells in the original scoring, for a paired comparison
        orig = [P[b["base"]] for b in pp]
        ro = [roles_of(next(x for x in main if x["id"] == b["base"]), P[b["base"]]) for b in pp] if False else None
        R["arm_" + arm] = {"n": len(rs), "read": float(d1.mean()), "J": float(s["J"]), "rho": float(s["rho"]),
                           "read_r06": float(d1[r2 == 0.6].mean()) if np.any(r2 == 0.6) else None,
                           "per_price": per_price_arm(d1, d2, d3, p2, r2)}
    base_lookup = {r["id"]: i for i, r in enumerate(main)}
    for arm in ("rule", "pgiven", "share"):
        if "arm_" + arm in R:
            ids = [REV[r["id"]]["base"] for r in load("age_%s" % arm, model) if r["id"] in REV]
            ii = [base_lookup[i] for i in ids if i in base_lookup]
            if ii:
                s0 = regret_summary(dr[ii], dm[ii], do[ii], p[ii], rr[ii])
                R["arm_" + arm]["baseline_same_cells"] = {"read": float(dr[ii].mean()), "J": float(s0["J"]), "rho": float(s0["rho"]),
                                                          "per_price": per_price_arm(dr[ii], dm[ii], do[ii], p[ii], rr[ii])}
    # reasoning arm (generation)
    rs = [r for r in load("age_reason", model) if r["id"] in REV]
    if rs:
        lab, pp = [], []
        forced = {f["id"][len("force-"):]: max(f["probs"], key=f["probs"].get) for f in load("age_force", model)}
        n_forced = 0
        for r in rs:
            b = REV[r["id"]]
            L = parse_answer(r["text"])
            if not L and r["id"] in forced:      # budget exhausted: answer forced after the model's own reasoning
                L = forced[r["id"]]
                n_forced += 1
            lab.append(b["roles"].get(L, "unparsed") if L else "unparsed")
            pp.append(b)
        lab = np.array(lab)
        role = np.array([("mem" if x == b["mem_role"] else ("read" if x == "read" else ("opp" if x in ("act", "none") else x)))
                         for x, b in zip(lab, pp)])
        mainm = np.array([b["part"] == "main" for b in pp])
        p3 = np.array([b["p_change"] for b in pp])
        r3 = np.array([b["r"] for b in pp])
        d1 = (role == "read").astype(float)
        d2 = (role == "mem").astype(float)
        d3 = 1 - d1 - d2
        s = regret_summary(d1[mainm], d2[mainm], d3[mainm], p3[mainm], r3[mainm])
        R["arm_reason"] = {"n": len(rs), "parsed": float(np.mean(lab != "unparsed")), "forced": n_forced / len(rs), "self_answered": float(np.mean([parse_answer(r["text"]) is not None for r in rs])), "read": float(d1[mainm].mean()),
                           "J": float(s["J"]), "rho": float(s["rho"]),
                           "fresh_read": float(d1[~mainm].mean()) if np.any(~mainm) else None,
                           "read_r06": float(d1[mainm & (r3 == 0.6)].mean()),
                           "per_price": per_price_arm(d1[mainm], d2[mainm], d3[mainm], p3[mainm], r3[mainm])}
    # stratified extraction
    ex = [e for e in load("age_vext", model) if e["id"] in REV]
    if ex:
        ok = {"age": [], "changes_per_day": [], "read_cost": [], "wrong_command_cost": [], "parsed": []}
        gate_d, pp = [], []
        for e in ex:
            b = P[REV[e["id"]]["base"]]
            pp.append(b)
            d = 0.0
            try:
                j = json.loads(re.search(r"\{.*\}", e["text"], re.S).group(0))
                t1 = datetime.fromisoformat(str(j["current_time"]).replace("Z", "+00:00"))
                t0 = datetime.fromisoformat(str(j["as_of"]).replace("Z", "+00:00"))
                D = (t1 - t0).total_seconds()
                ok["age"].append(abs(D - b["age_s"]) < 1)
                ok["changes_per_day"].append(abs(float(j["changes_per_day"]) - b["q_day"]) < 1e-6)
                ok["read_cost"].append(abs(float(j["read_cost"]) - b["c"]) < 1e-6)
                ok["wrong_command_cost"].append(abs(float(j["wrong_command_cost"]) - b["L"]) < 1e-6)
                pe = 0.5 * (1 - math.exp(-2 * float(j["changes_per_day"]) / 86400 * max(D, 0)))
                d = 1.0 if pe > float(j["read_cost"]) / float(j["wrong_command_cost"]) else 0.0
                ok["parsed"].append(True)
            except Exception:
                ok["parsed"].append(False)
            gate_d.append(d)
        gate_d = np.array(gate_d)
        p4 = np.array([b["p_change"] for b in pp])
        r4 = np.array([b["r"] for b in pp])
        opt = (p4 > r4).astype(float)
        s = regret_summary(gate_d, 1 - gate_d, 0 * gate_d, p4, r4)
        R["vext"] = {"n": len(ex), "field_acc": {k: float(np.mean(v)) if v else None for k, v in ok.items()},
                     "gate_rho": float(s["rho"]), "decision_flip_vs_declared": float(np.mean(gate_d != opt))}
    return R


def per_price_arm(d1, d2, d3, p, r):
    """regret per read price; at prices where a fixed rule is optimal rho is undefined and the excess cost is given."""
    out = {}
    for v in sorted(set(np.round(r, 6))):
        k = np.isclose(r, v)
        s_ = regret_summary(d1[k], d2[k], d3[k], p[k], r[k])
        gap = s_["cost_best_fixed"] - s_["cost_opt"]
        out[str(v)] = {"read": float(np.mean(d1[k])), "excess": float(s_["cost"] - s_["cost_opt"]),
                       "rho": float(s_["rho"]) if gap > 1e-9 else None}
    return out


out = {}
for m in MODELS:
    r = analyse(m)
    if r:
        out[m] = r
        print(m, json.dumps({k: r[k] for k in ("per_price", "psi_robust", "recalibrated", "cluster_ci") if k in r})[:900])
        for k in ("toolcall_v2", "gate_stale", "arm_rule", "arm_pgiven", "arm_share", "arm_reason", "vext", "noread"):
            if k in r:
                print("   ", k, r[k])
json.dump(out, open(os.path.join(RES, "analysis_age_v2.json"), "w"), indent=1)
