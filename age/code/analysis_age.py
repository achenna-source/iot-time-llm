# -*- coding: utf-8 -*-
"""AGE analysis (local, CPU): gates, staleness discrimination J, normalised regret, the age/volatility/price
symmetry test, presentation variants, native tool-call validation and the end-to-end freshness gate.

python analysis_age.py <runs_dir> <results_dir>
runs_dir holds age_all__<model>.jsonl (or age_main__), age_tool__<model>.jsonl, age_ext__<model>.jsonl.
"""
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
import statsmodels.api as sm

from metrics import regret_summary, youden

RUNS, RES = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
PROBES = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "probes.jsonl"), encoding="utf-8"))}
TOOL = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "toolcall.jsonl"), encoding="utf-8"))}
EXT = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "extract.jsonl"), encoding="utf-8"))}
NOREAD = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "noread.jsonl"), encoding="utf-8"))}
ORDER_M = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "granite3.3-8b", "olmo2-7b"]


def load(prefix, model):
    out = []
    for fn in glob.glob(os.path.join(RUNS, "**", "%s__%s.jsonl" % (prefix, model)), recursive=True):
        out += [json.loads(l) for l in open(fn, encoding="utf-8")]
    return out


def role_probs(rec):
    pr = PROBES[rec["id"]]
    d = {"read": 0.0, "mem": 0.0, "opp": 0.0}
    for L, v in rec["probs"].items():
        role = pr["roles"][L]
        if role == "read":
            d["read"] += v
        elif role == pr["mem_role"]:
            d["mem"] += v
        else:
            d["opp"] += v
    return d


def base_key(p):
    return (p["part"], p["scen"], p["q_day"], p["age_s"], p["r"], p["cond_true"])


def boot(fn, groups, n=1000):
    """cluster bootstrap over base items (rotations of one situation stay together)."""
    keys = sorted(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        vals.append(fn(idx))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def frac_logit(y, X):
    """fractional logit (quasi-binomial GLM) with robust standard errors."""
    y = np.clip(y, 1e-4, 1 - 1e-4)
    m = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit(cov_type="HC1")
    return m.params, m.bse


def analyse_model(model):
    recs = load("age_all", model) or load("age_main", model)
    if not recs:
        return None
    R = {"model": model, "n": len(recs)}
    mass = np.array([r["mass"] for r in recs])
    R["format_mass_median"] = float(np.median(mass))
    R["format_pass"] = float(np.mean(mass > 0.9))
    by_part = {}
    for r in recs:
        by_part.setdefault(PROBES[r["id"]]["part"], []).append(r)
    # ---------------- gates
    g = {}
    for part in ("gate_fresh", "gate_stale"):
        rs = by_part.get(part, [])
        if rs:
            rp = [role_probs(r) for r in rs]
            pread = np.array([x["read"] for x in rp])
            cond = np.array([x["mem"] / max(x["mem"] + x["opp"], 1e-9) for x in rp])
            amx = np.array([max(x, key=x.get) for x in rp])
            g[part] = {"P_read": float(pread.mean()), "act_consistent_given_act": float(cond.mean()),
                       "argmax_read": float(np.mean(amx == "read")), "n": len(rs)}
    R["gates"] = g
    # ---------------- main
    rs = by_part.get("main", [])
    P = [PROBES[r["id"]] for r in rs]
    rp = [role_probs(r) for r in rs]
    pr = np.array([x["read"] for x in rp])
    pm = np.array([x["mem"] for x in rp])
    po = np.array([x["opp"] for x in rp])
    p = np.array([x["p_change"] for x in P])
    rr = np.array([x["r"] for x in P])
    groups = {}
    for i, x in enumerate(P):
        groups.setdefault(base_key(x), []).append(i)
    groups = {k: np.array(v) for k, v in groups.items()}
    am = np.array([max(x, key=x.get) for x in rp])
    dr, dm, do = (am == "read").astype(float), (am == "mem").astype(float), (am == "opp").astype(float)
    s = regret_summary(pr, pm, po, p, rr)
    R["main"] = {k: float(v) for k, v in s.items()}
    R["main"]["J_ci"] = boot(lambda idx: youden(pr[idx], p[idx]), groups)
    R["main"]["rho_ci"] = boot(lambda idx: regret_summary(pr[idx], pm[idx], po[idx], p[idx], rr[idx])["rho"], groups)
    sd = regret_summary(dr, dm, do, p, rr)
    R["main_argmax"] = {k: float(v) for k, v in sd.items()}
    R["main_argmax"]["J_ci"] = boot(lambda idx: youden(dr[idx], p[idx]), groups)
    R["main_argmax"]["rho_ci"] = boot(lambda idx: regret_summary(dr[idx], dm[idx], do[idx], p[idx], rr[idx])["rho"], groups)
    R["P_opp"] = float(po.mean())
    R["cost_ratio_argmax"] = float(sd["cost"] / sd["cost_opt"])
    R["reads_per_1000_argmax"] = float(1000 * dr.mean())
    # rotation (letter) check: P(read) by letter position of the read option
    pos = np.array([[L for L, role in x["roles"].items() if role == "read"][0] for x in P])
    R["P_read_by_letter"] = {L: float(pr[pos == L].mean()) for L in "ABC"}
    # marginal profiles
    age = np.array([x["age_s"] for x in P], float)
    q = np.array([x["q_day"] for x in P], float)
    R["P_read_by_age"] = {str(int(a)): float(pr[age == a].mean()) for a in sorted(set(age))}
    R["P_read_by_q"] = {str(v): float(pr[q == v].mean()) for v in sorted(set(q))}
    R["P_read_by_r"] = {str(v): float(pr[rr == v].mean()) for v in sorted(set(rr))}
    R["opt_read_by_r"] = {str(v): float(np.mean(p[rr == v] > v)) for v in sorted(set(rr))}
    u = q / 86400 * age
    R["curve"] = {}
    for v in sorted(set(rr)):
        m = rr == v
        cu = {}
        for uu, a1, a2 in zip(u[m], dr[m], pr[m]):
            cu.setdefault(round(float(uu), 8), []).append((a1, a2))
        R["curve"][str(v)] = [[k, float(np.mean([x[0] for x in w])), float(np.mean([x[1] for x in w]))] for k, w in sorted(cu.items())]
    # ---------------- symmetry test (fractional logit on log age, log volatility, log price)
    X = np.column_stack([np.log(age), np.log(q), np.log(rr)])
    try:
        b, se = frac_logit(pr, X)
        R["sym"] = {"b_age": float(b[1]), "b_q": float(b[2]), "b_r": float(b[3]), "se_age": float(se[1]),
                    "se_q": float(se[2]), "se_r": float(se[3]),
                    "ratio_q_age": float(b[2] / b[1]) if abs(b[1]) > 1e-6 else None,
                    "ratio_r_age": float(-b[3] / b[1]) if abs(b[1]) > 1e-6 else None}
    except Exception as e:
        R["sym"] = {"error": str(e)}
    # equal normalised-age pairs: (q, D) vs (4q, D/4)
    cell = {}
    for i, x in enumerate(P):
        cell.setdefault((x["q_day"], x["age_s"]), []).append(pr[i])
    pairs = []
    for (qq, aa), v in cell.items():
        k2 = (qq * 4, aa / 4)
        if k2 in cell:
            pairs.append((np.mean(v), np.mean(cell[k2]), qq, aa))
    if pairs:
        d = np.array([a - b for a, b, _, _ in pairs])
        R["equal_u_pairs"] = {"n": len(pairs), "mean_diff_slowold_minus_fastyoung": float(d.mean()),
                              "mean_abs_diff": float(np.abs(d).mean())}
    # ---------------- variants (same two scenarios, matched cells)
    var = {}
    for part in ("rel", "qual", "none"):
        rv = by_part.get(part, [])
        if not rv:
            continue
        Pv = [PROBES[r["id"]] for r in rv]
        rpv = [role_probs(r) for r in rv]
        prv = np.array([x["read"] for x in rpv])
        pv = np.array([x["p_change"] for x in Pv])
        if part == "none":
            # age not shown: J against the age-averaged staleness of each (scen, q, r, cond) cell
            pv = np.array([np.mean([0.5 * (1 - math.exp(-2 * x["q_day"] / 86400 * a)) for a in
                                    (120, 600, 2700, 10800, 43200, 172800)]) for x in Pv])
        sv = regret_summary(prv, [y["mem"] for y in rpv], [y["opp"] for y in rpv], pv, [x["r"] for x in Pv])
        var[part] = {"J": float(sv["J"]), "rho": float(sv["rho"]), "read_rate": float(sv["read_rate"]), "n": len(rv)}
    # matched absolute-time baseline on the same scenarios
    m_abs = np.array([x["scen"] in (0, 5) for x in P])
    sb = regret_summary(pr[m_abs], pm[m_abs], po[m_abs], p[m_abs], rr[m_abs])
    var["abs_same_scen"] = {"J": float(sb["J"]), "rho": float(sb["rho"]), "read_rate": float(sb["read_rate"])}
    R["variants"] = var
    # ---------------- comprehension gate: re-reading unavailable, act on the remembered state is correct
    nr = load("age_noread", model)
    if nr:
        pm_nr = np.array([sum(v for L, v in r["probs"].items() if NOREAD[r["id"]]["roles"][L] == NOREAD[r["id"]]["mem_role"]) for r in nr])
        R["noread"] = {"n": len(nr), "P_mem": float(pm_nr.mean()), "argmax_acc": float(np.mean(pm_nr > 0.5)),
                       "mass": float(np.median([r["mass"] for r in nr]))}
    # ---------------- native tool calls
    tl = load("age_tool", model)
    if tl:
        rows = []
        for t in tl:
            base = PROBES[TOOL[t["id"]]["base"]]
            txt = t["text"]
            i_read = txt.find("read_state")
            i_exec = txt.find("execute_command")
            if i_read >= 0 and (i_exec < 0 or i_read < i_exec):
                act = "read"
            elif i_exec >= 0:
                act = "act"
            else:
                act = "none"
            # did the act match the remembered state? (act == command; none == do nothing)
            mem_act = "act" if base["cond_true"] else "none"
            role = "read" if act == "read" else ("mem" if act == mem_act else "opp")
            rows.append((base, role))
        trd = np.array([1.0 if r == "read" else 0.0 for _, r in rows])
        tm = np.array([1.0 if r == "mem" else 0.0 for _, r in rows])
        to = np.array([1.0 if r == "opp" else 0.0 for _, r in rows])
        tp = np.array([b["p_change"] for b, _ in rows])
        trr = np.array([b["r"] for b, _ in rows])
        st = regret_summary(trd, tm, to, tp, trr)
        # letter-probe read probability on the same base situations (averaged over rotations)
        lp = []
        for b, _ in rows:
            key = base_key(b)
            lp.append(dr[groups[key]].mean() if key in groups else np.nan)
        lp = np.array(lp)
        ok = ~np.isnan(lp)
        R["toolcall"] = {"n": len(rows), "read_rate": float(trd.mean()), "J": float(st["J"]), "rho": float(st["rho"]),
                         "probe_read_rate_same_items": float(np.nanmean(lp)),
                         "agreement_argmax": float(np.mean((lp[ok] > 0.5) == (trd[ok] > 0.5))),
                         "read_rate_r06": float(trd[trr == 0.6].mean()) if np.any(trr == 0.6) else None,
                         "read_rate_r01": float(trd[trr == 0.1].mean()) if np.any(trr == 0.1) else None}
    # ---------------- extraction -> external freshness gate
    ex = load("age_ext", model)
    if ex:
        n_ok, field_ok, gate_dec, own_dec, ps, rs_ = 0, {}, [], [], [], []
        for e in ex:
            base = PROBES[EXT[e["id"]]["base"]]
            m = re.search(r"\{.*\}", e["text"], re.S)
            truth = {"changes_per_day": base["q_day"], "read_cost": base["c"], "wrong_command_cost": base["L"]}
            dec = None
            try:
                j = json.loads(m.group(0))
                t1 = datetime.fromisoformat(str(j["current_time"]).replace("Z", "+00:00"))
                t0 = datetime.fromisoformat(str(j["as_of"]).replace("Z", "+00:00"))
                D = (t1 - t0).total_seconds()
                qd = float(j["changes_per_day"])
                c, L = float(j["read_cost"]), float(j["wrong_command_cost"])
                field_ok.setdefault("age", []).append(abs(D - base["age_s"]) < 1)
                for k, v in truth.items():
                    field_ok.setdefault(k, []).append(abs(float(j[k]) - v) < 1e-6)
                pe = 0.5 * (1 - math.exp(-2 * qd / 86400 * max(D, 0)))
                dec = pe > c / L
                n_ok += 1
            except Exception:
                field_ok.setdefault("parse", []).append(False)
            gate_dec.append(1.0 if dec else 0.0)      # parse failure -> act on memory
            key = base_key(base)
            own_dec.append(dr[groups[key]].mean() if key in groups else np.nan)
            ps.append(base["p_change"])
            rs_.append(base["r"])
        gate_dec, ps, rs_ = np.array(gate_dec), np.array(ps), np.array(rs_)
        own = np.array(own_dec)
        se2 = regret_summary(gate_dec, 1 - gate_dec, 0 * gate_dec, ps, rs_)
        opt = (ps > rs_).astype(float)
        so = regret_summary(opt, 1 - opt, 0 * opt, ps, rs_)
        sown = regret_summary(own, 1 - own, 0 * own, ps, rs_)
        R["gate_e2e"] = {"n": len(ex), "parsed": n_ok, "J": float(se2["J"]), "rho": float(se2["rho"]),
                         "J_opt_same_items": float(so["J"]), "J_model_same_items": float(sown["J"]),
                         "rho_model_same_items": float(sown["rho"]),
                         "field_acc": {k: float(np.mean(v)) for k, v in field_ok.items()}}
    return R


def main():
    models = sorted({re.sub(r"^.*__|\.jsonl$", "", os.path.basename(f))
                     for f in glob.glob(os.path.join(RUNS, "**", "age_*__*.jsonl"), recursive=True)},
                    key=lambda m: ORDER_M.index(m) if m in ORDER_M else 99)
    out = {}
    # reference: the same logistic fit on a smoothed optimal policy (ratios do not depend on the smoothing)
    main = [p for p in PROBES.values() if p["part"] == "main"]
    pp = np.array([max(p["p_change"], 1e-9) for p in main])
    rr0 = np.array([p["r"] for p in main])
    pref = 1 / (1 + np.exp(-2 * (np.log(pp) - np.log(rr0))))
    X0 = np.column_stack([np.log([p["age_s"] for p in main]), np.log([p["q_day"] for p in main]), np.log(rr0)])
    b0, _ = frac_logit(pref, X0)
    out["_reference"] = {"sym": {"b_age": float(b0[1]), "b_q": float(b0[2]), "b_r": float(b0[3]),
                                 "ratio_q_age": float(b0[2] / b0[1]), "ratio_r_age": float(-b0[3] / b0[1])}}
    print("reference ratios", out["_reference"])
    for m in models:
        r = analyse_model(m)
        if r:
            out[m] = r
            mm = r["main"]
            aa = r["main_argmax"]
            print("%-15s n %5d mass %.3f | argmax J %+.3f [%+.3f,%+.3f] rho %.3f [%.3f,%.3f] read %.2f | prob J %+.3f rho %.3f | gates fresh %.2f stale %.2f"
                  % (m, r["n"], r["format_mass_median"], aa["J"], aa["J_ci"][0], aa["J_ci"][1], aa["rho"], aa["rho_ci"][0],
                     aa["rho_ci"][1], aa["read_rate"], mm["J"], mm["rho"], r["gates"].get("gate_fresh", {}).get("P_read", float("nan")),
                     r["gates"].get("gate_stale", {}).get("P_read", float("nan"))))
            print("   by r", r["P_read_by_r"], " opt", r["opt_read_by_r"])
            print("   sym", r.get("sym"), " letters", r["P_read_by_letter"])
            if "toolcall" in r:
                print("   tool", r["toolcall"])
            if "gate_e2e" in r:
                print("   gate", r["gate_e2e"])
            print("   variants", r["variants"])
    json.dump(out, open(os.path.join(RES, "analysis_age.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
