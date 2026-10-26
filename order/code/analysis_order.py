# -*- coding: utf-8 -*-
"""ORDER analysis (local, CPU): gates, arbitration index, cue weights against the Bayes attributor,
position reliance, certification, free-text validation and the end-to-end relocated attributor.

python analysis_order.py <runs_dir> <results_dir>
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

from metrics import arbitration, directions, position_reliance

RUNS, RES = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
PROBES = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "probes.jsonl"), encoding="utf-8"))}
FREE = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "free.jsonl"), encoding="utf-8"))}
EXT = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "extract.jsonl"), encoding="utf-8"))}
ORDER_M = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "granite3.3-8b", "olmo2-7b"]
TAG = os.environ.get("ORDER_TAG", "order_all")          # order_px: assistant turn prefilled with "Answer:"


def load(prefix, model):
    out = []
    for fn in glob.glob(os.path.join(RUNS, "**", "%s__%s.jsonl" % (prefix, model)), recursive=True):
        out += [json.loads(l) for l in open(fn, encoding="utf-8")]
    return [r for r in out if r["id"].split("-free")[0].split("-ext")[0] in PROBES]


def p_role(rec, role):
    pr = PROBES[rec["id"]]
    return sum(v for L, v in rec["probs"].items() if pr["roles"][L] == role)


def boot(fn, n_items, n=1000):
    vals = [fn(rng.integers(0, n_items, n_items)) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def events(probe):
    roles = probe["roles"]
    txt = probe["prompt"].split("Question:")[1]
    opts = {}
    for line in txt.splitlines():
        if len(line) > 3 and line[1] == ")":
            opts[roles[line[0]]] = line[3:].strip()
    return opts


def parse_ts(s):
    return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S.%f")


def analyse_model(model):
    recs = load(TAG, model) or (load("order_main", model) if TAG == "order_all" else [])
    if not recs:
        return None
    R = {"model": model, "n": len(recs)}
    mass = np.array([r["mass"] for r in recs])
    R["format_mass_median"] = float(np.median(mass))
    R["format_pass"] = float(np.mean(mass > 0.9))
    parts = {}
    for r in recs:
        parts.setdefault(PROBES[r["id"]]["part"], []).append(r)
    # ---------------- gates
    g = {}
    rs = parts.get("gate_ts", [])
    if rs:
        P = [PROBES[r["id"]] for r in rs]
        pa = np.array([p_role(r, "A") for r in rs])
        ts_A = np.array([x["D"] > 0 for x in P])
        follow = np.where(ts_A, pa, 1 - pa)
        post = np.array([x["post_A"] for x in P])
        bayes_follow = np.where(ts_A, post > 0.5, post < 0.5)
        g["gate_ts"] = {"P_follow_ts": float(follow.mean()), "bayes_follow_ts": float(bayes_follow.mean()),
                        "by_gamma": {str(gm): float(follow[np.array([x["gamma"] == gm for x in P])].mean())
                                     for gm in sorted(set(x["gamma"] for x in P))},
                        "acc": float(np.mean(np.where([x["a_first"] for x in P], pa, 1 - pa))), "n": len(rs)}
    rs = parts.get("gate_mech", [])
    if rs:
        P = [PROBES[r["id"]] for r in rs]
        pa = np.array([p_role(r, "A") for r in rs])
        me = np.array([x["fav_fwd"] for x in P])
        g["gate_hist"] = {"P_follow_hist": float(np.mean(np.where(me, pa, 1 - pa))),
                          "by_pi": {str(pi): float(np.mean(np.where(me, pa, 1 - pa)[np.array([x["pi"] == pi for x in P])]))
                                    for pi in sorted(set(x["pi"] for x in P))}, "n": len(rs)}
    R["gates"] = g
    # ---------------- main
    rs = parts.get("main", [])
    P = [PROBES[r["id"]] for r in rs]
    pa = np.array([p_role(r, "A") for r in rs])
    soft = arbitration(pa, P, soft=True)
    real = arbitration(pa, P, soft=False)
    R["main"] = {"A_soft": soft["A"], "A_real": real["A"], "acc_S_ts": soft["acc_S_ts"],
                 "acc_S_hist": soft["acc_S_mech"], "acc": real["acc"], "n": len(rs)}
    R["main"]["A_soft_ci"] = boot(lambda idx: arbitration(pa[idx], [P[i] for i in idx])["A"], len(P))
    R["main"]["A_real_ci"] = boot(lambda idx: arbitration(pa[idx], [P[i] for i in idx], soft=False)["A"], len(P))
    post = np.array([x["post_A"] for x in P])
    R["main"]["acc_bayes"] = float(np.mean(np.where([x["a_first"] for x in P], post > 0.5, post < 0.5)))
    am = (pa > 0.5).astype(float)
    R["main_argmax"] = {"A_soft": arbitration(am, P)["A"], "A_real": arbitration(am, P, soft=False)["A"],
                        "acc": arbitration(am, P, soft=False)["acc"],
                        "A_soft_ci": boot(lambda idx: arbitration(am[idx], [P[i] for i in idx])["A"], len(P)),
                        "acc_S_ts": arbitration(am, P)["acc_S_ts"], "acc_S_hist": arbitration(am, P)["acc_S_mech"]}
    R["main"]["A_soft_calibrated_bayes"] = arbitration(post, P)["A"]
    ts_A, me_A = directions(P)
    R["P_follow_ts"] = float(np.mean(np.where(ts_A, pa, 1 - pa)))
    R["P_follow_hist"] = float(np.mean(np.where(me_A, pa, 1 - pa)))
    R["position_reliance"] = position_reliance(pa, P)
    R["position_ci"] = boot(lambda idx: position_reliance(pa[idx], [P[i] for i in idx]), len(P))
    R["P_letter_A"] = float(np.mean([r["probs"].get("A", 0) for r in rs]))
    gam = np.array([x["gamma"] for x in P])
    R["by_gamma"] = {}
    for gm in sorted(set(gam)):
        m = gam == gm
        Pm = [P[i] for i in np.where(m)[0]]
        R["by_gamma"][str(gm)] = {"P_follow_ts": float(np.mean(np.where(ts_A[m], pa[m], 1 - pa[m]))),
                                  "bayes_follow_ts": float(np.mean(np.where(ts_A[m], post[m] > 0.5, post[m] < 0.5))),
                                  "A_soft": arbitration(pa[m], Pm)["A"],
                                  "A_bayes": arbitration((post[m] > 0.5).astype(float), Pm)["A"]}
    # ---------------- cue weights: logit P(A) on prior log-odds, timestamp LLR and line position (Bayes: 1, 1, 0)
    prior_lo = np.array([math.log(x["prior_A"] / (1 - x["prior_A"])) for x in P])
    ts_llr = np.array([x["g"] * x["D"] / x["sigma"] ** 2 for x in P])
    pos = np.array([1.0 if x["list_a_first"] else -1.0 for x in P])
    y = np.clip(pa, 1e-4, 1 - 1e-4)
    X = sm.add_constant(np.column_stack([prior_lo, ts_llr, pos]))
    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
        R["cue_weights"] = {"b_prior": float(fit.params[1]), "b_ts": float(fit.params[2]), "b_pos": float(fit.params[3]),
                            "se_prior": float(fit.bse[1]), "se_ts": float(fit.bse[2]), "se_pos": float(fit.bse[3])}
    except Exception as e:
        R["cue_weights"] = {"error": str(e)}
    # sign-only use of timestamps: logit on sign(D) instead of its size
    Xs = sm.add_constant(np.column_stack([prior_lo, np.sign(ts_llr), np.abs(ts_llr) * np.sign(ts_llr), pos]))
    try:
        fs = sm.GLM(y, Xs, family=sm.families.Binomial()).fit(cov_type="HC1")
        R["cue_weights_sign"] = {"b_sign": float(fs.params[2]), "b_size": float(fs.params[3]),
                                 "se_sign": float(fs.bse[2]), "se_size": float(fs.bse[3])}
    except Exception as e:
        R["cue_weights_sign"] = {"error": str(e)}
    # ---------------- certification
    rs = parts.get("cert", [])
    if rs:
        P = [PROBES[r["id"]] for r in rs]
        pc = np.array([p_role(r, x["correct"]) for r, x in zip(rs, P)])
        und = np.array([p_role(r, "undetermined") for r in rs])
        certifiable = np.array([abs(x["D"]) > 2 * x["delta"] for x in P])
        wrong_order = np.array([p_role(r, "B_first" if x["D"] > 0 else "A_first") for r, x in zip(rs, P)])
        ratio = np.array([abs(x["D"]) / x["delta"] for x in P])
        R["cert"] = {"acc": float(pc.mean()),
                     "overclaim": float((1 - und[~certifiable]).mean()),
                     "underclaim": float(und[certifiable].mean()),
                     "misorder": float(wrong_order.mean()),
                     "P_undetermined_by_ratio": {str(v): float(und[ratio == v].mean()) for v in sorted(set(ratio))},
                     "n": len(rs)}
    # ---------------- free text
    fr = load("order_free", model)
    if fr:
        agree, named = [], []
        pa_main = {r["id"]: p_role(r, "A") for r in parts.get("main", [])}
        for f in fr:
            base = PROBES[FREE[f["id"]]["base"]]
            ev = events(base)
            t = f["text"].lower()
            ia, ib = t.find(ev["A"].lower()), t.find(ev["B"].lower())
            if ia < 0 and ib < 0:
                key = lambda s: s.lower().split()[0:2]
                ia = t.find(" ".join(key(ev["A"])))
                ib = t.find(" ".join(key(ev["B"])))
            choice = None
            if ia >= 0 and (ib < 0 or ia < ib):
                choice = "A"
            elif ib >= 0:
                choice = "B"
            named.append(choice is not None)
            if choice and base["id"] in pa_main:
                agree.append((pa_main[base["id"]] > 0.5) == (choice == "A"))
        R["free"] = {"n": len(fr), "named_one_event": float(np.mean(named)),
                     "agreement_with_probe": float(np.mean(agree)) if agree else None}
    # ---------------- extraction -> external Bayes attributor
    ex = load("order_ext", model)
    if ex:
        fields, dec, pp = {}, [], []
        for e in ex:
            base = PROBES[EXT[e["id"]]["base"]]
            pp.append(base)
            m = re.search(r"\{.*\}", e["text"], re.S)
            d = 0.5
            try:
                j = json.loads(m.group(0))
                D = (parse_ts(j["timestamp_B"]) - parse_ts(j["timestamp_A"])).total_seconds()
                sd = float(j["offset_sd_s"])
                lat = float(j["latency_s"])
                hf = str(j["history_first"]).strip().upper()
                hp = float(j["history_percent"]) / 100.0
                fields.setdefault("D", []).append(abs(D - base["D"]) < 0.002)
                fields.setdefault("offset_sd", []).append(abs(sd - base["sigma"]) < 1e-6)
                fields.setdefault("latency", []).append(abs(lat - base["g"]) < 0.011)
                fields.setdefault("history_first", []).append(hf == ("A" if base["fav_fwd"] else "B"))
                fields.setdefault("history_percent", []).append(abs(hp - base["pi"]) < 1e-6)
                prior_A = hp if hf == "A" else (1 - hp if hf == "B" else 0.5)
                prior_A = min(max(prior_A, 1e-3), 1 - 1e-3)
                lo = math.log(prior_A / (1 - prior_A)) + lat * D / sd ** 2
                d = 1.0 if lo > 0 else 0.0
                fields.setdefault("parsed", []).append(True)
            except Exception:
                fields.setdefault("parsed", []).append(False)
            dec.append(d)
        dec = np.array(dec)
        own = np.array([p_role({"id": b["id"], "probs": next((r["probs"] for r in parts["main"] if r["id"] == b["id"]), {})}, "A")
                        if any(r["id"] == b["id"] for r in parts.get("main", [])) else np.nan for b in pp])
        post = np.array([b["post_A"] for b in pp])
        R["attributor_e2e"] = {"n": len(ex), "A_soft": arbitration(dec, pp)["A"], "acc": arbitration(dec, pp, soft=False)["acc"],
                               "A_bayes_same_items": arbitration((post > 0.5).astype(float), pp)["A"],
                               "A_model_same_items": arbitration(np.nan_to_num(own, nan=0.5), pp)["A"],
                               "field_acc": {k: float(np.mean(v)) for k, v in fields.items()}}
    return R


def main():
    models = sorted({re.sub(r"^.*__|\.jsonl$", "", os.path.basename(f))
                     for f in glob.glob(os.path.join(RUNS, "**", TAG + "__*.jsonl"), recursive=True)},
                    key=lambda m: ORDER_M.index(m) if m in ORDER_M else 99)
    out = {}
    for m in models:
        r = analyse_model(m)
        if not r:
            continue
        out[m] = r
        mm = r["main"]
        aa = r["main_argmax"]
        print("%-15s n %4d mass %.3f | argmax A %+.3f [%+.3f,%+.3f] | prob A %+.3f A_real %+.3f acc %.3f (Bayes %.3f) | ts %.2f hist %.2f pos %+.3f"
              % (m, r["n"], r["format_mass_median"], aa["A_soft"], aa["A_soft_ci"][0], aa["A_soft_ci"][1], mm["A_soft"], mm["A_real"],
                 mm["acc"], mm["acc_bayes"], r["P_follow_ts"], r["P_follow_hist"], r["position_reliance"]))
        print("   gates", r["gates"])
        print("   by gamma", r["by_gamma"])
        print("   cues", r.get("cue_weights"), r.get("cue_weights_sign"))
        print("   cert", r.get("cert"))
        print("   free", r.get("free"), " e2e", r.get("attributor_e2e"))
    json.dump(out, open(os.path.join(RES, "analysis_%s.json" % TAG.replace("order_all", "order")), "w"), indent=1)


if __name__ == "__main__":
    main()
