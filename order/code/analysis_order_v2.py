# -*- coding: utf-8 -*-
"""ORDER analysis, revision 2 (after the simulated review).

Primary scoring: assistant turn prefilled with "Answer:" (order_px) for every model.
- decision (argmax) and probability-weighted versions of every metric, each with its own reference
  (Bayes decisions; calibrated Bayes posterior)
- arbitration index by prior stratum and by latency ratio; weighted AUC of P(follow timestamps) between S_ts and
  S_hist; Youden-optimal ceiling
- cue model with sign(D), size of the evidence, history log-odds, line position and an intercept (preference for
  event A, which is always the textbook direction and is named first in the process note)
- certification with argmax rates and the always-abstain baseline
- stratified free-text validation (exact copy of an event name), Cohen's kappa, free-text A
- stratified extraction: field accuracy, attributor A with paired bootstrap CI against declared fields, flip rate
- D-stated arm (timestamp difference given in words) and reasoning arm (parsed "Answer: X")
python analysis_order_v2.py <runs_dir> <results_dir>
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
rng = np.random.default_rng(3)
P = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, "probes.jsonl"), encoding="utf-8"))}
REV = {}
for arm in ("reason", "dstated", "free", "ext"):
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


def pA(rec, probe):
    return sum(v for L, v in rec["probs"].items() if probe["roles"][L] == "A")


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def wauc(score, rows):
    """weighted AUC of score (P(follow ts)) between S_ts and S_hist items, posterior weights."""
    ts_A, me_A = directions(rows)
    post = np.array([r["post_A"] for r in rows])
    dis = ts_A != me_A
    w_ts = np.where(dis, np.where(ts_A, post, 1 - post), 0)
    w_h = np.where(dis, np.where(me_A, post, 1 - post), 0)
    i, j = np.where(w_ts > 0)[0], np.where(w_h > 0)[0]
    s_i, s_j = score[i][:, None], score[j][None, :]
    num = np.sum(w_ts[i][:, None] * w_h[j][None, :] * ((s_i > s_j) + 0.5 * (s_i == s_j)))
    return float(num / (w_ts[i].sum() * w_h[j].sum()))


def boot_A(dec, rows, n=1000):
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        vals.append(arbitration(dec[idx], [rows[k] for k in idx])["A"])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main_block(recs, tag):
    rows = [P[r["id"]] for r in recs if P[r["id"]]["part"] == "main"]
    pa = np.array([pA(r, P[r["id"]]) for r in recs if P[r["id"]]["part"] == "main"])
    mass = np.array([r["mass"] for r in recs if P[r["id"]]["part"] == "main"])
    dec = (pa > 0.5).astype(float)
    post = np.array([r["post_A"] for r in rows])
    ts_A, me_A = directions(rows)
    out = {"mass_median": float(np.median(mass)), "mass_share_gt09": float(np.mean(mass > 0.9))}
    out["A_dec"] = arbitration(dec, rows)["A"]
    out["A_dec_ci"] = boot_A(dec, rows)
    out["A_prob"] = arbitration(pa, rows)["A"]
    out["acc_dec"] = arbitration(dec, rows, soft=False)["acc"]
    out["acc_prob"] = arbitration(pa, rows, soft=False)["acc"]
    out["follow_ts_dec"] = float(np.mean(np.where(ts_A, dec, 1 - dec)))
    out["follow_hist_dec"] = float(np.mean(np.where(me_A, dec, 1 - dec)))
    out["position_dec"] = position_reliance(dec, rows)
    favA = np.array([r["fav_fwd"] for r in rows])
    out["follow_hist_when_fav_A"] = float(np.mean(dec[favA]))
    out["follow_hist_when_fav_B"] = float(np.mean(1 - dec[~favA]))
    out["position_prob"] = position_reliance(pa, rows)
    pi = np.array([r["pi"] for r in rows])
    gam = np.array([r["gamma"] for r in rows])
    out["A_by_pi"] = {str(v): arbitration(dec[pi == v], [rows[k] for k in np.where(pi == v)[0]])["A"] for v in (0.7, 0.9)}
    out["A_bayes_by_pi"] = {str(v): arbitration((post[pi == v] > 0.5).astype(float), [rows[k] for k in np.where(pi == v)[0]])["A"] for v in (0.7, 0.9)}
    out["A_by_gamma"] = {str(v): arbitration(dec[gam == v], [rows[k] for k in np.where(gam == v)[0]])["A"] for v in (0.5, 1.0, 2.0)}
    score_ts = np.where(ts_A, pa, 1 - pa)
    out["auc"] = wauc(score_ts, rows)
    out["auc_bayes"] = wauc(np.where(ts_A, post, 1 - post), rows)
    # disagreement-only follow-ts curve by gamma
    dis = ts_A != me_A
    out["follow_ts_dis_by_gamma"] = {str(v): float(np.mean(score_ts[(gam == v) & dis])) for v in (0.5, 1.0, 2.0)}
    out["follow_ts_dis_by_gamma_bayes"] = {str(v): float(np.mean(np.where(ts_A, post > 0.5, post < 0.5)[(gam == v) & dis])) for v in (0.5, 1.0, 2.0)}
    # cue model with sign, size, history, position and intercept (event A = textbook direction, named first)
    prior_lo = np.array([math.log(r["prior_A"] / (1 - r["prior_A"])) for r in rows])
    llr = np.array([r["g"] * r["D"] / r["sigma"] ** 2 for r in rows])
    pos = np.array([1.0 if r["list_a_first"] else -1.0 for r in rows])
    X = sm.add_constant(np.column_stack([prior_lo, np.sign(llr), llr, pos]))
    try:
        fit = sm.GLM(np.clip(pa, 1e-4, 1 - 1e-4), X, family=sm.families.Binomial()).fit(cov_type="HC1")
        names = ["intercept_A", "history", "sign_D", "size_llr", "position"]
        out["cues"] = {n: [float(b), float(1.96 * s)] for n, b, s in zip(names, fit.params, fit.bse)}
    except Exception as e:
        out["cues"] = {"error": str(e)}
    return out


def gates(recs):
    out = {}
    for part in ("gate_ts", "gate_mech"):
        rows = [P[r["id"]] for r in recs if P[r["id"]]["part"] == part]
        if not rows:
            continue
        pa = np.array([pA(r, P[r["id"]]) for r in recs if P[r["id"]]["part"] == part])
        post = np.array([r["post_A"] for r in rows])
        if part == "gate_ts":
            ts_A = np.array([r["D"] > 0 for r in rows])
            out[part] = {"dec": float(np.mean(np.where(ts_A, pa > 0.5, pa < 0.5))), "prob": float(np.mean(np.where(ts_A, pa, 1 - pa))),
                         "bayes_dec": float(np.mean(np.where(ts_A, post > 0.5, post < 0.5))),
                         "bayes_calibrated": float(np.mean(np.where(ts_A, post, 1 - post)))}
        else:
            me = np.array([r["fav_fwd"] for r in rows])
            out[part] = {"dec": float(np.mean(np.where(me, pa > 0.5, pa < 0.5))), "prob": float(np.mean(np.where(me, pa, 1 - pa))),
                         "bayes_dec": 1.0, "bayes_calibrated": float(np.mean([r["pi"] for r in rows]))}
    return out


def cert(recs):
    rows = [(r, P[r["id"]]) for r in recs if P[r["id"]]["part"] == "cert"]
    if not rows:
        return None
    probs = [{P_["roles"][L]: v for L, v in r["probs"].items()} for r, P_ in rows]
    dec = [max(p, key=p.get) for p in probs]
    corr = [P_["correct"] for _, P_ in rows]
    certif = np.array([abs(P_["D"]) > 2 * P_["delta"] for _, P_ in rows])
    dec = np.array(dec)
    wrong_order = np.array([(d == "B_first" and P_["D"] > 0) or (d == "A_first" and P_["D"] < 0) for d, (_, P_) in zip(dec, rows)])
    ratio = np.array([round(abs(P_["D"]) / P_["delta"], 2) for _, P_ in rows])
    pu = np.array([p.get("undetermined", 0.0) / max(sum(p.values()), 1e-9) for p in probs])
    by_ratio = {str(v): {"dec": float(np.mean(dec[ratio == v] == "undetermined")), "prob": float(np.mean(pu[ratio == v]))}
                for v in sorted(set(ratio))}
    return {"by_ratio": by_ratio, "acc_dec": float(np.mean(dec == np.array(corr))), "overclaim_dec": float(np.mean(dec[~certif] != "undetermined")),
            "underclaim_dec": float(np.mean(dec[certif] == "undetermined")), "misorder_dec": float(np.mean(wrong_order)),
            "always_abstain": float(np.mean(np.array(corr) == "undetermined"))}


def parse_ts(s):
    return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S.%f")


ANS = re.compile(r"answer\s*[:\-]?\s*\**\s*\(?([A-C])\b", re.I)


def within_pi(dec, rows):
    """mean of the arbitration index within each history prior (removes the switching between priors)."""
    pis = np.array([r["pi"] for r in rows])
    vals = [arbitration(dec[pis == v], [rows[k] for k in np.where(pis == v)[0]])["A"] for v in (0.7, 0.9) if np.sum(pis == v) > 10]
    return float(np.mean(vals)) if vals else None


def analyse(model):
    px = [r for r in load("order_px", model) if r["id"] in P]
    if not px:
        return None
    R = {"scoring": "prefix"}
    R["main"] = main_block(px, "px")
    R["gates"] = gates(px)
    np_ = [r for r in load("order_all", model) if r["id"] in P]
    if np_:
        R["main_noprefix"] = main_block(np_, "np")
        R["cert"] = cert(np_)
    R["cert_px"] = cert(px)
    main_px = {r["id"]: pA(r, P[r["id"]]) for r in px if P[r["id"]]["part"] == "main"}
    # stratified free text
    fr = [f for f in load("order_vfree", model) if f["id"] in REV]
    if fr:
        named, agree_lab, probe_lab, dec_free, rows_f = [], [], [], [], []
        for f in fr:
            b = P[REV[f["id"]]["base"]]
            txt = b["prompt"].split("Question:")[1]
            ev = {b["roles"][l[0]]: l[3:].strip().lower() for l in txt.splitlines() if len(l) > 3 and l[1] == ")"}
            t = f["text"].strip().strip('"').strip(".").lower()
            ia, ib = ev["A"] in t, ev["B"] in t
            lab = "A" if ia and not ib else ("B" if ib and not ia else None)
            named.append(lab is not None)
            if lab and b["id"] in main_px:
                agree_lab.append(lab)
                probe_lab.append("A" if main_px[b["id"]] > 0.5 else "B")
                dec_free.append(1.0 if lab == "A" else 0.0)
                rows_f.append(b)
        R["free_v2"] = {"n": len(fr), "named_exactly_one": float(np.mean(named)),
                        "agreement": float(np.mean(np.array(agree_lab) == np.array(probe_lab))) if agree_lab else None,
                        "kappa": kappa(agree_lab, probe_lab) if agree_lab else None,
                        "A_free": arbitration(np.array(dec_free), rows_f)["A"] if len(rows_f) > 20 else None,
                        "A_probe_same_items": arbitration(np.array([1.0 if x == "A" else 0.0 for x in probe_lab]), rows_f)["A"] if len(rows_f) > 20 else None}
    # stratified extraction
    ex = [e for e in load("order_vext", model) if e["id"] in REV]
    if ex:
        fa = {k: [] for k in ("D", "offset_sd", "latency", "history_first", "history_percent", "parsed")}
        dec, rows_e = [], []
        for e in ex:
            b = P[REV[e["id"]]["base"]]
            rows_e.append(b)
            d = 0.5
            try:
                j = json.loads(re.search(r"\{.*\}", e["text"], re.S).group(0))
                D = (parse_ts(j["timestamp_B"]) - parse_ts(j["timestamp_A"])).total_seconds()
                sd, lat = float(j["offset_sd_s"]), float(j["latency_s"])
                hf = str(j["history_first"]).strip().upper()
                hp = float(j["history_percent"]) / 100
                fa["D"].append(abs(D - b["D"]) < 0.002)
                fa["offset_sd"].append(abs(sd - b["sigma"]) < 1e-6)
                fa["latency"].append(abs(lat - b["g"]) < 0.011)
                fa["history_first"].append(hf == ("A" if b["fav_fwd"] else "B"))
                fa["history_percent"].append(abs(hp - b["pi"]) < 1e-6)
                pA_ = hp if hf == "A" else (1 - hp if hf == "B" else 0.5)
                pA_ = min(max(pA_, 1e-3), 1 - 1e-3)
                d = 1.0 if math.log(pA_ / (1 - pA_)) + lat * D / sd ** 2 > 0 else 0.0
                fa["parsed"].append(True)
            except Exception:
                fa["parsed"].append(False)
            dec.append(d)
        dec = np.array(dec)
        bay = np.array([1.0 if r["post_A"] > 0.5 else 0.0 for r in rows_e])
        diffs = []
        for _ in range(1000):
            idx = rng.integers(0, len(rows_e), len(rows_e))
            rr = [rows_e[k] for k in idx]
            diffs.append(arbitration(dec[idx], rr)["A"] - arbitration(bay[idx], rr)["A"])
        own = np.array([1.0 if main_px.get(b["id"], 0.5) > 0.5 else 0.0 for b in rows_e])
        R["ext_v2"] = {"n": len(ex), "A_model_same": arbitration(own, rows_e)["A"], "field_acc": {k: float(np.mean(v)) if v else None for k, v in fa.items()},
                       "A_attributor": arbitration(dec, rows_e)["A"], "A_bayes_same": arbitration(bay, rows_e)["A"],
                       "A_diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                       "flip_rate": float(np.mean(dec != bay))}
    # D-stated arm
    ds = [r for r in load("order_dstated", model) if r["id"] in REV]
    if ds:
        rows_d = [P[REV[r["id"]]["base"]] for r in ds]
        pa = np.array([sum(v for L, v in r["probs"].items() if REV[r["id"]]["roles"][L] == "A") for r in ds])
        dec = (pa > 0.5).astype(float)
        base_dec = np.array([1.0 if main_px.get(b["id"], 0.5) > 0.5 else 0.0 for b in rows_d])
        R["arm_dstated"] = {"n": len(ds), "A_dec": arbitration(dec, rows_d)["A"], "A_dec_ci": boot_A(dec, rows_d),
                            "A_baseline_same_items": arbitration(base_dec, rows_d)["A"],
                            "acc_dec": arbitration(dec, rows_d, soft=False)["acc"],
                            "A_within_pi": within_pi(dec, rows_d), "A_within_pi_baseline": within_pi(base_dec, rows_d),
                            "A_within_pi_bayes": within_pi(np.array([1.0 if b["post_A"] > 0.5 else 0.0 for b in rows_d]), rows_d)}
    # reasoning arm
    rs = [r for r in load("order_reason", model) if r["id"] in REV]
    if rs:
        main_r, cert_r = [], []
        forced = {f["id"][len("force-"):]: max(f["probs"], key=f["probs"].get) for f in load("order_force", model)}
        n_forced = 0
        for r in rs:
            b = REV[r["id"]]
            m = ANS.findall(r["text"])
            L = m[-1].upper() if m else None
            if not L and r["id"] in forced:      # budget exhausted: answer forced after the model's own reasoning
                L = forced[r["id"]]
                n_forced += 1
            role = b["roles"].get(L) if L else None
            (main_r if b["part"] == "main" else cert_r).append((role, P[b["base"]]))
        mr = [(ro, b) for ro, b in main_r if ro in ("A", "B")]
        dec = np.array([1.0 if ro == "A" else 0.0 for ro, _ in mr])
        rows_r = [b for _, b in mr]
        base_dec = np.array([1.0 if main_px.get(b["id"], 0.5) > 0.5 else 0.0 for b in rows_r])
        cr = [(ro, b) for ro, b in cert_r if ro]
        R["arm_reason"] = {"n_main": len(main_r), "parsed_main": len(mr) / max(len(main_r), 1), "forced": n_forced / len(rs), "self_answered": float(np.mean([bool(ANS.findall(r["text"])) for r in rs if REV[r["id"]]["part"] == "main"])),
                           "A_dec": arbitration(dec, rows_r)["A"] if len(rows_r) > 20 else None,
                           "A_dec_ci": boot_A(dec, rows_r) if len(rows_r) > 20 else None,
                           "A_baseline_same_items": arbitration(base_dec, rows_r)["A"] if len(rows_r) > 20 else None,
                           "acc_dec": arbitration(dec, rows_r, soft=False)["acc"] if len(rows_r) > 20 else None,
                           "A_within_pi": within_pi(dec, rows_r) if len(rows_r) > 20 else None,
                           "A_within_pi_baseline": within_pi(base_dec, rows_r) if len(rows_r) > 20 else None,
                           "A_bayes_same_items": arbitration(np.array([1.0 if b["post_A"] > 0.5 else 0.0 for b in rows_r]), rows_r)["A"] if len(rows_r) > 20 else None,
                           "cert_acc": float(np.mean([ro == b["correct"] for ro, b in cr])) if cr else None,
                           "cert_acc_baseline_abstain": float(np.mean([b["correct"] == "undetermined" for _, b in cert_r])) if cert_r else None,
                           "cert_parsed": len(cr) / max(len(cert_r), 1)}
    return R


out = {}
for m in MODELS:
    r = analyse(m)
    if r:
        out[m] = r
        mm = r["main"]
        print("%-14s A_dec %+.3f %s acc %.3f | by pi %s (Bayes %s) | AUC %.3f (Bayes %.3f) | pos %+.3f | mass>0.9 %.2f"
              % (m, mm["A_dec"], mm["A_dec_ci"], mm["acc_dec"], {k: round(v, 3) for k, v in mm["A_by_pi"].items()},
                 {k: round(v, 3) for k, v in mm["A_bayes_by_pi"].items()}, mm["auc"], mm["auc_bayes"], mm["position_dec"], mm["mass_share_gt09"]))
        print("    cues", mm.get("cues"))
        print("    gates", r["gates"], " cert", r.get("cert"))
        for k in ("free_v2", "ext_v2", "arm_dstated", "arm_reason"):
            if k in r:
                print("   ", k, r[k])
json.dump(out, open(os.path.join(RES, "analysis_order_v2.json"), "w"), indent=1)
