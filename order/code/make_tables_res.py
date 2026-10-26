# -*- coding: utf-8 -*-
"""ORDER result tables and manuscript fills (revision after the simulated review).

One scoring protocol per table: the forced-choice parts are scored with the assistant turn prefilled with
"Answer:" for every model (analysis_order_v2.json, block "main"); the unprefixed scoring is a robustness check.
Reads analysis_order_v2.json, reference_bounds.json, misspec_stress.json and realdata_order_summary.json.
python make_tables_res.py <results_dir> <tables_dir> <manuscript_dir>
"""
import json
import os
import re
import sys

RES, TAB, MAN = sys.argv[1:4]
NL = chr(10)
A2 = json.load(open(os.path.join(RES, "analysis_order_v2.json")))
REF = json.load(open(os.path.join(RES, "reference_bounds.json")))
ORDER = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "granite3.3-8b", "olmo2-7b", "qwen2.5-14b"]
LAB = {"qwen2.5-7b": "Qwen2.5-7B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B",
       "granite3.3-8b": "Granite-3.3-8B", "olmo2-7b": "OLMo-2-7B", "qwen2.5-14b": "Qwen2.5-14B"}
models = [m for m in ORDER if m in A2]
core = [m for m in models if m != "qwen2.5-14b"]


def s(x, d=3):
    return ("%+." + str(d) + "f") % x if x is not None and x == x else "n/a"


def m_(x):
    return re.sub(r"(^|[\s\[(|/])-(?=\d)", r"\1−", x)


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


CAPS = {'arbitration': ('Arbitration between timestamps and maintenance history', 'Main part, 1,728 probes.'), 'arms': ('Arbitration when the model is helped', ''), 'cert': ('Certification under a hard offset bound', '240 probes; prefilled scoring.'), 'cues': ("Cue weights in each model's choice", 'Fractional logit of the probability of naming event A on the history log-odds, the sign of the timestamp difference, the size of the timestamp evidence $gD/σ_{θ}^{2}$, the line position (±1) and an intercept.'), 'relocate': ('Extraction fidelity and free answers', 'Stratified random subsets.'), 'misspec': ('The attributor under a misdeclared clock model', ''), 'sync': ('Counterfactual timestamp-only error on measured event delays', 'By oscillator and synchronisation interval; NTP residual 10 ms, uniform prior.')}


def write(name, rows):
    if name in CAPS:
        rows = shorten(rows, *CAPS[name])
    open(os.path.join(TAB, "table_%s.md" % name), "w", encoding="utf-8").write(NL.join(rows) + NL)


def ref(key):
    return REF[next(k for k in REF if k.startswith(key))]


# ---------------- Table 2: arbitration
rows = ["Table: Arbitration between timestamps and maintenance history on the main part (1,728 probes). Decisions are the most probable option, with the assistant turn prefilled with \"Answer:\". $A$ with a 95% bootstrap interval; within $π$: $A$ computed separately for the history priors 0.7 and 0.9; AUC: how well the probability of following the timestamps separates the items on which they are right from those on which the history is right.",
        "| Model | Letter mass | Accuracy | $A$ [95% CI] | $A$ within $π$ (0.7 / 0.9) | AUC | Position reliance | Timestamp gate | History gate |",
        "|---|---|---|---|---|---|---|---|---|"]
for m in models:
    a, g = A2[m]["main"], A2[m]["gates"]
    rows.append(m_("| %s | %.2f | %.3f | %s [%s, %s] | %s / %s | %.2f | %s | %.2f | %.2f |" % (
        LAB[m], a["mass_median"], a["acc_dec"], s(a["A_dec"]), s(a["A_dec_ci"][0]), s(a["A_dec_ci"][1]),
        s(a["A_by_pi"]["0.7"]), s(a["A_by_pi"]["0.9"]), a["auc"], s(a["position_dec"]), g["gate_ts"]["dec"], g["gate_mech"]["dec"])))
b = ref("Bayes attributor")
c = ref("calibrated Bayes")
g0 = A2[core[0]]["gates"]
rows.append(m_("| Bayes attributor | n/a | %.3f | %s | %s / %s | %.2f | %s | %.2f | %.2f |" % (
    b["realised"]["acc"], s(b["soft"]["A"]), s(b["A_within_pi"][0]), s(b["A_within_pi"][1]), A2[core[0]]["main"]["auc_bayes"],
    s(b["position"]), g0["gate_ts"]["bayes_dec"], g0["gate_mech"]["bayes_dec"])))
rows.append(m_("| Calibrated analyst | n/a | %.3f | %s | %s / %s | %.2f | %s | %.2f | %.2f |" % (
    c["realised"]["acc"], s(c["soft"]["A"]), s(c["A_within_pi"][0]), s(c["A_within_pi"][1]), A2[core[0]]["main"]["auc_bayes"],
    s(c["position"]), g0["gate_ts"]["bayes_calibrated"], g0["gate_mech"]["bayes_calibrated"])))
rows.append("Note: Letter mass: median probability on the option letters. Position reliance: probability of naming the event whose line comes first, minus 1/2 (line order is declared uninformative). Timestamp gate: share of decisions that follow the earlier timestamp when the history is uninformative (432 probes). History gate: share that follow the history when the clocks are declared unsynchronised (96 probes). The calibrated analyst answers each event with its posterior probability; its gate columns are expected rates.")
write("arbitration", rows)

# ---------------- Table 3: cue weights
rows = ["Table: Weights of the cues in each model's choice, from a fractional logit of the probability of naming event A on the history log-odds, the sign of the timestamp difference, the size of the timestamp evidence $gD/σ_{θ}^{2}$, the line position (±1) and an intercept. Event A is the event whose mechanism the process note states first.",
        "| Model | History log-odds | Sign of $D$ | Size of the evidence | Line position | Intercept |", "|---|---|---|---|---|---|"]
for m in models:
    cw = A2[m]["main"]["cues"]
    if "error" in cw:
        continue
    cell = lambda k: "%.2f ± %.2f" % (cw[k][0], cw[k][1])
    rows.append(m_("| %s | %s | %s | %s | %s | %s |" % (LAB[m], cell("history"), cell("sign_D"), "%.3f ± %.3f" % tuple(cw["size_llr"]),
                                                     cell("position"), cell("intercept_A"))))
rows.append("| Bayes attributor | 1 | 0 | 1 | 0 | 0 |")
rows.append("Note: ± gives 1.96 robust standard errors.")
write("cues", rows)

# ---------------- Table 4: certification
rows = ["Table: Certification under a hard offset bound $δ$ (240 probes; prefilled scoring). Overclaim: the model asserts an order when $|D| < 2δ$; underclaim: it declines when $|D| > 2δ$; misorder: it asserts the order opposite to the timestamps.",
        "| Model | Accuracy | Overclaim | Underclaim | Misorder |", "|---|---|---|---|---|"]
for m in models:
    cc = A2[m]["cert_px"]
    rows.append("| %s | %.2f | %.2f | %.2f | %.2f |" % (LAB[m], cc["acc_dec"], cc["overclaim_dec"], cc["underclaim_dec"], cc["misorder_dec"]))
rows.append("| Order certifier (Lemma 1) | 1.00 | 0.00 | 0.00 | 0.00 |")
rows.append("| Always \"not established\" | 0.60 | 0.00 | 1.00 | 0.00 |")
rows.append("| Always the timestamp order | 0.40 | 1.00 | 0.00 | 0.00 |")
write("cert", rows)

# ---------------- Table 5: helped arms and scale
rows = ["Table: Does helping the model change the arbitration? D stated: every second main item with the timestamp difference written out after the log (864 probes). Reasoning: every ninth main item and every fifth certification item, answered after step-by-step reasoning within a budget of 512 new tokens and parsed for a final \"Answer: X\"; when the budget ran out first, the answer was forced by appending \"Answer:\" to the model's own reasoning. $A$ within $π$ on the arm's items, for the arm and for the same items in the corpus format.",
        "| Model | Arm | Items | Answered within budget | $A$ within $π$, arm | $A$ within $π$, corpus format | Accuracy, arm | Certification, arm |",
        "|---|---|---|---|---|---|---|---|"]
for m in models:
    a = A2[m]
    first = True
    for k, name in (("arm_dstated", "D stated"), ("arm_reason", "Reasoning")):
        v = a.get(k)
        if not v:
            continue
        n = v.get("n", v.get("n_main"))
        rows.append(m_("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            LAB[m] if first else "", name, n, pct(v.get("self_answered", 1.0)), s(v.get("A_within_pi")), s(v.get("A_within_pi_baseline")),
            ("%.3f" % v["acc_dec"]) if v.get("acc_dec") is not None else "n/a",
            ("%.2f" % v["cert_acc"]) if v.get("cert_acc") is not None else "n/a")))
        first = False
rows.append(m_("| Bayes attributor | | | | %s | | %.3f | 1.00 |" % (s(sum(ref("Bayes attributor")["A_within_pi"]) / 2), ref("Bayes attributor")["realised"]["acc"])))
write("arms", rows)

# ---------------- Table 6: relocation and free text
rows = ["Table: Extraction fidelity and free answers on stratified random subsets. Extraction: share of 120 JSON outputs in which each field is exact; $A$ for the attributor fed with the extracted fields and for the Bayes attributor with the declared fields on the same items, with a paired bootstrap interval for the difference; flips: share of decisions that differ. Free text: 216 incidents answered without options; agreement and Cohen's $κ$ with the forced choice on the same incident.",
        "| Model | Timestamps | Offset SD | Latency | History | $A$ extracted − declared [95% CI] | Flips | Free: one event named | Agreement | $κ$ |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for m in models:
    e, fr = A2[m].get("ext_v2"), A2[m].get("free_v2")
    fa = e["field_acc"] if e else {}
    hist = min(fa.get("history_first") or 0, fa.get("history_percent") or 0) if e else None
    rows.append(m_("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        LAB[m], pct(fa.get("D")), pct(fa.get("offset_sd")), pct(fa.get("latency")), pct(hist),
        ("%s [%s, %s]" % (s(e["A_attributor"] - e["A_bayes_same"]), s(e["A_diff_ci"][0]), s(e["A_diff_ci"][1]))) if e else "pending",
        pct(e["flip_rate"]) if e else "pending", pct(fr["named_exactly_one"]) if fr else "pending",
        pct(fr["agreement"]) if fr and fr["agreement"] is not None else "n/a",
        ("%.2f" % fr["kappa"]) if fr and fr.get("kappa") is not None and fr["kappa"] == fr["kappa"] else "n/a")))
rows.append("Note: Rates in percent. Timestamps: the difference between the two extracted timestamps is exact to the millisecond.")
write("relocate", rows)

# ---------------- Table 7: misspecification stress test of the attributor (no language model)
MS = json.load(open(os.path.join(RES, "misspec_stress.json")))["scenarios"]
rows = ["Table: The attributor when its declared model is wrong. Realised accuracy and arbitration index $A$ over the main grid ($π$ of 0.7 or 0.9, $g/σ_{θ}$ of 0.5, 1 or 2; 200,000 simulated incidents per cell). True Bayes: the optimal rule for the process that actually generated the data.",
        "| Departure from the declared model | Attributor accuracy | Attributor $A$ | True Bayes accuracy | True Bayes $A$ | Earliest timestamp accuracy |",
        "|---|---|---|---|---|---|"]
for name, v in MS.items():
    a, b_, t = v["attributor (declared)"], v["Bayes (true process)"], v["earliest timestamp"]
    nm = name.replace("sigma declared", "offset spread declared").replace("x", "×").replace("e×p", "exp").replace("3 dof", "3 degrees of freedom")
    rows.append(m_("| %s | %.3f | %s | %.3f | %s | %.3f |" % (nm[0].upper() + nm[1:], a["acc"], s(a["A"]), b_["acc"], s(b_["A"]), t["acc"])))
rows.append("Note: The maintenance-history rule has accuracy 0.800 and $A = 0$ in every row; the earliest-timestamp rule has $A = 0$ by construction. Ignoring the prior turns the attributor into the earliest-timestamp rule.")
write("misspec", rows)

# ---------------- Table 8: real systems (counterfactual: measured latencies, simulated clock offsets)
R = json.load(open(os.path.join(RES, "realdata_order_summary.json")))
rows = ["Table: Counterfactual error of a timestamp-only analyst on measured delays between related events, by oscillator and synchronisation interval (NTP residual 10 ms, uniform prior). The delays are measured; the clock offsets are simulated under Definition 1 with drift uniform over the tolerance (200,000 draws per cell). Uncertifiable: share of measured delays below $2δ$ for the hard bound $δ = 3σ_{s} + ρ_{max}T$.",
        "| Delays | Oscillator | 17 min | 1 h | 1 day | 1 week | 30 days | Uncertifiable, 1 day |", "|---|---|---|---|---|---|---|---|"]
for srcn in R["n"]:
    for clock in ("XO +-20 ppm", "TCXO +-2 ppm", "disciplined (+-0.5 ppm residual)"):
        rr = {r["interval"]: r for r in R["rows"] if r["source"] == srcn and r["clock"] == clock and r["sync"].startswith("NTP")}
        if not rr:
            continue
        rows.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f |" % (
            srcn.split(":")[0] + (" (%d pairs)" % R["n"][srcn]) if clock.startswith("XO") else "",
            clock.replace("+-", "±").replace("disciplined (±0.5 ppm residual)", "disciplined, ±0.5 ppm"),
            rr["17 min (NTP maxpoll)"]["bayes_error"], rr["1 h"]["bayes_error"], rr["1 day"]["bayes_error"],
            rr["1 week"]["bayes_error"], rr["30 days"]["bayes_error"], rr["1 day"]["share_uncertifiable"]))
mse = max(r["mc_se"] for r in R["rows"])
rows.append("Note: 17 min is the default maximum NTP polling interval ($2^{10}$ s); a disciplined clock keeps a residual frequency error after NTP has steered it. Largest Monte Carlo standard error in the table: %.4f." % mse)
write("sync", rows)

# ---------------- fills
NUMW = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
F = {"N_MODELS": NUMW[len(core)]}
A_ = [A2[m]["main"]["A_dec"] for m in core]
F["A_MIN"], F["A_MAX"] = m_(s(min(A_))), m_(s(max(A_)))
W = [A2[m]["main"]["A_by_pi"][k] for m in core for k in ("0.7", "0.9")]
F["AW_MIN"], F["AW_MAX"] = m_(s(min(W))), m_(s(max(W)))
acc = [A2[m]["main"]["acc_dec"] for m in core]
F["ACC_MIN"], F["ACC_MAX"] = "%.3f" % min(acc), "%.3f" % max(acc)
auc = [A2[m]["main"]["auc"] for m in core]
F["AUC_MIN"], F["AUC_MAX"] = "%.2f" % min(auc), "%.2f" % max(auc)
cw = {m: A2[m]["main"]["cues"] for m in core if "error" not in A2[m]["main"]["cues"]}
F["BPR_MIN"], F["BPR_MAX"] = "%.2f" % min(c["history"][0] for c in cw.values()), "%.2f" % max(c["history"][0] for c in cw.values())
F["BSIGN_MIN"], F["BSIGN_MAX"] = m_("%.2f" % min(c["sign_D"][0] for c in cw.values())), m_("%.2f" % max(c["sign_D"][0] for c in cw.values()))
F["BSIZE_MAXABS"] = "%.3f" % max(abs(c["size_llr"][0]) for c in cw.values())
F["BPOS_MIN"], F["BPOS_MAX"] = m_("%.2f" % min(c["position"][0] for c in cw.values())), m_("%.2f" % max(c["position"][0] for c in cw.values()))
gts = [A2[m]["gates"]["gate_ts"]["dec"] for m in core]
F["GTS_MIN"], F["GTS_MAX"] = pct(min(gts)), pct(max(gts))
gh = [A2[m]["gates"]["gate_mech"]["dec"] for m in core]
F["GHIST_MIN"], F["GHIST_MAX"] = pct(min(gh)), pct(max(gh))
pos = [A2[m]["main"]["position_dec"] for m in core]
F["POS_MIN"], F["POS_MAX"] = m_(s(min(pos))), m_(s(max(pos)))
F["N_POS"] = NUMW[sum(abs(p) > 0.05 for p in pos)]
ca = [A2[m]["cert_px"]["acc_dec"] for m in core]
F["CERT_MIN"], F["CERT_MAX"] = "%.2f" % min(ca), "%.2f" % max(ca)
F["N_SIGN"] = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}[sum(abs(c["sign_D"][0]) > c["sign_D"][1] for c in cw.values())]
mo = {m: A2[m]["cert_px"]["misorder_dec"] for m in core}
big = [LAB[m] for m in core if mo[m] >= 0.15]
F["MISORDER_TEXT"] = ("%s misorder%s %.0f%% to %.0f%% of the items, asserting the order opposite to the timestamps." % (
    " and ".join(big) if len(big) <= 2 else ", ".join(big[:-1]) + " and " + big[-1], "" if len(big) > 1 else "s",
    100 * min(mo[m] for m in core if mo[m] >= 0.15), 100 * max(mo.values()))) if big else "No model misorders more than 15% of the items."
fa = [A2[m]["main"]["follow_hist_when_fav_A"] for m in core]
fb = [A2[m]["main"]["follow_hist_when_fav_B"] for m in core]
F["FAVA_MIN"], F["FAVA_MAX"], F["FAVB_MIN"], F["FAVB_MAX"] = pct(min(fa)), pct(max(fa)), pct(min(fb)), pct(max(fb))
# revision arms, extraction and free text (ranges over the core models that have them)
def rng_(vals, f=s):
    vals = [v for v in vals if v is not None and v == v]
    return (m_(f(min(vals))), m_(f(max(vals)))) if vals else ("n/a", "n/a")


ds = {m: A2[m]["arm_dstated"] for m in core if "arm_dstated" in A2[m]}
F["DST_AW_MIN"], F["DST_AW_MAX"] = rng_([v["A_within_pi"] for v in ds.values()])
F["DST_BASE_MIN"], F["DST_BASE_MAX"] = rng_([v["A_within_pi_baseline"] for v in ds.values()])
F["DST_N"] = NUMW[len(ds)]
rs = {m: A2[m]["arm_reason"] for m in core if "arm_reason" in A2[m]}
F["RS_AW_MIN"], F["RS_AW_MAX"] = rng_([v.get("A_within_pi") for v in rs.values()])
F["RS_SELF_MIN"], F["RS_SELF_MAX"] = rng_([v.get("self_answered") for v in rs.values()], pct)
F["RS_CERT_MIN"], F["RS_CERT_MAX"] = rng_([v.get("cert_acc") for v in rs.values()], lambda x: "%.2f" % x)
F["RS_ACC_MIN"], F["RS_ACC_MAX"] = rng_([v.get("acc_dec") for v in rs.values()], lambda x: "%.3f" % x)
F["RS_N"] = NUMW[len(rs)]
ex_all = {m: A2[m]["ext_v2"] for m in core if "ext_v2" in A2[m]}
ex = {m: v for m, v in ex_all.items() if v["field_acc"]["D"] >= 0.5}          # models that read the timestamps
poor = {m: v for m, v in ex_all.items() if m not in ex}
F["EXT_GOOD_N"] = NUMW[len(ex)]
F["EXT_POOR_TEXT"] = " ".join("%s copies the timestamps correctly in only %s%% of the outputs and the latency in %s%%, and the attributor fed with its fields falls to an index of %s, no better than the model's own choices (%s): relocation is only as good as the reader." % (
    LAB[m], pct(v["field_acc"]["D"]), pct(v["field_acc"]["latency"]), m_(s(v["A_attributor"])), m_(s(v["A_model_same"]))) for m, v in poor.items())
F["EXT_N"] = NUMW[len(ex)]
F["EXT_TS_MIN"], F["EXT_TS_MAX"] = rng_([v["field_acc"]["D"] for v in ex.values()], pct)
F["EXT_LAT_MIN"], F["EXT_LAT_MAX"] = rng_([v["field_acc"]["latency"] for v in ex.values()], pct)
F["EXT_HIST_MIN"], F["EXT_HIST_MAX"] = rng_([min(v["field_acc"]["history_first"], v["field_acc"]["history_percent"]) for v in ex.values()], pct)
F["EXT_A_MIN"], F["EXT_A_MAX"] = rng_([v["A_attributor"] for v in ex.values()])
F["EXT_LOSS_MIN"], F["EXT_LOSS_MAX"] = rng_([v["A_bayes_same"] - v["A_attributor"] for v in ex.values()], lambda x: "%.3f" % x)
F["EXT_BAYES"] = m_(s(list(ex.values())[0]["A_bayes_same"])) if ex else "n/a"
F["EXT_OWN_MIN"], F["EXT_OWN_MAX"] = rng_([v.get("A_model_same") for v in ex.values()])
F["EXT_FLIP_MIN"], F["EXT_FLIP_MAX"] = rng_([v["flip_rate"] for v in ex.values()], pct)
F["EXT_CI_ALL"] = "excludes zero for every model" if ex and all(v["A_diff_ci"][1] < 0 for v in ex.values()) else "excludes zero for some models"
fr = {m: A2[m]["free_v2"] for m in core if "free_v2" in A2[m]}
F["FREE_NAMED_MIN"], F["FREE_NAMED_MAX"] = rng_([v["named_exactly_one"] for v in fr.values()], pct)
F["FREE_AGREE_MIN"], F["FREE_AGREE_MAX"] = rng_([v["agreement"] for v in fr.values()], pct)
F["FREE_KAPPA_MIN"], F["FREE_KAPPA_MAX"] = rng_([v["kappa"] for v in fr.values()], lambda x: "%.2f" % x)
F["FREE_AFREE_MIN"], F["FREE_AFREE_MAX"] = rng_([v.get("A_free") for v in fr.values()])
F["EXT_KEEP_MIN"], F["EXT_KEEP_MAX"] = rng_([v["A_attributor"] / v["A_bayes_same"] for v in ex.values()], pct)
F["EXT_SD_MIN"] = pct(min(v["field_acc"]["offset_sd"] for v in ex.values())) if ex else "n/a"
# scale check
q = A2.get("qwen2.5-14b")
if q:
    aw = max(abs(q["main"]["A_by_pi"]["0.7"]), abs(q["main"]["A_by_pi"]["0.9"]))
    F["Q14_PHRASE"] = ("like the smaller models" if aw < 0.1 and q["main"]["auc"] < 0.6 else
                       ("somewhat better, but far from the Bayes attributor" if aw < 0.2 else "much closer to the Bayes attributor"))
    F["Q14_A"] = m_(s(q["main"]["A_dec"]))
    F["Q14_AW"] = "%s / %s" % (m_(s(q["main"]["A_by_pi"]["0.7"])), m_(s(q["main"]["A_by_pi"]["0.9"])))
    F["Q14_ACC"] = "%.3f" % q["main"]["acc_dec"]
    F["Q14_AUC"] = "%.2f" % q["main"]["auc"]
    F["Q14_SIZE"] = m_("%.3f" % q["main"]["cues"]["size_llr"][0]) if "error" not in q["main"]["cues"] else "n/a"
    F["Q14_SIGN"] = m_("%.2f" % q["main"]["cues"]["sign_D"][0]) if "error" not in q["main"]["cues"] else "n/a"
    F["Q14_CERT"] = "%.2f" % q["cert_px"]["acc_dec"]
    F["Q14_POS"] = m_("%.2f" % q["main"]["cues"]["position"][0]) if "error" not in q["main"]["cues"] else "n/a"
    F["Q14_DST"] = m_(s(q.get("arm_dstated", {}).get("A_within_pi")))
    F["Q14_RS"] = m_(s(q.get("arm_reason", {}).get("A_within_pi")))
if "Q14_PHRASE" in F:
    F["Q14_SENT"] = "The scale check behaves %s. Qwen2.5-14B reaches an index of %s (%s within the two priors) and an AUC of %s; it gives the sign of the timestamp difference a weight of %s but its size a weight of %s, follows the line order (weight %s), and certifies with an accuracy of %s, that of always abstaining. Stating the difference or reasoning step by step leaves its index within each prior at %s and %s (Table 5), and it extracts every field almost exactly." % (F["Q14_PHRASE"], F["Q14_A"], F["Q14_AW"], F["Q14_AUC"], F["Q14_SIGN"], F["Q14_SIZE"], F["Q14_POS"], F["Q14_CERT"], F["Q14_DST"], F["Q14_RS"])
else:
    F["Q14_PHRASE"] = '`[to be completed: Qwen2.5-14B scale check, run still in progress]`'
    F["Q14_SENT"] = "`[To be completed: Qwen2.5-14B scale check, run still in progress.]`"
json.dump(F, open(os.path.join(MAN, "fills_auto.json"), "w"), indent=1)
print(json.dumps(F, indent=1))
