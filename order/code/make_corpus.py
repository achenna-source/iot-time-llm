# -*- coding: utf-8 -*-
"""ORDER probe corpus: which of two IoT events started an incident, when device clocks are skewed?

Generative process (declared in every prompt): the true first event is drawn from the maintenance-history
prior pi (favoured direction counterbalanced between the textbook and the interlock direction); the second
event follows after the declared latency g; each device clock adds an independent offset N(0, sigma^2).
D = tau_B - tau_A. Posterior log-odds(A first) = logit(prior_A) + g D / sigma^2.
Parts
  main      forced choice, pi in {0.7, 0.9}, gamma = g/sigma in {0.5, 1, 2}   (arbitration index)
  gate_ts   forced choice, no maintenance prior (pi = 0.5)                   (timestamp competence)
  gate_mech forced choice, clocks declared unsynchronised by minutes         (mechanism competence)
  cert      bounded skew |offset| <= delta, three options incl. "not established" (certification)
Deterministic given SEED.
"""
import json
import math
import random
import sys
from datetime import datetime, timedelta

SEED = 20261026
OUT = sys.argv[1] if len(sys.argv) > 1 else "probes.jsonl"

# site, (dev_a, event_a), (dev_b, event_b), A->B mechanism, B->A mechanism, alarm, distractor devices
SCEN = [
    ("a village water network", ("bp2", "booster pump BP2 stopped"), ("fs2", "flow switch FS2 reports no flow"),
     "a stopped booster pump cuts the flow", "a no-flow condition trips the pump on its dry-run protection",
     "LOW DELIVERY PRESSURE ZONE 2", ["lvl_t1", "gw_01", "ph_02"]),
    ("a commercial greenhouse", ("ip1", "irrigation pump IP1 stopped"), ("pt_ir1", "header pressure low on PT-IR1"),
     "a stopped pump lets the header pressure fall", "a low-pressure switch stops the pump to protect against a burst line",
     "IRRIGATION ZONE 4 NOT WATERED", ["sm_04", "gw_02", "rh_01"]),
    ("an office building", ("sf1", "supply fan SF1 stopped"), ("dp1", "duct static pressure low on DP1"),
     "a stopped fan lets the duct pressure fall", "the loss-of-airflow interlock stops the fan when duct pressure collapses",
     "AHU-1 SUPPLY AIR FAULT", ["zt_301", "bms_gw", "co2_302"]),
    ("a distribution substation", ("cb4", "breaker CB4 opened"), ("vm_f4", "voltage sag on feeder F4"),
     "an opening breaker drops the feeder voltage", "the undervoltage relay opens the breaker during a sag",
     "FEEDER F4 OUTAGE", ["tx_temp", "rtu_02", "cb5"]),
    ("a solar microgrid", ("inv3", "PV inverter INV3 disconnected"), ("fm_b1", "frequency deviation on bus B1"),
     "losing an inverter shifts the frequency of the weak bus", "anti-islanding protection disconnects the inverter on a frequency deviation",
     "MICROGRID LOAD SHEDDING STAGE 1", ["bat_soc", "inv2", "met_01"]),
    ("a campus video network", ("sw_p12", "switch port 12 link down"), ("cam07", "camera CAM-07 offline"),
     "a port going down takes its PoE camera offline", "a short in the camera's PoE circuit makes the switch shut the port",
     "PERIMETER CAMERA COVERAGE GAP", ["cam06", "nvr_01", "sw_p11"]),
    ("a packaging plant", ("c1", "conveyor C1 stopped"), ("j1", "jam sensor J1 blocked"),
     "a stopped conveyor lets cartons pile up at J1", "a jam at J1 stops the conveyor",
     "LINE 1 THROUGHPUT ZERO", ["c2", "plc_01", "scale_01"]),
    ("a machining workshop", ("m3", "spindle M3 stopped"), ("cp3", "coolant pressure low on CP3"),
     "the coolant pump is slaved to the spindle and stops with it", "the low-coolant interlock stops the spindle",
     "CELL 3 PART REJECTED", ["m2", "edge_gw", "vib_03"]),
    ("a data centre", ("crac2", "cooling unit CRAC-2 stopped"), ("ft_cw2", "chilled-water flow low on FT-CW2"),
     "a stopped unit closes its chilled-water valve", "a low chilled-water flow trips the cooling unit",
     "ROW B INLET TEMPERATURE HIGH", ["crac1", "pdu_b3", "th_b2"]),
    ("a battery storage site", ("br2", "battery rack BR2 isolated"), ("dc_bus", "DC bus voltage dip"),
     "isolating a rack dips the DC bus", "the battery management system isolates a rack during a bus dip",
     "STORAGE OUTPUT DERATED", ["br1", "hvac_c", "pcs_01"]),
    ("a cold-storage warehouse", ("cp1", "compressor CP1 tripped"), ("op1", "oil pressure low on OP1"),
     "the oil pump is driven by the compressor and loses pressure when it stops", "the low-oil-pressure switch trips the compressor",
     "FREEZER ROOM 2 TEMPERATURE RISING", ["cp2", "door_r2", "gw_cs"]),
    ("a smart irrigation scheme", ("sv4", "solenoid valve SV4 closed"), ("fm4", "flow meter FM4 reads zero flow"),
     "a closed valve stops the flow through the meter", "the controller closes the valve when the meter reports a dry line",
     "PLOT 4 IRRIGATION SKIPPED", ["sm_12", "lora_gw", "rain_01"]),
]

SIGMAS = [0.5, 1.0, 2.0]
GAMMAS = [0.5, 1.0, 2.0]
PIS = [0.7, 0.9]


def ts(t):
    return t.strftime("%Y-%m-%d %H:%M:%S.") + "%03d" % (t.microsecond // 1000)


def fmt_s(x):
    return ("%.2f" % x).rstrip("0").rstrip(".")


def build():
    rnd = random.Random(SEED)
    rows = []

    def item(part, si, sigma, gamma, pi, fav_fwd, list_a_first, opt_a_first, D_override=None, delta=None):
        site, (da, ea), (db, eb), mf, mr, alarm, distr = SCEN[si]
        g = gamma * sigma if gamma else 2.0
        # truth: A first with probability prior_A
        prior_A = (pi if fav_fwd else 1 - pi) if pi is not None else 0.5
        a_first = rnd.random() < prior_A
        t0 = datetime(2026, 1, 1) + timedelta(seconds=rnd.randint(0, 300 * 86400), microseconds=rnd.randint(0, 999) * 1000)
        tA, tB = (t0, t0 + timedelta(seconds=g)) if a_first else (t0 + timedelta(seconds=g), t0)
        if D_override is not None:                       # certification part: D fixed, truth from bounded skew
            D = D_override
            tauA = t0
            tauB = t0 + timedelta(seconds=D)
            a_first = None
        else:
            s_off = sigma if part != "gate_mech" else 240.0
            thA, thB = rnd.gauss(0, s_off), rnd.gauss(0, s_off)
            tauA = tA + timedelta(seconds=thA)
            tauB = tB + timedelta(seconds=thB)
            D = (tauB - tauA).total_seconds()
        t_alarm = max(tA, tB) + timedelta(seconds=rnd.uniform(3, 8))
        lines_ev = [(tauA, da, ea, "A"), (tauB, db, eb, "B")]
        if not list_a_first:
            lines_ev = lines_ev[::-1]
        dl = []
        for dv in distr:
            tt = t0 + timedelta(seconds=rnd.uniform(-60, 60))
            msg = rnd.choice(["heartbeat ok", "battery 81%", "telemetry frame received", "status nominal"])
            dl.append((tt, dv, msg, None))
        body_lines = dl[:1] + lines_ev[:1] + dl[1:2] + lines_ev[1:] + dl[2:]
        body_lines.append((t_alarm, "scada", "ALARM: " + alarm, None))
        log = "\n".join("%s  [%s]  %s" % (ts(t), dv, msg) for t, dv, msg, _ in body_lines)
        if part == "cert":
            clock = ("Clock note: every device clock is disciplined by GNSS-referenced NTP and is guaranteed to stay "
                     "within %s s of true time (bounded offset)." % fmt_s(delta))
        elif part == "gate_mech":
            clock = ("Clock note: these two devices are not synchronised; their clocks are known to be several minutes "
                     "apart, so their timestamps cannot be compared with each other.")
        else:
            clock = ("Clock note: device clocks are synchronised by NTP once a day; between synchronisations each clock "
                     "drifts, so every device timestamp carries an independent offset with a standard deviation of "
                     "about %s s." % fmt_s(sigma))
        proc = ("Process note: either event can cause the other: %s, and %s. Whichever happens first, the "
                "other follows about %s s later.") % (mf, mr, fmt_s(g)) if part != "cert" else (
            "Process note: either event can cause the other (%s; %s)." % (mf, mr))
        if pi is None:
            hist = "Maintenance history: in past incidents of this kind, each of the two events came first about equally often."
        else:
            fe = ea if fav_fwd else eb
            hist = ("Maintenance history: in past incidents of this kind at this site, the event \"%s\" came first in "
                    "%d%% of cases." % (fe, round(100 * pi)))
        if part == "cert":
            hist = ""
        intro = ("You are assisting with root-cause analysis for %s. Below is an excerpt of the incident log collected by "
                 "the gateway. Each line carries the timestamp written by the device that produced it. Lines appear in "
                 "the order the collector received them; devices upload in batches, so line order does not reflect "
                 "when events happened." % site)
        if part == "cert":
            opts_roles = [("A_first", "\"%s\" happened before \"%s\"" % (ea, eb)),
                          ("B_first", "\"%s\" happened before \"%s\"" % (eb, ea)),
                          ("undetermined", "The log does not establish which of the two happened first")]
            rot = opt_a_first
            opts_roles = opts_roles[rot:] + opts_roles[:rot]
            q = "Question: Does this log establish which of the two events happened first?"
        else:
            opts_roles = [("A", ea), ("B", eb)] if opt_a_first else [("B", eb), ("A", ea)]
            q = "Question: Which event started this incident?"
        letters = "ABC"
        opts = "\n".join("%s) %s" % (letters[i], txt) for i, (_, txt) in enumerate(opts_roles))
        notes = "\n".join(x for x in (clock, proc, hist) if x)
        prompt = "%s\n\n%s\n\nLog:\n%s\n\n%s\n%s\nAnswer with a single letter." % (intro, notes, log, q, opts)
        if part == "cert":
            post_A = None
            correct = "undetermined" if abs(D) <= 2 * delta else ("A_first" if D > 0 else "B_first")
        else:
            sig_eff = sigma if part != "gate_mech" else 240.0
            lo = math.log(prior_A / (1 - prior_A)) + g * D / sig_eff ** 2
            post_A = 1 / (1 + math.exp(-lo))
            correct = "A" if a_first else "B"
        rows.append({"id": "ord-%05d" % len(rows), "part": part, "scen": si, "sigma": sigma, "gamma": gamma, "g": g,
                     "pi": pi, "fav_fwd": fav_fwd, "prior_A": prior_A, "a_first": a_first, "D": D, "delta": delta,
                     "list_a_first": list_a_first, "roles": {letters[i]: r for i, (r, _) in enumerate(opts_roles)},
                     "post_A": post_A, "correct": correct, "prompt": prompt})

    for si in range(len(SCEN)):
        for sigma in SIGMAS:
            for gamma in GAMMAS:
                for pi in PIS:
                    for fav in (True, False):
                        for la in (True, False):
                            for _rep in range(2):
                                item("main", si, sigma, gamma, pi, fav, la, rnd.random() < 0.5)
                for la in (True, False):
                    for oa in (True, False):
                        item("gate_ts", si, sigma, gamma, None, True, la, oa)
        for pi in PIS:
            for fav in (True, False):
                for la in (True, False):
                    item("gate_mech", si, 1.0, 1.0, pi, fav, la, rnd.random() < 0.5)
        for ratio in (0.5, 1.0, 1.5, 2.5, 4.0):
            for sign in (1, -1):
                for la in (True, False):
                    delta = rnd.choice([0.25, 0.5, 1.0])
                    item("cert", si, None, None, None, True, la, rnd.randrange(3), D_override=sign * ratio * delta,
                         delta=delta)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    rows = build()
    from collections import Counter
    print(len(rows), Counter(r["part"] for r in rows))
    print(rows[0]["prompt"])
