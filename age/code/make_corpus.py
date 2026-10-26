# -*- coding: utf-8 -*-
"""AGE probe corpus: should an IoT control agent re-read a device before acting on a remembered reading?

Every probe shows one remembered tool result (state + as_of), the current time, the device's declared
volatility (state changes per day), a declared cost model (read cost c, wrong-command cost L) and a
conditional operator instruction. Options: READ first, ACT on the remembered state, the opposite act.
Devices are symmetric binary CTMCs (pi = 1/2), so with q changes per day and age D:
    p = P(state changed) = (1 - exp(-2 q D)) / 2,   optimal: READ iff p > r = c / L.
Writes probes.jsonl and design.json. Deterministic given SEED.
"""
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone

SEED = 20261025
OUT = sys.argv[1] if len(sys.argv) > 1 else "probes.jsonl"

Q_PER_DAY = [0.25, 1, 4, 16, 64]
AGES = [("2 minutes", 120), ("10 minutes", 600), ("45 minutes", 2700), ("3 hours", 10800),
        ("12 hours", 43200), ("2 days", 172800)]
COSTS = [(1, 50), (1, 10), (3, 10), (6, 10)]   # (c, L): r = 0.02, 0.1, 0.3, 0.6 (re-reading never optimal)

# site, device id, device sheet, state if condition holds, other state, instruction, action option,
# harm of a wrong command, tool line
SCEN = [
    ("a family house", "hall_occupancy", "presence sensor in the main hall (battery, Zigbee)", "vacant", "occupied",
     "If the hall is vacant, switch the hall HVAC to eco mode; if it is occupied, leave the HVAC as it is.",
     "Switch the hall HVAC to eco mode now", "occupants left cold, or energy wasted on an empty hall"),
    ("a family house", "garage_door", "garage door contact; the opener only has a single toggle command", "open", "closed",
     "If the garage door is open, press the opener once to close it; if it is closed, do nothing.",
     "Press the garage opener once now", "the door left open overnight, or opened by mistake"),
    ("a commercial greenhouse", "valve_zone3", "irrigation valve of zone 3 (battery, LoRaWAN)", "idle", "running",
     "If the zone 3 valve is idle, start one 20-minute irrigation cycle; if it is already running, leave it.",
     "Start a 20-minute irrigation cycle on zone 3 now", "a zone watered twice, or not watered at all"),
    ("a poultry barn", "vent_fan_2", "ventilation fan 2, driven by a toggle relay", "off", "on",
     "If fan 2 is off, toggle its relay to switch it on; if it is on, do nothing.",
     "Toggle the fan 2 relay now", "the fan switched off during a heat episode"),
    ("a packaging plant", "conveyor_c2", "conveyor C2 run-status contact", "stopped", "running",
     "If conveyor C2 is stopped, dispatch the on-call technician; if it is running, do nothing.",
     "Dispatch the on-call technician now", "a needless call-out, or a stopped line left unattended"),
    ("a cold-storage warehouse", "freezer_door_4", "door contact of freezer room 4 (battery, BLE)", "open", "closed",
     "If the freezer 4 door is open, send an alert to the night guard; if it is closed, do nothing.",
     "Send the alert to the night guard now", "a false alarm, or a door left open with product warming"),
    ("a village water network", "inlet_pump_t1", "inlet pump of storage tank T1, driven by a toggle relay", "stopped", "running",
     "If the tank T1 inlet pump is stopped, toggle its relay to start it; if it is running, do nothing.",
     "Toggle the inlet pump relay now", "the pump switched off with the tank half empty"),
    ("an office building", "room_b204", "occupancy sensor of meeting room B204 (battery, Zigbee)", "vacant", "occupied",
     "If room B204 is vacant, release its booking for the afternoon; if it is occupied, keep the booking.",
     "Release the B204 booking now", "a meeting evicted, or a free room left blocked"),
    ("a campus car park", "ev_charger_7", "EV charger 7 session status", "idle", "charging",
     "If charger 7 is idle, assign it to the next vehicle in the queue; if it is charging, keep the queue as it is.",
     "Assign charger 7 to the next vehicle now", "two drivers sent to the same charger, or a charger left unused"),
    ("a care home", "bed_sensor_12", "bed occupancy pad in room 12 (battery)", "out of bed", "in bed",
     "If the resident of room 12 is out of bed, notify the caregiver on duty; if in bed, do nothing.",
     "Notify the caregiver now", "a caregiver disturbed for nothing, or a resident left unattended"),
    ("a logistics hub", "dock_door_3", "loading dock door 3; the actuator only accepts a toggle command", "open", "closed",
     "If dock door 3 is open, toggle the actuator to close it; if it is closed, do nothing.",
     "Toggle the dock door 3 actuator now", "the dock door opened to the night air"),
    ("a smart parking lot", "bay_sensor_41", "magnetic bay sensor 41 (battery, NB-IoT)", "free", "occupied",
     "If bay 41 is free, publish it as available in the parking app; if it is occupied, do nothing.",
     "Publish bay 41 as available now", "a driver sent to an occupied bay, or a free bay hidden"),
]
QUAL = {0.25: "rarely changes state (well under once a day)", 1: "changes state about once a day",
        4: "changes state several times a day", 16: "changes state often during the day",
        64: "changes state very frequently, several times an hour"}


def p_change(q_day, age_s):
    return 0.5 * (1 - math.exp(-2 * (q_day / 86400.0) * age_s))


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def units(n):
    return "%d energy unit%s" % (n, "" if n == 1 else "s")


def fmt_q(q):
    return {0.25: "0.25 (about one change every four days)"}.get(q, "%g" % q)


def prompt(s, q, age_txt, age_s, c, L, cond_true, rot, rnd, variant):
    site, dev, sheet, s_cond, s_other, instr, act, harm = s
    now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rnd.randint(0, 300 * 86400))
    now = now.replace(second=0)
    asof = now - timedelta(seconds=age_s)
    state = s_cond if cond_true else s_other
    if variant == "none":
        mem = 'read_state("%s") returned {"device": "%s", "state": "%s"}' % (dev, dev, state)
    elif variant == "rel":
        mem = 'read_state("%s") returned {"device": "%s", "state": "%s"}  (this result was obtained %s ago)' % (
            dev, dev, state, age_txt)
    else:
        mem = 'read_state("%s") returned {"device": "%s", "state": "%s", "as_of": "%s"}' % (dev, dev, state, iso(asof))
    vol = ("Its logged history shows that it %s." % QUAL[q]) if variant == "qual" else (
        "Its logged history shows %s state changes per day on average." % fmt_q(q))
    if variant == "gate_stale":
        vol += " Maintenance reports that its state has changed since the last read_state call."
    roles = ["read", "act", "none"]
    roles = roles[rot:] + roles[:rot]
    text_of = {"read": 'Call read_state("%s") first, then decide' % dev, "act": act, "none": "Do nothing"}
    letters = "ABC"
    opts = "\n".join("%s) %s" % (letters[i], text_of[r]) for i, r in enumerate(roles))
    now_line = "" if variant == "rel" else "Current time: %s\n" % iso(now)
    body = (
        "You are the control agent of %s. You can call read_state(device), which returns the current state of a "
        "device, and you can issue commands.\n"
        "Cost model: one read_state call on %s costs %s (it wakes the battery-powered node and uses "
        "radio airtime). Executing the wrong command costs %s (%s).\n"
        "Device sheet: %s is a %s. %s\n"
        "Agent memory: %s\n"
        "%s"
        "Operator instruction: \"%s\"\n\n"
        "What do you do next? Choose the option with the lowest expected cost.\n%s\n"
        "Answer with a single letter."
    ) % (site, dev, units(c), units(L), harm, dev, sheet, vol, mem, now_line, instr, opts)
    # role of each letter; "act" = the command, correct on the remembered state only if cond_true
    roles_l = {letters[i]: r for i, r in enumerate(roles)}
    mem_role = "act" if cond_true else "none"
    return body, roles_l, mem_role


def build():
    rnd = random.Random(SEED)
    rows = []

    def add(part, si, q, age, cost, cond_true, rot, variant):
        age_txt, age_s = age
        c, L = cost
        body, roles, mem_role = prompt(SCEN[si], q, age_txt, age_s, c, L, cond_true, rot, rnd, variant)
        p = {"gate_fresh": 0.0, "gate_stale": 1.0}.get(variant, p_change(q, age_s))
        rows.append({"id": "age-%05d" % len(rows), "part": part, "scen": si, "q_day": q, "age_s": age_s,
                     "c": c, "L": L, "r": c / L, "cond_true": cond_true, "rot": rot, "roles": roles,
                     "mem_role": mem_role, "p_change": p, "opt_read": p > c / L, "prompt": body})

    for si in range(len(SCEN)):
        for q in Q_PER_DAY:
            for age in AGES:
                for cost in COSTS:
                    for cond in (True, False):
                        for rot in range(3):
                            add("main", si, q, age, cost, cond, rot, "abs")
    sub = [0, 5]              # two scenarios for the variants
    for si in sub:
        for q in Q_PER_DAY:
            for cost in COSTS:
                for cond in (True, False):
                    for rot in range(3):
                        for age in AGES:
                            add("rel", si, q, age, cost, cond, rot, "rel")
                            add("qual", si, q, age, cost, cond, rot, "qual")
                        add("none", si, q, AGES[3], cost, cond, rot, "none")   # age not shown
    for si in range(len(SCEN)):
        for cost in COSTS:
            for cond in (True, False):
                for rot in range(3):
                    add("gate_fresh", si, 16, ("0 minutes", 0), cost, cond, rot, "gate_fresh")
                    add("gate_stale", si, 16, AGES[3], cost, cond, rot, "gate_stale")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    rows = build()
    from collections import Counter
    print(len(rows), Counter(r["part"] for r in rows))
    print(rows[0]["prompt"])
