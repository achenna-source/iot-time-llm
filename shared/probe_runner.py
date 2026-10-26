# -*- coding: utf-8 -*-
"""Kaggle GPU runner shared by the AGE and ORDER audits (one model per kernel).

Task "letters": next-token probability of each option letter after the chat template's generation prompt
                (one forward pass per probe, batched with left padding); also the total probability mass
                on the letter tokens, which the format gate uses.
Task "generate": greedy completion (native tool calls, structured extraction).
Configuration is injected as module-level assignments by push.py.
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SHORT = "qwen2.5-7b"
LOAD_4BIT = True
INPUTS = []            # list of [glob pattern, task, part filter or None, limit or None]
BATCH = 8
MAX_NEW = 200
OUT_DIR = "/kaggle/working"


def ensure_libs():
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=False)
    try:
        import transformers
        print("transformers", transformers.__version__, flush=True)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.float16, "device_map": "auto", "trust_remote_code": True, "attn_implementation": "sdpa"}
    if LOAD_4BIT:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    return tok, model


def render(tok, messages, tools=None):
    kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    if tools:
        kw["tools"] = tools
    try:
        return tok.apply_chat_template(messages, **kw)
    except Exception as e:  # template without system role or without tools
        if messages and messages[0]["role"] == "system":
            merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
            kw.pop("tools", None)
            return tok.apply_chat_template(merged, **kw)
        raise e


def letter_ids(tok, letters):
    ids = {}
    for L in letters:
        cand = set()
        for s in (L, " " + L):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                cand.add(enc[0])
        ids[L] = sorted(cand)
    return ids


def run_letters(tok, model, rows, out, prefix=""):
    import torch
    letters = sorted({L for r in rows for L in r["letters"]})
    lid = letter_ids(tok, letters)
    print("letter token ids", lid, "assistant prefix %r" % prefix, flush=True)
    texts = [render(tok, r["messages"]) + r.get("prefix", prefix) for r in rows]   # per-row prefix: forced answers
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    t0 = time.time()
    done = 0
    with open(out, "a", encoding="utf-8") as f:
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.inference_mode():
                try:
                    logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
                except TypeError:
                    logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, -1)
            top = lp.argmax(-1).tolist()
            for j, i in enumerate(idx):
                r = rows[i]
                pr = {L: float(sum(math.exp(lp[j, t].item()) for t in lid[L])) for L in r["letters"]}
                mass = sum(pr.values())
                rec = {"id": r["id"], "model": SHORT, "mass": mass, "top": tok.decode([top[j]]),
                       "probs": {L: (pr[L] / mass if mass > 0 else 1.0 / len(pr)) for L in pr}, "n_tok": lens[i]}
                f.write(json.dumps(rec) + "\n")
            done += len(idx)
            if done % (BATCH * 25) < BATCH or done == len(rows):
                el = time.time() - t0
                print("  %d/%d  %.2f s/probe  eta %.0f min" % (done, len(rows), el / done, el / done * (len(rows) - done) / 60), flush=True)


def run_generate_batched(tok, model, rows, out, bs):
    """greedy generation in left-padded batches (rows without tools only)."""
    import torch
    t0 = time.time()
    texts = [render(tok, r["messages"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    done = 0
    with open(out, "a", encoding="utf-8") as f:
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            mx = max(rows[i].get("max_new", MAX_NEW) for i in idx)
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                ans = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"id": rows[i]["id"], "model": SHORT, "text": ans}) + "\n")
            done += len(idx)
            el = time.time() - t0
            print("  gen-batch %d/%d  %.2f s/item" % (done, len(rows), el / done), flush=True)


def run_generate(tok, model, rows, out, bs=1):
    import torch
    if bs > 1 and not any(r.get("tools") for r in rows):
        return run_generate_batched(tok, model, rows, out, bs)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            text = render(tok, r["messages"], r.get("tools"))
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=r.get("max_new", MAX_NEW), do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            ans = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            f.write(json.dumps({"id": r["id"], "model": SHORT, "text": ans}) + "\n")
            if n % 25 == 0 or n == len(rows):
                el = time.time() - t0
                print("  gen %d/%d  %.2f s/item" % (n, len(rows), el / n), flush=True)


def load_rows(pattern, task, part, limit):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print("WARNING: no file for", pattern, "- skipped", flush=True)
        return []
    rows = []
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            if part and r.get("part") not in part:
                continue
            if "messages" not in r:
                r["messages"] = [{"role": "user", "content": r["prompt"]}]
            if task == "letters" and "letters" not in r:
                r["letters"] = sorted(r["roles"].keys())
            rows.append({k: r[k] for k in ("id", "messages", "letters", "tools", "max_new", "prefix") if k in r})
    if limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    print(pattern, task, part, "->", len(rows), "rows", flush=True)
    return rows


def main():
    ensure_libs()
    t = time.time()
    tok, model = load()
    print("loaded %s in %.0f s" % (MODEL_ID, time.time() - t), flush=True)
    meta = {"model_id": MODEL_ID, "short": SHORT, "load_4bit": LOAD_4BIT, "inputs": INPUTS}
    for spec in INPUTS:
        pattern, task, part, limit, tag = spec[:5]
        prefix = spec[5] if len(spec) > 5 else ""
        rows = load_rows(pattern, task, part, limit)
        if not rows:
            continue
        out = os.path.join(OUT_DIR, "%s__%s.jsonl" % (tag, SHORT))
        t1 = time.time()
        if task == "letters":
            run_letters(tok, model, rows, out, prefix)
        elif task == "generate_batched":
            run_generate(tok, model, rows, out, bs=int(prefix or 8))
        else:
            run_generate(tok, model, rows, out)
        meta[tag] = {"n": len(rows), "seconds": time.time() - t1}
    json.dump(meta, open(os.path.join(OUT_DIR, "meta__%s.json" % SHORT), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
