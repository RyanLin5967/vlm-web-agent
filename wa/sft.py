"""Phase 3/4 SFT: (observation -> reasoning line + action JSON), loss on assistant tokens only.

LoRA r=64 on the language model's linear layers + the vision merger/projector (spec fallback when full FT
does not fit — it does not on a 16 GB T4). Plain PyTorch loop, gradient checkpointing, grad accumulation
to the global batch, cosine schedule. Checkpoint chosen on synthetic dev success, never on WebArena.

Data: JSONL rows from wa.dataset.build_rows: {"messages": [...], "target": "...", "image": "<png path>"}.
usage: python -m wa.sft --data train.jsonl --out ckpt/ [--base Qwen/Qwen3.5-2B] [--epochs 2] ..."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from PIL import Image


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def encode(processor, row, max_len, max_pixels):
    img = Image.open(row["image"]).convert("RGB")
    msgs = json.loads(json.dumps(row["messages"]))
    for m in msgs:
        if isinstance(m["content"], list):
            for p in m["content"]:
                if p["type"] == "image":
                    p.clear(); p.update({"type": "image"})
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = prompt + row["target"] + processor.tokenizer.eos_token
    if max_pixels and img.width * img.height > max_pixels:  # spec: fit the model's pixel budget
        f = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((int(img.width * f), int(img.height * f)))
    kw = {"images": [img], "return_tensors": "pt"}
    enc_p = processor(text=[prompt], **kw)
    enc_f = processor(text=[full], **kw)
    ids = enc_f["input_ids"][0]
    if ids.shape[0] > max_len:
        return None  # never truncate the target away; drop the row (counted and reported)
    labels = ids.clone()
    labels[: enc_p["input_ids"].shape[1]] = -100
    out = {k: v for k, v in enc_f.items()}
    out["labels"] = labels.unsqueeze(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--global_batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_pixels", type=int, default=1280 * 720)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--limit_hours", type=float, default=10.5)  # stop and save before Kaggle's 12 h cutoff
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge_out", default="")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        a.dtype = "float32"  # CPU smoke tests only
    dtype = getattr(torch, a.dtype)
    processor = AutoProcessor.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, torch_dtype=dtype, device_map={"": dev},
                                                        attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    lm_targets = [t for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                              "in_proj_qkvz", "in_proj_ba", "out_proj"] if t in names]
    # vision projector / merger linears (names differ across Qwen-VL versions)
    proj = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "merger" in n and "visual" in n]
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=lm_targets + proj,
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():  # LoRA params in fp32 for stable fp16 training
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    rows = load_rows(a.data)
    random.shuffle(rows)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total_updates = max(1, math.ceil(len(rows) * a.epochs / a.global_batch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * total_updates))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_updates))))
    scaler = torch.amp.GradScaler("cuda", enabled=a.dtype == "float16" and dev != "cpu")
    model.train()
    t0, seen, upd, dropped, losses = time.time(), 0, 0, 0, []
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    stop = False
    order = [rows[i % len(rows)] for i in range(int(len(rows) * a.epochs))]
    micro = 0
    for row in order:
        enc = encode(processor, row, a.max_len, a.max_pixels)
        if enc is None:
            dropped += 1
            continue
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda" if dev != "cpu" else "cpu", dtype=dtype, enabled=a.dtype != "float32"):
            out = model(**enc)
        loss = out.loss / a.global_batch
        scaler.scale(loss).backward()
        losses.append(out.loss.item()); seen += 1; micro += 1
        if micro == a.global_batch:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            micro, upd = 0, upd + 1
            rec = {"update": upd, "of": total_updates, "seen": seen, "loss": sum(losses[-a.global_batch:]) / a.global_batch,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0), "dropped_too_long": dropped}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if not math.isfinite(rec["loss"]):
                raise SystemExit("REFUSE: non-finite loss — dtype overflow or bad data; not saving")
            if time.time() - t0 > a.limit_hours * 3600:
                stop = True
                break
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    if a.merge_out:  # full fp16 model for vLLM (no runtime LoRA, so the vision-projector adapter is kept)
        merged = model.merge_and_unload()
        merged = merged.to(torch.float16) if dev != "cpu" else merged
        merged.save_pretrained(a.merge_out, safe_serialization=True)
        processor.save_pretrained(a.merge_out)
    json.dump({"base": a.base, "rows": len(rows), "seen": seen, "updates": upd, "dropped_too_long": dropped,
               "stopped_early_for_time": stop, "args": vars(a), "elapsed_s": round(time.time() - t0)},
              open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    if seen == 0:
        raise SystemExit("REFUSE: zero training rows consumed")


if __name__ == "__main__":
    main()
