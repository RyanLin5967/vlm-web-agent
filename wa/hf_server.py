"""Fallback policy server when vLLM cannot run (e.g. Turing T4): transformers generate behind a minimal
OpenAI-compatible /v1/chat/completions, with micro-batching (left-padded) across concurrent requests.

usage: python -m wa.hf_server --model Qwen/Qwen3.5-2B [--adapter ckpt/] --port 8000 [--device 0] [--max-batch 8]"""
from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

Q: "queue.Queue" = queue.Queue()


def decode_messages(messages):
    """OpenAI messages -> (HF chat messages with {"type":"image"} placeholders, list of PIL images)."""
    imgs, out = [], []
    for m in messages:
        if isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"]})
            continue
        parts = []
        for p in m["content"]:
            if p["type"] == "image_url":
                url = p["image_url"]["url"]
                imgs.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": p["text"]})
        out.append({"role": m["role"], "content": parts})
    return out, imgs


def worker(model, proc, max_batch, model_name):
    proc.tokenizer.padding_side = "left"
    while True:
        batch = [Q.get()]
        t_end = time.time() + 0.05
        while len(batch) < max_batch and time.time() < t_end:
            try:
                batch.append(Q.get(timeout=max(0.0, t_end - time.time())))
            except queue.Empty:
                break
        # group by sampling params so one generate() call serves the group
        groups = {}
        for item in batch:
            key = (item["body"].get("temperature", 0.0), item["body"].get("max_tokens", 384))
            groups.setdefault(key, []).append(item)
        for (temp, max_tokens), items in groups.items():
            try:
                texts, images = [], []
                for it in items:
                    msgs, imgs = decode_messages(it["body"]["messages"])
                    kw = it["body"].get("chat_template_kwargs") or {}
                    texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw))
                    images.extend(imgs)
                enc = proc(text=texts, images=images or None, return_tensors="pt", padding=True).to(model.device)
                gen_kw = {"max_new_tokens": max_tokens, "do_sample": temp > 0}
                if temp > 0:
                    gen_kw["temperature"] = temp
                with torch.no_grad():
                    out = model.generate(**enc, **gen_kw)
                n_in = enc["input_ids"].shape[1]
                for i, it in enumerate(items):
                    new = out[i, n_in:]
                    txt = proc.tokenizer.decode(new, skip_special_tokens=True)
                    prompt_tokens = int(enc["attention_mask"][i].sum())
                    it["result"] = {"id": "x", "object": "chat.completion", "model": model_name,
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": prompt_tokens,
                                              "completion_tokens": int((new != proc.tokenizer.pad_token_id).sum())}}
            except Exception as e:  # noqa: BLE001
                for it in items:
                    it["error"] = repr(e)[:500]
            for it in items:
                it["done"].set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.adapter or a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype), device_map={"": a.device})
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    name = a.adapter or a.model
    threading.Thread(target=worker, args=(model, proc, a.max_batch, name), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"data": [{"id": name}]}) if self.path.startswith("/v1/models") else self._send(404, {})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._send(404, {})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            item = {"body": body, "done": threading.Event()}
            Q.put(item)
            item["done"].wait()
            self._send(500, {"error": item["error"]}) if "error" in item else self._send(200, item["result"])

    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
