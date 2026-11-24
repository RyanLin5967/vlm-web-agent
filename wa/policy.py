"""Policy = the only thing that differs between models (spec rule 2).

Every model is called through an OpenAI-compatible chat endpoint (vLLM `serve` for open models), with the
identical message built by `build_messages`. Token usage is returned so cost per task can be computed."""
from __future__ import annotations

import base64
import io
import json
import urllib.request

from wa.actions import ACTION_DOC
from wa.obs import element_line

MAX_HISTORY = 8


def build_messages(system: str, intent: str, obs: dict, history: list[str], last_error: str, som: bool = True) -> list[dict]:
    els = "\n".join(element_line(e) for e in obs["elements"]) or "(none)"
    txt = "\n".join(obs["text"]) or "(none)"
    hist = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(history[-MAX_HISTORY:])) or "(none)"
    y, h = obs["scroll"]
    user = (f"Task: {intent}\n\nCurrent URL: {obs['url']}\nPage title: {obs['title']}\n"
            f"Scroll: {y}px of {h}px\n\nPrevious actions:\n{hist}\n")
    if last_error:
        user += f"\nLast action error: {last_error}\n"
    user += (f"\nInteractive elements{' (numbered boxes in the screenshot)' if som else ''}:\n{els}\n\n"
             f"Visible text:\n{txt}\n\n{ACTION_DOC}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image", "image": obs["image"]}, {"type": "text", "text": user}]}]


def _png_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(m)
            continue
        parts = []
        for p in m["content"]:
            if p["type"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": _png_data_url(p["image"])}})
            else:
                parts.append(p)
        out.append({"role": m["role"], "content": parts})
    return out


class OpenAIPolicy:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 384, api_key: str = "x",
                 seed: int | None = None):
        self.base_url, self.model, self.temperature, self.max_tokens, self.api_key = base_url.rstrip("/"), model, temperature, max_tokens, api_key
        self.seed = seed

    def __call__(self, messages: list[dict]) -> tuple[str, dict]:
        body = {"model": self.model, "messages": to_openai(messages), "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if self.seed is not None:
            body["seed"] = self.seed
        if "qwen3.5" in self.model.lower():  # identical setting for every Qwen3.5 size: no thinking blocks
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
        u = resp.get("usage", {})
        return resp["choices"][0]["message"]["content"] or "", {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
