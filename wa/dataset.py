"""Turn verified episodes into SFT rows. Each step becomes one (observation -> reasoning + action) example,
rebuilt with exactly the message format the policy sees at eval time (wa.policy.build_messages)."""
from __future__ import annotations

import json
import pathlib

from wa.actions import to_str
from wa.policy import build_messages


class _ImgRef:
    """Stand-in for a PIL image inside messages: serialises as the path."""
    def __init__(self, path):
        self.path = str(path)


def rows_from_episode(ep_dir: pathlib.Path, intent: str) -> list[dict]:
    meta = json.loads((ep_dir / "meta.json").read_text())
    rows = []
    for line in open(ep_dir / "traj.jsonl"):
        s = json.loads(line)
        if "action" not in s or "image" not in s:
            continue  # parse errors / env errors are not imitated
        obs = {"image": _ImgRef(ep_dir / s["image"]), "elements": s["elements"], "text": s["text"],
               "url": s["url"], "title": s["title"], "scroll": tuple(s["scroll"])}
        msgs = build_messages(meta["system"], intent, obs, s["history"], s["last_error"])
        for m in msgs:
            if isinstance(m["content"], list):
                m["content"] = [{"type": "image"} if p["type"] == "image" else p for p in m["content"]]
        target = (s.get("reasoning") or "Next step.").strip() + "\n" + to_str(s["action"])
        rows.append({"messages": msgs, "target": target, "image": obs["image"].path,
                     "task_id": meta["task_id"], "step": s["step"]})
    return rows


def write_rows(episodes: list[tuple[pathlib.Path, str]], out: pathlib.Path) -> dict:
    n_steps, n_eps = 0, 0
    with open(out, "w") as f:
        for ep_dir, intent in episodes:
            rows = rows_from_episode(ep_dir, intent)
            if rows:
                n_eps += 1
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_steps += 1
    if n_steps == 0:
        raise SystemExit("REFUSE: zero SFT rows written")
    return {"episodes": n_eps, "steps": n_steps}
