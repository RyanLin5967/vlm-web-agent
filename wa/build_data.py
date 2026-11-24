"""Collect verified episodes from collection outputs into SFT data, with a task-level dev split and the
spec rule-1 leakage gate over every training intent (exit non-zero on any hit or on zero rows).

usage: python -m wa.build_data --inputs DIR [DIR ...] --out data_dir [--dev-frac 0.05]
Each input DIR contains out_collect*/ (synthetic_tasks.jsonl, verified.jsonl, runs/<task>-s<k>/...)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from wa import leakage
from wa.dataset import rows_from_episode


def find_collect_dirs(inputs):
    """-> (collect dirs, judge overrides {episode_id: verified}). Judge outputs (wa.run judge) carry source_dir.txt."""
    out, overrides = [], {}
    for d in inputs:
        for v in pathlib.Path(d).rglob("verified.jsonl"):
            if (v.parent / "source_dir.txt").exists():
                for r in map(json.loads, filter(str.strip, open(v))):
                    overrides[r["task_id"]] = bool(r.get("verified"))
            else:
                out.append(v.parent)
    return sorted(set(out)), overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dirs, overrides = find_collect_dirs(a.inputs)
    if not dirs:
        raise SystemExit(f"REFUSE: no verified.jsonl under {a.inputs}")
    train, dev_tasks, stats = [], [], {}
    seen_base = {}
    for d in dirs:
        tasks = {t["task_id"]: t for t in map(json.loads, open(d / "synthetic_tasks.jsonl")) if t}
        for v in map(json.loads, open(d / "verified.jsonl")):
            base = v["task_id"].rsplit("-s", 1)[0]
            t = tasks.get(base)
            if t is None:
                continue
            # task-level split, stable across rounds: hash of the base task id
            is_dev = int(hashlib.sha256(base.encode()).hexdigest(), 16) % 10000 < a.dev_frac * 10000
            if is_dev:
                if base not in seen_base:
                    dev_tasks.append(t); seen_base[base] = 1
                continue
            if not overrides.get(v["task_id"], v["verified"]):
                continue
            rows = rows_from_episode(d / "runs" / v["task_id"], t["intent"])
            for r in rows:
                r["image"] = str(pathlib.Path(r["image"]).resolve())
            train.extend(rows)
            s = stats.setdefault(t["site"], {"episodes": 0, "steps": 0})
            s["episodes"] += 1; s["steps"] += len(rows)
    if not train:
        raise SystemExit("REFUSE: zero verified training steps")
    intents = list({r["messages"][1]["content"][1]["text"].split("\n", 1)[0][len("Task: "):] for r in train})
    hits = leakage.check([{"intent": i} for i in intents])
    (out / "leakage_report.json").write_text(json.dumps({"n_intents": len(intents), "n_hits": len(hits), "hits": hits}, indent=1))
    if hits:
        raise SystemExit(f"REFUSE: {len(hits)} leakage hits in training intents (see leakage_report.json)")
    with open(out / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "dev_tasks.jsonl", "w") as f:
        for t in dev_tasks:
            f.write(json.dumps(t) + "\n")
    summary = {"collect_dirs": [str(d) for d in dirs], "train_steps": len(train), "train_intents": len(intents),
               "dev_tasks": len(dev_tasks), "per_site": stats, "leakage_hits": 0,
               "judge_overrides": len(overrides)}
    (out / "data_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
