"""Build the README results table from results/*.json (spec rule 3: numbers are copied from files, never typed).

usage: python -m wa.report results/ > results/TABLE.md
Seeds of the same model are pooled: mean over seeds, 95% bootstrap CI over tasks of the per-task mean."""
from __future__ import annotations

import json
import pathlib
import sys

from wa.run import bootstrap_ci


def main(d: str):
    files = sorted(pathlib.Path(d).glob("*.json"))
    if not files:
        raise SystemExit(f"REFUSE: no results JSON under {d}")
    by_model: dict[str, list[dict]] = {}
    for f in files:
        r = json.loads(f.read_text())
        if "rows" in r:
            by_model.setdefault(f"{r['model']}{'' if r.get('som', True) else ' (no SoM)'}", []).append(r)
    sites = sorted({s for rs in by_model.values() for r in rs for s in r["per_site"]})
    print("| Model | Seeds | Tasks | Success (95% CI) | " + " | ".join(sites) + " | Mean steps |")
    print("|---|---|---|---|" + "---|" * len(sites) + "---|")
    for model, rs in by_model.items():
        per_task: dict = {}
        for r in rs:
            for row in r["rows"]:
                per_task.setdefault(row["task_id"], []).append(row["success"])
        xs = [sum(v) / len(v) for v in per_task.values()]
        lo, hi = bootstrap_ci(xs)
        site_cells = []
        for s in sites:
            vals = [r["per_site"][s]["success"] for r in rs if s in r["per_site"]]
            site_cells.append(f"{100 * sum(vals) / len(vals):.1f}" if vals else "–")
        steps = sum(r["mean_steps"] for r in rs) / len(rs)
        print(f"| {model} | {len(rs)} | {len(per_task)} | {100 * sum(xs) / len(xs):.1f} ({100 * lo:.1f}–{100 * hi:.1f}) | "
              + " | ".join(site_cells) + f" | {steps:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
