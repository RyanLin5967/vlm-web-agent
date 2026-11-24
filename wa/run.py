"""Run a policy over tasks with N concurrent episodes in one browser.

  eval:    WebArena-Verified test tasks (rendered by the official `agent-input-get`), scored by the official
           `eval-tasks`, summarised into results/<name>.json (per-task success, steps, tokens, bootstrap CI).
  collect: synthetic tasks (JSONL from wa.taskgen), verified by wa.verify, for SFT data.

usage:
  python -m wa.run eval    --config wa_config.json --sites shopping_admin --model M --endpoint http://localhost:8000/v1 --out out/x
  python -m wa.run collect --tasks syn.jsonl --model M --endpoint ... --out out/c --samples 1"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import subprocess
import sys
import time

from playwright.async_api import async_playwright

from wa.episode import CREDS, login, run_episode
from wa.policy import OpenAIPolicy


def bootstrap_ci(xs: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        raise SystemExit("REFUSE: zero scored tasks — nothing to report")
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


async def run_many(tasks, policy, out_root: pathlib.Path, base_urls: dict, concurrency: int, som: bool,
                   save_images: bool) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--disable-dev-shm-usage"])
        states = {}
        for site in sorted({s for t in tasks for s in t["sites"]}):
            if site in CREDS:
                sp = out_root / f".state_{site}.json"
                await login(browser, site, base_urls[site], sp)
                states[site] = sp
        sem = asyncio.Semaphore(concurrency)
        done = []

        async def one(t):
            async with sem:
                d = out_root / str(t["task_id"])
                if (d / "meta.json").exists():  # resumable: never redo a finished episode
                    done.append(json.loads((d / "meta.json").read_text()))
                    return
                site = next((s for s in t["sites"] if s in states), None)
                # multi-site tasks: merge storage states so every site is logged in
                sp = states.get(site)
                if len([s for s in t["sites"] if s in states]) > 1:
                    merged = {"cookies": [], "origins": []}
                    for s in t["sites"]:
                        if s in states:
                            st = json.loads(states[s].read_text())
                            merged["cookies"] += st.get("cookies", []); merged["origins"] += st.get("origins", [])
                    sp = d.parent / f".state_{t['task_id']}.json"
                    d.parent.mkdir(parents=True, exist_ok=True)
                    sp.write_text(json.dumps(merged))
                meta = await run_episode(browser, t, policy, d, sp, som=som, save_images=save_images)
                done.append(meta)
                print(json.dumps({"done": len(done), "of": len(tasks), "task": t["task_id"], "steps": meta["steps"],
                                  "wall_s": meta["wall_s"]}), flush=True)

        await asyncio.gather(*(one(t) for t in tasks))
        await browser.close()
    return done


class ScriptedPolicy:
    """Pipeline smoke test only: scrolls once, then gives up. Exercises every stage except the model."""
    def __call__(self, msgs):
        if "Previous actions:\n(none)" in msgs[1]["content"][1]["text"]:
            return 'Look around.\n{"action": "scroll", "direction": "down"}', {"in": 0, "out": 0}
        return ('Give up.\n{"action": "stop", "answer": {"task_type": "RETRIEVE", "status": "UNKNOWN_ERROR", '
                '"retrieved_data": null, "error_details": "smoke"}}'), {"in": 0, "out": 0}


def cmd_eval(a):
    cfg = json.loads(pathlib.Path(a.config).read_text())
    base_urls = {k.strip("_").lower(): v["urls"][0] for k, v in cfg["environments"].items()}
    tasks_file = pathlib.Path(a.out) / "agent_input.json"
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    cmd = ["webarena-verified", "agent-input-get", "--config", a.config, "--output", str(tasks_file)]
    if a.sites:
        cmd += ["--sites", a.sites]
    if a.task_ids:
        cmd += ["--task-ids", a.task_ids]
    subprocess.run(cmd, check=True)
    tasks = json.loads(tasks_file.read_text())
    if a.sites:  # only tasks whose every site is served in this run
        allowed = set(a.sites.split(","))
        tasks = [t for t in tasks if set(t["sites"]) <= allowed]
    if a.limit:
        tasks = tasks[: a.limit]
    if not tasks:
        raise SystemExit("REFUSE: zero tasks selected")
    policy = (ScriptedPolicy() if a.model == "scripted" else
              OpenAIPolicy(a.endpoint, a.model, temperature=a.temperature, api_key=a.api_key, seed=a.seed))
    t0 = time.time()
    metas = asyncio.run(run_many(tasks, policy, pathlib.Path(a.out) / "runs", base_urls, a.concurrency, not a.no_som,
                                 save_images=False))
    subprocess.run(["webarena-verified", "eval-tasks", "--config", a.config, "--output-dir", str(pathlib.Path(a.out) / "runs")],
                   check=False)
    summarize(a, tasks, metas, time.time() - t0)


def read_score(task_dir: pathlib.Path):
    """Read the official per-task eval result; None if the evaluator did not produce one."""
    for name in ("eval_result.json",):
        f = task_dir / name
        if f.exists():
            r = json.loads(f.read_text())
            s = r.get("score", r.get("success"))
            return float(s) if s is not None else None
    return None


def summarize(a, tasks, metas, wall_s):
    runs = pathlib.Path(a.out) / "runs"
    bym = {str(m["task_id"]): m for m in metas}
    rows, missing = [], []
    for t in tasks:
        sc = read_score(runs / str(t["task_id"]))
        if sc is None:
            missing.append(t["task_id"])
            continue
        m = bym.get(str(t["task_id"]), {})
        rows.append({"task_id": t["task_id"], "sites": t["sites"], "success": sc, "steps": m.get("steps"),
                     "tokens_in": m.get("usage", {}).get("in"), "tokens_out": m.get("usage", {}).get("out")})
    if missing:
        print(f"WARNING: {len(missing)} tasks have no official eval result: {missing[:20]}", file=sys.stderr)
    xs = [r["success"] for r in rows]
    lo, hi = bootstrap_ci(xs)
    per_site = {}
    for r in rows:
        key = r["sites"][0] if len(r["sites"]) == 1 else "multi"
        per_site.setdefault(key, []).append(r["success"])
    res = {"model": a.model, "endpoint": a.endpoint, "seed": a.seed, "temperature": a.temperature, "som": not a.no_som,
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_tasks": len(rows), "n_missing_eval": len(missing),
           "success": sum(xs) / len(xs), "ci95": [lo, hi],
           "per_site": {k: {"n": len(v), "success": sum(v) / len(v)} for k, v in sorted(per_site.items())},
           "mean_steps": sum(r["steps"] or 0 for r in rows) / len(rows),
           "mean_tokens_in": sum(r["tokens_in"] or 0 for r in rows) / len(rows),
           "mean_tokens_out": sum(r["tokens_out"] or 0 for r in rows) / len(rows),
           "wall_s": round(wall_s), "rows": rows}
    out = pathlib.Path(a.out) / "results.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


def _base_urls(config_path):
    cfg = json.loads(pathlib.Path(config_path).read_text())
    return {k.strip("_").lower(): v["urls"][0] for k, v in cfg["environments"].items()}


def cmd_collect(a):
    """Phase 1+2 on one site: crawl -> propose -> leakage filter -> teacher episodes -> verify -> keep successes."""
    from wa import leakage, taskgen, verify
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base_urls = _base_urls(a.config)
    site = a.sites
    home = base_urls[site]
    policy = OpenAIPolicy(a.endpoint, a.model, temperature=a.temperature, api_key=a.api_key, seed=a.seed)
    text_policy = OpenAIPolicy(a.endpoint, a.model, temperature=0.7, max_tokens=1500, api_key=a.api_key)
    tasks_f = out / "synthetic_tasks.jsonl"
    if not tasks_f.exists():
        async def gen():
            async with async_playwright() as p:
                b = await p.chromium.launch(args=["--disable-dev-shm-usage"])
                sp = out / f".state_{site}.json"
                await login(b, site, home, sp)
                pages = await taskgen.crawl(b, site, home, sp, max_pages=a.crawl_pages, seed=a.seed)
                await b.close()
            (out / "crawl.jsonl").write_text("\n".join(json.dumps(x) for x in pages))
            return await taskgen.propose_all(pages, {site: home}, text_policy, k=4, concurrency=a.concurrency * 2)
        props = asyncio.run(gen())
        if not props:
            raise SystemExit("REFUSE: zero task proposals")
        hits = leakage.check(props)
        bad = {h["i"] for h in hits}
        (out / "leakage_drops.json").write_text(json.dumps(hits, indent=1))
        kept = [dict(t, task_id=f"syn-{site}-{a.seed}-{i}") for i, t in enumerate(props) if i not in bad]
        tasks_f.write_text("\n".join(json.dumps(t) for t in kept))
        print(json.dumps({"proposals": len(props), "leakage_dropped": len(bad), "kept": len(kept)}), flush=True)
    tasks = [json.loads(l) for l in open(tasks_f) if l.strip()]
    if a.limit:
        tasks = tasks[: a.limit]
    runs = out / "runs"
    expanded = [dict(t, task_id=f"{t['task_id']}-s{k}") for t in tasks for k in range(a.samples)]
    metas = asyncio.run(run_many(expanded, policy, runs, base_urls, a.concurrency, True, save_images=True))
    bym = {m["task_id"]: m for m in metas}

    async def verify_all():
        async with async_playwright() as p:
            b = await p.chromium.launch(args=["--disable-dev-shm-usage"])
            sp = runs / f".state_{site}.json"
            res = []
            for t in expanded:
                d = runs / t["task_id"]
                m = bym.get(t["task_id"])
                if not m or not (d / "agent_response.json").exists():
                    continue
                ans = json.loads((d / "agent_response.json").read_text())
                if t["task_type"] == "RETRIEVE":
                    prog = verify.check_retrieve(t, ans)
                elif t["task_type"] == "NAVIGATE":
                    prog = verify.check_navigate(t, ans, m.get("final_url"))
                else:
                    prog = await verify.check_mutate(b, t, ans, sp)
                judge = None
                if prog and (d / "final.png").exists():
                    from PIL import Image
                    acts = [json.dumps(json.loads(l).get("action")) for l in open(d / "traj.jsonl")]
                    raw, _ = await asyncio.to_thread(policy, verify.judge_messages(t["intent"], ans, acts, Image.open(d / "final.png")))
                    judge = verify.judge_says_yes(raw)
                res.append({"task_id": t["task_id"], "task_type": t["task_type"], "program": prog, "judge": judge,
                            "verified": bool(prog and judge), "steps": m["steps"], "intent": t["intent"]})
            await b.close()
            return res

    res = asyncio.run(verify_all())
    (out / "verified.jsonl").write_text("\n".join(json.dumps(r) for r in res))
    ok = [r for r in res if r["verified"]]
    summary = {"site": site, "model": a.model, "episodes": len(res), "program_pass": sum(r["program"] for r in res),
               "verified": len(ok), "verified_steps": sum(r["steps"] for r in ok),
               "by_type": {tt: sum(1 for r in ok if r["task_type"] == tt) for tt in ("RETRIEVE", "NAVIGATE", "MUTATE")}}
    (out / "collect_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if not res:
        raise SystemExit("REFUSE: zero episodes verified-or-not — collection produced nothing")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--config", required=True)
    e.add_argument("--sites", default="")
    e.add_argument("--task-ids", default="")
    e.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("collect")
    c.add_argument("--config", required=True)
    c.add_argument("--sites", required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--samples", type=int, default=1)
    c.add_argument("--crawl-pages", type=int, default=250)
    for p in (e, c):
        p.add_argument("--model", required=True)
        p.add_argument("--endpoint", required=True)
        p.add_argument("--api-key", default="x")
        p.add_argument("--out", required=True)
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-som", action="store_true")
    a = ap.parse_args()
    {"eval": cmd_eval, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
