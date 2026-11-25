"""Trajectory-first relabeling (hindsight task synthesis) for SYNTHETIC collection runs only.

A teacher episode that wandered somewhere coherent is a correct demonstration of the task it actually did. For each
episode we pick an endpoint step j (a page reached after >= 2 actions, before any loop), ask the teacher to write the
task that steps 0..j-1 accomplish (NAVIGATE to that page, or RETRIEVE a fact visible on it) plus one reasoning line
per step, then verify like Phase 2: (a) programmatic - a RETRIEVE answer must literally occur in the endpoint page
text; (b) the LLM judge on the endpoint screenshot. Output is a collect-style dir that wa.build_data reads.

NEVER point this at eval runs: those trajectories were driven by WebArena test intents (rule 1).
usage: python -m wa.relabel --inputs DIR... --out DIR --model M --endpoint URL"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from wa import verify
from wa.actions import to_str
from wa.episode import system_prompt
from wa.policy import OpenAIPolicy

PROMPT = """Below is a sequence of browser actions an agent took on {blurb}, starting at {start}.
Each line: step number, the page URL, and the action (clicked/typed element names are given).
{steps}

The final page reached ({end_url}, title "{end_title}") shows this text:
{end_text}

Write ONE realistic task a user could ask for which this exact action sequence is a sensible solution.
Choose task_type NAVIGATE (the goal is to reach this final page) or RETRIEVE (the goal is a fact shown on the final
page; copy the answer EXACTLY from the final page text). Do not mention step numbers or element ids.
Also write one short reasoning sentence per step explaining why that action moves toward the task.
Return ONLY JSON: {{"task_type": "NAVIGATE|RETRIEVE", "intent": "...", "answer": "..." or null, "reasoning": ["...", ...]}}
(the reasoning list must have exactly {n} items)"""


def _name(step: dict, act: dict) -> str:
    if "id" in act:
        e = next((e for e in step.get("elements", []) if e["id"] == act["id"]), None)
        if e:
            return f"{act['action']} [{e['role']} '{e['name']}']" + (f" text='{act['text']}'" if act.get("text") else "")
    return to_str(act)


def pick_endpoint(steps: list[dict]) -> int | None:
    """Index j of the step whose observation is the endpoint: the last new URL reached before any loop, j >= 2."""
    acts = [s for s in steps if "action" in s and "image" in s]
    if len(acts) < 3 or any("env_error" in s for s in steps):
        return None
    last_new, seen, run = None, set(), 1
    for i, s in enumerate(acts):
        # a loop is the same action on the same page (clicking element 1 on three different pages is progress)
        if i and (s["url"], to_str(s["action"])) == (acts[i - 1]["url"], to_str(acts[i - 1]["action"])):
            run += 1
            if run > 2:  # looping: stop considering later states
                break
        else:
            run = 1
        if s["url"] not in seen:
            seen.add(s["url"])
            if i >= 2:
                last_new = i
    return last_new


def relabel_episode(ep: pathlib.Path, site_blurb: str, policy, judge) -> dict | None:
    steps = [json.loads(l) for l in open(ep / "traj.jsonl")]
    acts = [s for s in steps if "action" in s and "image" in s]
    j = pick_endpoint(steps)
    if j is None:
        return {"drop": "no_endpoint" if len(acts) >= 3 else "too_short"}
    end = acts[j]
    lines = "\n".join(f"{i + 1}. {s['url']} -> {_name(s, s['action'])}" for i, s in enumerate(acts[:j]))
    end_text = "\n".join(end["text"])[:2500]
    raw, _ = policy([{"role": "user", "content": PROMPT.format(
        blurb=site_blurb, start=acts[0]["url"], steps=lines, end_url=end["url"], end_title=end["title"],
        end_text=end_text, n=j)}])
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", raw, flags=re.S), re.S)
    if not m:
        return {"drop": "no_json"}
    try:
        lab = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"drop": "bad_json"}
    tt, intent, reasons = lab.get("task_type"), lab.get("intent"), lab.get("reasoning")
    if tt not in ("NAVIGATE", "RETRIEVE") or not isinstance(intent, str) or len(intent) < 15:
        return {"drop": "bad_label"}
    if not isinstance(reasons, list) or len(reasons) != j or not all(isinstance(r, str) for r in reasons):
        return {"drop": "reasoning_count"}
    if tt == "RETRIEVE":  # (a) programmatic check: answer literally on the endpoint page
        ans = lab.get("answer")
        if not isinstance(ans, str) or len(ans.strip()) < 1 or ans.strip().lower() not in end_text.lower():
            return {"drop": "answer_not_on_page"}
        answer = {"task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": [ans.strip()], "error_details": None}
    else:
        answer = {"task_type": "NAVIGATE", "status": "SUCCESS", "retrieved_data": None, "error_details": None}
    # (b) the Phase-2 LLM judge on the endpoint screenshot
    acts_str = [to_str(s["action"]) for s in acts[:j]]
    jraw, _ = judge(verify.judge_messages(intent, answer, acts_str, Image.open(ep / end["image"])))
    if not verify.judge_says_yes(jraw):
        return {"drop": "judge_no"}
    # new trajectory: steps 0..j-1 with new reasoning, then stop at step j; history rebuilt for the new intent
    new_steps, history = [], []
    for i in range(j + 1):
        s = dict(acts[i])
        s["image"] = acts[i]["image"]  # copied next to the new traj in main(): paths differ across kernels
        s["history"], s["last_error"] = history[-8:], ""
        if i < j:
            s["reasoning"] = reasons[i].strip()
            history = history + [to_str(s["action"])]
        else:
            s["action"] = {"action": "stop", "answer": answer}
            s["reasoning"] = "The task is complete." if tt == "NAVIGATE" else "The answer is shown on this page."
        new_steps.append(s)
    return {"task_type": tt, "intent": intent.strip(), "answer": answer, "steps": new_steps, "source": str(ep)}


def main():
    from wa.taskgen import SITE_BLURB
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--config", default="")
    ap.add_argument("--sites", default="")
    a, _ = ap.parse_known_args()
    policy = OpenAIPolicy(a.endpoint, a.model, temperature=0.3, max_tokens=1200)
    judge = OpenAIPolicy(a.endpoint, a.model, temperature=0.0, max_tokens=8)
    out = pathlib.Path(a.out)
    runs_out = out / "runs"
    runs_out.mkdir(parents=True, exist_ok=True)
    eps = []
    for d in a.inputs:
        for tasks_f in pathlib.Path(d).rglob("synthetic_tasks.jsonl"):
            if "out_eval" in str(tasks_f):
                raise SystemExit(f"REFUSE: {tasks_f} looks like an eval run; relabeling eval trajectories leaks test tasks")
            site = json.loads(tasks_f.open().readline())["site"]
            for ep in sorted((tasks_f.parent / "runs").iterdir()):
                if (ep / "traj.jsonl").exists():
                    eps.append((ep, site))
    if not eps:
        raise SystemExit("REFUSE: no collection episodes found")

    def one(item):
        ep, site = item
        try:
            return site, ep, relabel_episode(ep, SITE_BLURB[site], policy, judge)
        except Exception as e:  # noqa: BLE001
            return site, ep, {"error": repr(e)[:200]}

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, eps))
    tasks, verified, n_ok, errs = [], [], 0, 0
    drops = {}
    for k, (site, ep, r) in enumerate(res):
        if "error" in r:
            errs += 1
            continue
        if "drop" in r:
            drops[r["drop"]] = drops.get(r["drop"], 0) + 1
            continue
        tid = f"rel-{site}-{ep.parent.parent.name}-{ep.name}"
        d = runs_out / f"{tid}-s0"
        d.mkdir(parents=True, exist_ok=True)
        for s in r["steps"]:  # self-contained episode: copy its screenshots
            shutil.copy(ep / s["image"], d / s["image"])
        (d / "traj.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in r["steps"]))
        (d / "agent_response.json").write_text(json.dumps(r["answer"]))
        start = r["steps"][0]["url"]  # the original episode's start page
        (d / "meta.json").write_text(json.dumps({"task_id": f"{tid}-s0",
                                                 "system": system_prompt([site], r["intent"], [start])}))
        tasks.append({"task_id": tid, "site": site, "sites": [site], "intent": r["intent"], "task_type": r["task_type"],
                      "source": r["source"]})
        verified.append({"task_id": f"{tid}-s0", "task_type": r["task_type"], "program": True, "judge": True,
                         "verified": True, "steps": len(r["steps"]), "intent": r["intent"]})
        n_ok += 1
    (out / "synthetic_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    (out / "verified.jsonl").write_text("\n".join(json.dumps(v) for v in verified))
    summ = {"episodes_in": len(eps), "relabeled": n_ok, "errors": errs, "drops": drops,
            "relabeled_steps": sum(v["steps"] for v in verified)}
    (out / "relabel_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    if n_ok == 0:
        raise SystemExit("REFUSE: zero episodes relabeled")


if __name__ == "__main__":
    main()
