"""One episode: observe -> policy -> parse -> execute, up to 30 steps (spec), producing the files
WebArena-Verified scores (agent_response.json + network.har) and the trajectory used for training.

Login follows WebArena-Verified's reference agent: UI login in a separate context, storage state reused
by the task context, so login traffic never appears in the task's HAR."""
from __future__ import annotations

import json
import pathlib
import re
import time

from wa.actions import ParseError, execute, parse, to_str
from wa.obs import VIEWPORT, observe
from wa.policy import build_messages

MAX_STEPS = 30
PROMPTS = pathlib.Path(__file__).parent / "prompts"

CREDS = {
    "shopping": ("emma.lopez@gmail.com", "Password.123"),
    "shopping_admin": ("admin", "admin1234"),
    "gitlab": ("byteblaze", "hello1234"),
    "reddit": ("MarvelsGrantMan136", "test1234"),
}


def system_prompt(sites: list[str], intent: str, start_urls: list[str]) -> str:
    """Render WebArena-Verified's own per-site prompt (no jinja dependency: the templates use one loop)."""
    tpl = (PROMPTS / ("-".join(sorted(sites, key=str.lower)) + ".md")).read_text()
    tpl = re.sub(r"\{%\s*for start_url in START_URLS\s*%\}(.*?)\{%\s*endfor\s*%\}",
                 lambda m: "".join(m.group(1).replace("{{ start_url }}", u) for u in start_urls), tpl, flags=re.S)
    return tpl.replace("{{INTENT}}", intent)


async def login(browser, site: str, base_url: str, state_path: pathlib.Path) -> None:
    ctx = await browser.new_context(viewport=VIEWPORT)
    page = await ctx.new_page()
    user, pw = CREDS[site]
    try:
        if site == "shopping":
            await page.goto(f"{base_url}/customer/account/login/")
            await page.get_by_label("Email", exact=True).fill(user)
            await page.get_by_label("Password", exact=True).fill(pw)
            await page.get_by_role("button", name="Sign In").click()
        elif site == "shopping_admin":
            await page.goto(base_url)
            await page.get_by_label("Username").fill(user)
            await page.get_by_label("Password").fill(pw)
            await page.get_by_role("button", name="Sign in").click()
        elif site == "gitlab":
            await page.goto(f"{base_url}/users/sign_in")
            await page.get_by_label("Username or email").fill(user)
            await page.get_by_label("Password").fill(pw)
            await page.get_by_role("button", name="Sign in").click()
        elif site == "reddit":
            await page.goto(base_url)
            await page.get_by_role("link", name="Log in").click()
            await page.get_by_label("Username").fill(user)
            await page.get_by_label("Password").fill(pw)
            await page.get_by_role("button", name="Log in").click()
        await page.wait_for_load_state("networkidle", timeout=15000)
    finally:
        await ctx.storage_state(path=str(state_path))
        await ctx.close()


async def run_episode(browser, task: dict, policy, out_dir: pathlib.Path, state_path: pathlib.Path | None,
                      som: bool = True, save_images: bool = True) -> dict:
    """task: {task_id, sites, start_urls, intent}. Writes out_dir/{agent_response.json, network.har, traj.jsonl}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = await browser.new_context(viewport=VIEWPORT, record_har_path=str(out_dir / "network.har"),
                                    record_har_content="embed",
                                    storage_state=str(state_path) if state_path and state_path.exists() else None)
    page = await ctx.new_page()
    system = system_prompt(task["sites"], task["intent"], task["start_urls"])
    history, last_error, steps, answer = [], "", [], None
    usage = {"in": 0, "out": 0, "calls": 0}
    t0 = time.time()
    try:
        await page.goto(task["start_urls"][0], timeout=60000)
        for step in range(MAX_STEPS):
            if len(ctx.pages) > 1 and ctx.pages[-1] is not page:  # follow a newly opened tab
                page = ctx.pages[-1]
            obs = await observe(page, som=som)
            msgs = build_messages(system, task["intent"], obs, history, last_error, som=som)
            raw, u = policy(msgs)
            usage["in"] += u["in"]; usage["out"] += u["out"]; usage["calls"] += 1
            rec = {"step": step, "url": obs["url"], "raw": raw, "elements": obs["elements"], "text": obs["text"],
                   "title": obs["title"], "scroll": obs["scroll"], "history": history[-8:], "last_error": last_error}
            if save_images:
                rec["image"] = f"step{step:02d}.png"
                obs["image"].save(out_dir / rec["image"])
            try:
                reasoning, act = parse(raw)
            except ParseError as e:
                last_error = f"could not parse your output ({e}); reply with one reasoning line then one JSON action"
                rec["parse_error"] = str(e)
                steps.append(rec)
                history.append("(invalid output)")
                continue
            rec["reasoning"], rec["action"] = reasoning, act
            steps.append(rec)
            if act["action"] == "stop":
                answer = act["answer"]
                break
            last_error = await execute(page, act, obs["elements"])
            rec["exec_error"] = last_error
            history.append(to_str(act) + (f"  -> error: {last_error}" if last_error else ""))
    except Exception as e:  # noqa: BLE001 — an env failure ends the episode, recorded, never silently
        steps.append({"step": len(steps), "env_error": repr(e)[:500]})
    finally:
        final_url = page.url
        try:
            final_png = await page.screenshot(type="png")
            (out_dir / "final.png").write_bytes(final_png)
        except Exception:  # noqa: BLE001
            pass
        await ctx.close()  # flushes the HAR
    if answer is None:
        answer = {"task_type": "RETRIEVE", "status": "UNKNOWN_ERROR", "retrieved_data": None,
                  "error_details": "no stop action within the step limit"}
    (out_dir / "agent_response.json").write_text(json.dumps(answer))
    with open(out_dir / "traj.jsonl", "w") as f:
        for s in steps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta = {"task_id": task["task_id"], "steps": len(steps), "stopped": answer.get("error_details") != "no stop action within the step limit",
            "final_url": final_url, "usage": usage, "wall_s": round(time.time() - t0, 1), "system": system}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta
