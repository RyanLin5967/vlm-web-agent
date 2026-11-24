"""Phase 2 verifier for synthetic episodes: keep a trajectory only if BOTH
  (a) the programmatic check passes (answer / URL / end-state text), and
  (b) an LLM judge reading the final screenshot + intent + answer says the task was done.
Never used on WebArena test tasks — those are scored only by webarena-verified's own evaluator."""
from __future__ import annotations

import json
import re

from wa.obs import VIEWPORT
from wa.policy import _png_data_url  # noqa: F401  (re-exported for judge callers)


def _n(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.$-]", " ", str(s).lower())).strip()


def check_retrieve(task: dict, answer: dict) -> bool:
    if answer.get("status") != "SUCCESS" or not isinstance(answer.get("retrieved_data"), list):
        return False
    got = " | ".join(_n(json.dumps(x) if isinstance(x, (dict, list)) else x) for x in answer["retrieved_data"])
    return bool(got) and all(_n(a) in got for a in task["answer"]) and len(got) <= 4 * sum(len(_n(a)) for a in task["answer"]) + 40


def check_navigate(task: dict, answer: dict, final_url: str) -> bool:
    return answer.get("status") == "SUCCESS" and task["target_url_contains"].lower() in (final_url or "").lower()


async def check_mutate(browser, task: dict, answer: dict, state_path) -> bool:
    if answer.get("status") != "SUCCESS":
        return False
    ctx = await browser.new_context(viewport=VIEWPORT, storage_state=str(state_path) if state_path else None)
    try:
        page = await ctx.new_page()
        await page.goto(task["check_url"], timeout=30000)
        body = await page.evaluate("() => document.body ? document.body.innerText : ''")
        return task["check_text"].lower() in body.lower()
    except Exception:  # noqa: BLE001
        return False
    finally:
        await ctx.close()


JUDGE = """You are grading whether a web agent completed a task. Task: "{intent}"
The agent's final answer: {answer}
The screenshot shows the browser at the end of the episode. Actions taken:
{actions}
Did the agent fully and correctly complete the task? Reply with one word: YES or NO."""


def judge_messages(intent: str, answer: dict, actions: list[str], final_img) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image", "image": final_img},
        {"type": "text", "text": JUDGE.format(intent=intent, answer=json.dumps(answer), actions="\n".join(actions[-12:]) or "(none)")}]}]


def judge_says_yes(raw: str) -> bool:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip().upper()
    return raw.startswith("YES")
