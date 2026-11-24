"""Action space (spec): click(id), type(id, text, enter?), select(id, option), scroll(up|down), go_back,
goto(url), stop(answer). The model emits one short reasoning line, then one JSON object."""
from __future__ import annotations

import json
import re

ACTION_DOC = """Actions (output exactly one JSON object on the last line):
{"action": "click", "id": <element id>}
{"action": "type", "id": <element id>, "text": "<text>", "enter": true|false}
{"action": "select", "id": <element id>, "option": "<visible option text>"}
{"action": "scroll", "direction": "up"|"down"}
{"action": "go_back"}
{"action": "goto", "url": "<url>"}
{"action": "stop", "answer": {"task_type": "RETRIEVE|MUTATE|NAVIGATE", "status": "SUCCESS|NOT_FOUND_ERROR|ACTION_NOT_ALLOWED_ERROR|PERMISSION_DENIED_ERROR|DATA_VALIDATION_ERROR|UNKNOWN_ERROR", "retrieved_data": [...] or null, "error_details": null or "<why>"}}
Format: first line = one short sentence of reasoning; second line = the JSON action."""

VALID = {"click", "type", "select", "scroll", "go_back", "goto", "stop"}


class ParseError(ValueError):
    pass


def _last_json_object(text: str) -> str:
    """Return the last balanced {...} in text (handles nested objects and strings)."""
    end = text.rfind("}")
    while end != -1:
        depth, in_str, esc = 0, False, False
        for i in range(end, -1, -1):
            c = text[i]
            if in_str:
                if c == '"' and not (i > 0 and text[i - 1] == "\\"):
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        end = text.rfind("}", 0, end)
    raise ParseError("no JSON object found")


def parse(text: str) -> tuple[str, dict]:
    """-> (reasoning, action). Raises ParseError with a message the policy can be shown."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"```(?:json)?", "", text)
    blob = _last_json_object(text)
    try:
        act = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e
    if not isinstance(act, dict) or act.get("action") not in VALID:
        raise ParseError(f"unknown action {act.get('action') if isinstance(act, dict) else act!r}")
    a = act["action"]
    if a in {"click", "type", "select"}:
        try:
            act["id"] = int(act["id"])
        except (KeyError, TypeError, ValueError) as e:
            raise ParseError(f"{a} needs an integer id") from e
    if a == "type" and not isinstance(act.get("text"), str):
        raise ParseError("type needs text")
    if a == "select" and not isinstance(act.get("option"), str):
        raise ParseError("select needs option")
    if a == "scroll" and act.get("direction") not in {"up", "down"}:
        raise ParseError("scroll direction must be up or down")
    if a == "goto" and not isinstance(act.get("url"), str):
        raise ParseError("goto needs url")
    if a == "stop":
        ans = act.get("answer")
        if not isinstance(ans, dict):
            raise ParseError("stop needs an answer object")
        ans.setdefault("retrieved_data", None)
        ans.setdefault("error_details", None)
    reasoning = text[: text.rfind(blob)].strip().splitlines()
    return (reasoning[-1].strip() if reasoning else ""), act


def to_str(act: dict) -> str:
    """Compact, canonical rendering (used in history and as the training target)."""
    return json.dumps(act, ensure_ascii=False, separators=(", ", ": "))


async def execute(page, act: dict, elements: list[dict]) -> str:
    """Run one non-stop action. Returns '' on success or an error string for the next observation."""
    a = act["action"]
    if a in {"click", "type", "select"}:
        if not any(e["id"] == act["id"] for e in elements):
            return f"element {act['id']} is not on the current screen"
        loc = page.locator(f'[data-wa-id="{act["id"]}"]').first
    try:
        if a == "click":
            try:
                await loc.click(timeout=5000)
            except Exception:  # noqa: BLE001 — covered element: click its centre like a user would
                e = next(e for e in elements if e["id"] == act["id"])
                await page.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
        elif a == "type":
            await loc.fill(act["text"], timeout=5000)
            if act.get("enter"):
                await loc.press("Enter", timeout=5000)
        elif a == "select":
            try:
                await loc.select_option(label=act["option"], timeout=5000)
            except Exception:  # noqa: BLE001
                await loc.select_option(value=act["option"], timeout=5000)
        elif a == "scroll":
            await page.mouse.move(640, 360)
            await page.mouse.wheel(0, 600 if act["direction"] == "down" else -600)
        elif a == "go_back":
            await page.go_back(timeout=15000)
        elif a == "goto":
            await page.goto(act["url"], timeout=30000)
    except Exception as e:  # noqa: BLE001
        return f"{a} failed: {str(e).splitlines()[0][:200]}"
    try:
        await page.wait_for_load_state("networkidle", timeout=4000)
    except Exception:  # noqa: BLE001
        pass
    return ""
