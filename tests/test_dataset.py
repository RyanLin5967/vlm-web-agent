import json
from wa.dataset import rows_from_episode, write_rows
from wa.actions import parse

def make_ep(d):
    (d / "meta.json").write_text(json.dumps({"task_id": "syn-1", "system": "SYS"}))
    steps = [
        {"step": 0, "url": "http://x/", "title": "Home", "scroll": [0, 900], "elements": [{"id": 0, "role": "link", "name": "Orders"}],
         "text": ["Welcome"], "history": [], "last_error": "", "image": "step00.png", "reasoning": "Open orders.",
         "action": {"action": "click", "id": 0}},
        {"step": 1, "url": "http://x/o", "title": "Orders", "scroll": [0, 900], "elements": [], "text": ["Order 7: $5"],
         "history": ['{"action": "click", "id": 0}'], "last_error": "", "image": "step01.png", "parse_error": "bad"},
        {"step": 2, "url": "http://x/o", "title": "Orders", "scroll": [0, 900], "elements": [], "text": ["Order 7: $5"],
         "history": ['{"action": "click", "id": 0}', "(invalid output)"], "last_error": "could not parse", "image": "step02.png",
         "reasoning": "Found it.", "action": {"action": "stop", "answer": {"task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": ["$5"], "error_details": None}}},
    ]
    (d / "traj.jsonl").write_text("\n".join(json.dumps(s) for s in steps))

def test_rows(tmp_path):
    make_ep(tmp_path)
    rows = rows_from_episode(tmp_path, "Find order 7 total")
    assert [r["step"] for r in rows] == [0, 2]  # the parse-error step is not imitated
    r = rows[1]
    assert r["messages"][0] == {"role": "system", "content": "SYS"}
    user = r["messages"][1]["content"]
    assert user[0] == {"type": "image"} and "Task: Find order 7 total" in user[1]["text"] and "Last action error: could not parse" in user[1]["text"]
    assert r["image"].endswith("step02.png")
    assert parse(r["target"])[1]["answer"]["retrieved_data"] == ["$5"]  # target round-trips through the eval parser
    out = tmp_path / "rows.jsonl"
    assert write_rows([(tmp_path, "Find order 7 total")], out) == {"episodes": 1, "steps": 2}
