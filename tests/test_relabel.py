import json, sys, pytest
from PIL import Image
from wa import relabel
from wa.dataset import rows_from_episode
from wa.actions import parse

def make(tmp, urls, acts, texts=None):
    ep = tmp / "c" / "out_collect" / "runs" / "syn-shopping-1-0-s0"; ep.mkdir(parents=True)
    for i, (u, a) in enumerate(zip(urls, acts)):
        Image.new("RGB", (64, 36)).save(ep / f"step{i:02d}.png")
    steps = [{"step": i, "url": u, "title": f"T{i}", "scroll": [0, 900], "history": [], "last_error": "",
              "elements": [{"id": 1, "role": "link", "name": f"Link{i}"}], "text": (texts or {}).get(i, ["Price $9.50"]),
              "image": f"step{i:02d}.png", "reasoning": "orig", "action": a} for i, (u, a) in enumerate(zip(urls, acts))]
    (ep / "traj.jsonl").write_text("\n".join(json.dumps(s) for s in steps))
    (ep / "meta.json").write_text(json.dumps({"task_id": "syn-shopping-1-0-s0", "system": "OLD INTENT"}))
    (tmp / "c" / "out_collect" / "synthetic_tasks.jsonl").write_text(json.dumps({"site": "shopping", "task_id": "syn-shopping-1-0"}))
    return ep

CLICK = {"action": "click", "id": 1}

def test_pick_endpoint_and_loops():
    st = lambda u, a: {"url": u, "action": a, "image": "x"}
    assert relabel.pick_endpoint([st("/", CLICK), st("/a", CLICK), st("/b", CLICK), st("/b", CLICK)]) == 2
    loop = {"action": "scroll", "direction": "down"}
    assert relabel.pick_endpoint([st("/", CLICK), st("/a", loop), st("/a", loop), st("/a", loop), st("/c", CLICK)]) is None
    assert relabel.pick_endpoint([st("/", CLICK), st("/a", CLICK)]) is None

def test_relabel_episode_end_to_end(tmp_path):
    ep = make(tmp_path, ["/", "/cat", "/p/1", "/p/1"], [CLICK, CLICK, CLICK, CLICK])
    teacher = lambda m: (json.dumps({"task_type": "RETRIEVE", "intent": "What is the price of the blue mug in the store?",
                                     "answer": "$9.50", "reasoning": ["Open the catalog.", "Open the product."]}), {})
    r = relabel.relabel_episode(ep, "a shop", teacher, lambda m: ("YES", {}))
    assert r and len(r["steps"]) == 3 and r["steps"][-1]["action"]["action"] == "stop"
    assert [s["reasoning"] for s in r["steps"][:2]] == ["Open the catalog.", "Open the product."]
    assert r["steps"][1]["history"] == ['{"action": "click", "id": 1}']
    # answer not on the endpoint page -> rejected; judge NO -> rejected
    bad = lambda m: (json.dumps({"task_type": "RETRIEVE", "intent": "What is the price of the red mug in the store?",
                                 "answer": "$1.00", "reasoning": ["a", "b"]}), {})
    assert relabel.relabel_episode(ep, "a shop", bad, lambda m: ("YES", {})) is None
    assert relabel.relabel_episode(ep, "a shop", teacher, lambda m: ("NO", {})) is None

def test_main_writes_collect_dir_and_rows(tmp_path, monkeypatch):
    make(tmp_path, ["http://localhost/", "http://localhost/cat", "http://localhost/p/1", "http://localhost/p/1"], [CLICK] * 4)
    calls = []
    class P:
        def __init__(self, *a, max_tokens=0, **k): self.judge = max_tokens == 8
        def __call__(self, m):
            calls.append(self.judge)
            if self.judge: return "YES", {}
            return json.dumps({"task_type": "NAVIGATE", "intent": "Open the product page of the blue mug please",
                               "answer": None, "reasoning": ["Open the catalog.", "Open the product."]}), {}
    monkeypatch.setattr(relabel, "OpenAIPolicy", P)
    out = tmp_path / "rel"
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(tmp_path / "c"), "--out", str(out), "--model", "m", "--endpoint", "e"])
    relabel.main()
    v = [json.loads(l) for l in open(out / "verified.jsonl")]
    assert len(v) == 1 and v[0]["verified"]
    ep = out / "runs" / v[0]["task_id"]
    meta = json.loads((ep / "meta.json").read_text())
    assert "Open the product page of the blue mug please" in meta["system"] and "OLD INTENT" not in meta["system"]
    rows = rows_from_episode(ep, "Open the product page of the blue mug please")
    assert len(rows) == 3 and parse(rows[-1]["target"])[1]["action"] == "stop"
    assert all(pathlib_exists(r["image"]) for r in rows)

def pathlib_exists(p):
    import pathlib; return pathlib.Path(p).exists()

def test_refuses_eval_runs(tmp_path, monkeypatch):
    d = tmp_path / "out_eval"; d.mkdir(); (d / "synthetic_tasks.jsonl").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(tmp_path), "--out", str(tmp_path / "o"), "--model", "m", "--endpoint", "e"])
    with pytest.raises(SystemExit, match="eval run"):
        relabel.main()
