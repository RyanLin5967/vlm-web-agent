import json, sys, pytest
from wa import build_data, leakage
from tests.test_dataset import make_ep

def setup(tmp_path, intent):
    c = tmp_path / "in" / "out_collect"; (c / "runs" / "syn-r-0-1-s0").mkdir(parents=True)
    make_ep(c / "runs" / "syn-r-0-1-s0")
    (c / "synthetic_tasks.jsonl").write_text(json.dumps({"task_id": "syn-r-0-1", "site": "reddit", "intent": intent, "task_type": "RETRIEVE"}))
    (c / "verified.jsonl").write_text(json.dumps({"task_id": "syn-r-0-1-s0", "verified": True}))

def run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(tmp_path / "in"), "--out", str(tmp_path / "data"), "--dev-frac", "0"])
    monkeypatch.setattr(leakage, "check", lambda train, test=None, embed=True: leakage.__dict__["_orig"](train, test, embed=False))
    build_data.main()

leakage._orig = leakage.check

def test_build_ok(tmp_path, monkeypatch):
    setup(tmp_path, "Find order 7 total in the forum shop")
    run(tmp_path, monkeypatch)
    s = json.loads((tmp_path / "data" / "data_summary.json").read_text())
    assert s["train_steps"] == 2 and s["leakage_hits"] == 0 and s["per_site"]["reddit"]["episodes"] == 1

def test_build_refuses_leak(tmp_path, monkeypatch):
    setup(tmp_path, "Get the top-1 best-selling product name(s) in 2022")  # a real test intent
    with pytest.raises(SystemExit, match="leakage"):
        run(tmp_path, monkeypatch)
    assert not (tmp_path / "data" / "train.jsonl").exists()


def test_judge_override_rejects(tmp_path, monkeypatch):
    setup(tmp_path, "Find order 7 total in the forum shop")
    j = tmp_path / "in" / "judged" / "out_collect"; j.mkdir(parents=True)
    (j / "verified.jsonl").write_text(json.dumps({"task_id": "syn-r-0-1-s0", "verified": False}))
    (j / "source_dir.txt").write_text("x")
    with pytest.raises(SystemExit, match="zero verified"):
        run(tmp_path, monkeypatch)
