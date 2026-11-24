import asyncio, pathlib, socket, pytest
from wa.run import run_many, ScriptedPolicy

def closed_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def test_preflight_refuses_unreachable_site(tmp_path):
    url = f"http://127.0.0.1:{closed_port()}/admin"
    tasks = [{"task_id": 1, "sites": ["map"], "start_urls": [url], "intent": "x"}]
    with pytest.raises(SystemExit, match="REFUSE: preflight"):
        asyncio.run(run_many(tasks, ScriptedPolicy(), tmp_path, {}, 1, True, False))
    assert not (tmp_path / "1" / "meta.json").exists()  # nothing scored
