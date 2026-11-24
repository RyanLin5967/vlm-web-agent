import asyncio, json, threading, http.server, functools
from playwright.async_api import async_playwright
from wa.episode import run_episode

PAGES = {"/": "<a href='/item'>Blue mug</a><p>Welcome</p>", "/item": "<h1>Blue mug</h1><p>Price: $7.25</p>"}

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGES.get(self.path, "nf").encode(); self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass

class FakePolicy:
    def __init__(self): self.calls = []
    def __call__(self, msgs):
        user = msgs[1]["content"][1]["text"]; self.calls.append(user)
        if len(self.calls) == 1: return "garbage with no json", {"in": 10, "out": 2}
        if "Price: $7.25" not in user: return 'Open the mug.\n{"action": "click", "id": 0}', {"in": 10, "out": 5}
        return 'Found.\n{"action":"stop","answer":{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":["$7.25"]}}', {"in": 10, "out": 9}

def test_episode(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    pol = FakePolicy()
    task = {"task_id": "t1", "sites": ["shopping"], "start_urls": [url], "intent": "What does the blue mug cost?"}
    async def go():
        async with async_playwright() as p:
            b = await p.chromium.launch(); m = await run_episode(b, task, pol, tmp_path, None); await b.close(); return m
    m = asyncio.run(go()); srv.shutdown()
    assert json.loads((tmp_path / "agent_response.json").read_text())["retrieved_data"] == ["$7.25"]
    assert m["steps"] == 3 and m["usage"] == {"in": 30, "out": 16, "calls": 3} and m["final_url"].endswith("/item")
    assert "Last action error: could not parse" in pol.calls[1]
    har = json.loads((tmp_path / "network.har").read_text()); assert any(e["request"]["url"].endswith("/item") for e in har["log"]["entries"])
    steps = [json.loads(l) for l in open(tmp_path / "traj.jsonl")]
    assert "parse_error" in steps[0] and steps[1]["action"] == {"action": "click", "id": 0} and (tmp_path / "step01.png").exists()
    assert "emma.lopez" in m["system"] and "What does the blue mug cost?" in m["system"]  # official shopping prompt rendered
