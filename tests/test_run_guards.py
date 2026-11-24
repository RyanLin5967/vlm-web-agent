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

def test_preflight_refuses_5xx(tmp_path):
    import http.server, threading
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(b"<title>Something went wrong (500)</title>")
        def log_message(self, *a): pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    tasks = [{"task_id": 1, "sites": ["map"], "start_urls": [f"http://127.0.0.1:{srv.server_address[1]}/"], "intent": "x"}]
    with pytest.raises(SystemExit, match="HTTP 500"):
        asyncio.run(run_many(tasks, ScriptedPolicy(), tmp_path, {}, 1, True, False))
    srv.shutdown()
