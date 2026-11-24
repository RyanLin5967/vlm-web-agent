import asyncio, json, threading, http.server
from wa.run import run_many, ScriptedPolicy

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(b"<p>hi</p>")
    def log_message(self, *a): pass

def test_deadline_skips_and_reports(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    tasks = [{"task_id": i, "sites": ["map"], "start_urls": [url], "intent": "x"} for i in range(4)]
    done = asyncio.run(run_many(tasks, ScriptedPolicy(), tmp_path, {}, 1, True, False, deadline_s=1e-9))
    srv.shutdown()
    skipped = json.loads((tmp_path / "deadline_skipped.json").read_text())
    assert len(done) + len(skipped) == 4 and len(skipped) >= 3
