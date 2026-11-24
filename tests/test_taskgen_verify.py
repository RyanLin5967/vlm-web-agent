import asyncio, json, threading, http.server
from playwright.async_api import async_playwright
from wa import taskgen, verify

PAGES = {"/": "<a href='/p/1'>Mug</a> <a href='/logout'>Log out</a> <a href='/p/2'>Lamp</a>",
         "/p/1": "<h1>Blue mug</h1><p>Price: $7.25</p><a href='/'>Home</a>", "/p/2": "<h1>Desk lamp</h1><p>Price: $30.00</p>"}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = PAGES.get(self.path, "nf").encode(); self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass

def test_crawl_skips_logout():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    async def go():
        async with async_playwright() as p:
            b = await p.chromium.launch(); r = await taskgen.crawl(b, "shopping", base, None, max_pages=10); await b.close(); return r
    pages = asyncio.run(go()); srv.shutdown()
    urls = sorted(x["url"].split("/", 3)[-1] for x in pages)
    assert urls == ["", "p/1", "p/2"], urls

def test_parse_proposals_grounding():
    page = {"site": "shopping", "url": "http://h/p/1", "title": "Blue mug", "text": "Blue mug\nPrice: $7.25\nReviews (3)"}
    raw = json.dumps([
        {"task_type": "RETRIEVE", "intent": "What is the price of the blue mug?", "answer": ["$7.25"]},
        {"task_type": "RETRIEVE", "intent": "What is the price of the red mug?", "answer": ["$9.99"]},      # not on page -> drop
        {"task_type": "NAVIGATE", "intent": "Open the product page of the blue mug", "target_url_contains": "/p/1"},
        {"task_type": "MUTATE", "intent": "Add the blue mug to your wishlist please", "check_url": "http://h/wishlist", "check_text": "Blue mug"},  # already on page -> drop
        {"task_type": "MUTATE", "intent": "Write a review titled Great Mug for the blue mug", "check_url": "http://h/p/1", "check_text": "Great Mug"},
        {"task_type": "DELETE", "intent": "Delete everything in the store now"}])
    got = taskgen.parse_proposals("<think>x</think>" + raw, page, "http://h/")
    assert [t["intent"] for t in got] == ["What is the price of the blue mug?", "Open the product page of the blue mug",
                                          "Write a review titled Great Mug for the blue mug"]
    assert taskgen.parse_proposals("no json here", page, "http://h/") == []

def test_verify_checks():
    t = {"answer": ["$7.25"]}
    ok = {"status": "SUCCESS", "retrieved_data": ["7.25"]}
    assert verify.check_retrieve(t, ok)  # currency symbol ignored
    assert verify.check_retrieve(t, {"status": "SUCCESS", "retrieved_data": ["$7.25"]})
    assert not verify.check_retrieve(t, {"status": "NOT_FOUND_ERROR", "retrieved_data": ["$7.25"]})
    assert not verify.check_retrieve(t, {"status": "SUCCESS", "retrieved_data": ["$7.25 " + "x" * 200]})  # answer-stuffing rejected
    assert verify.check_navigate({"target_url_contains": "/p/1"}, {"status": "SUCCESS"}, "http://h/p/1?x")
    assert not verify.check_navigate({"target_url_contains": "/p/1"}, {"status": "SUCCESS"}, "http://h/p/2")
    assert verify.judge_says_yes("<think>hmm</think> yes.") and not verify.judge_says_yes("NO")
