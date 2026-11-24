import asyncio, pathlib
from playwright.async_api import async_playwright
from wa.obs import VIEWPORT, observe, element_line
from wa.actions import execute

HTML = """<html><body style='margin:0'>
<a href='#p'>Products</a> <button aria-label='Search'>🔍</button>
<label for=q>Query</label><input id=q placeholder='Find'>
<select id=s><option>USA</option><option>Canada</option></select>
<div style='position:absolute;top:300px;left:0;width:400px;height:60px' onclick='this.innerText="hit"'>Covered</div>
<div style='position:absolute;top:300px;left:0;width:400px;height:60px;background:#fff'>Overlay text</div>
<button style='display:none'>Hidden</button>
<button style='position:absolute;top:2000px'>Below fold</button>
<p>Price: $12.50</p></body></html>"""

def test_marks_and_actions(tmp_path):
    async def go():
        async with async_playwright() as p:
            b = await p.chromium.launch()
            page = await (await b.new_context(viewport=VIEWPORT)).new_page()
            await page.set_content(HTML)
            o = await observe(page)
            names = [e["name"] for e in o["elements"]]
            assert names[:4] == ["Products", "Search", "Query", ""] , names
            assert "Hidden" not in names and "Below fold" not in names and "Covered" not in names
            sel = o["elements"][3]
            assert sel["role"] == "combobox" and sel["options"] == ["USA", "Canada"]
            assert any("Price: $12.50" in t for t in o["text"]) and o["image"].size == (1280, 720)
            assert await execute(page, {"action": "type", "id": 2, "text": "shoes"}, o["elements"]) == ""
            assert await execute(page, {"action": "select", "id": 3, "option": "Canada"}, o["elements"]) == ""
            o2 = await observe(page)
            assert "value='shoes'" in element_line(o2["elements"][2]) and o2["elements"][3]["value"] == "Canada"
            assert "not on the current screen" in await execute(page, {"action": "click", "id": 99}, o2["elements"])
            await b.close()
    asyncio.run(go())

def test_observe_survives_navigation_race():
    import http.server, threading
    pages = {"/": b"<a href='/x'>x</a><script>setTimeout(()=>location.href='/after', 40)</script>",
             "/after": b"<button>after</button>"}
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(pages.get(self.path, b""))
        def log_message(self, *a): pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    async def go():
        async with async_playwright() as p:
            b = await p.chromium.launch()
            page = await (await b.new_context(viewport=VIEWPORT)).new_page()
            await page.goto(f"http://127.0.0.1:{srv.server_address[1]}/")
            o = None
            for _ in range(30):  # observe repeatedly across the navigation; must never raise
                o = await observe(page)
                await page.wait_for_timeout(5)
            assert page.url.endswith("/after") and [e["name"] for e in o["elements"]] == ["after"]
            await b.close()
    asyncio.run(go())
    srv.shutdown()
