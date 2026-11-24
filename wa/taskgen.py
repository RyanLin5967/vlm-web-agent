"""Phase 1: synthetic tasks grounded in the self-hosted sites (never the 812 test tasks).

crawl():   BFS over same-origin links from a site's home, logged in, saving url/title/visible text per page.
propose(): the teacher LLM reads one crawled page and proposes tasks in WebArena-Verified's response
           vocabulary, each with something a program can check afterwards:
             RETRIEVE -> expected `answer` list, copied from the page text (verifier normalises + compares)
             NAVIGATE -> `target_url_contains`, a substring of a URL seen in the crawl
             MUTATE   -> `check_url` + `check_text`: after the episode, open check_url; check_text must appear
Then wa.leakage runs over every proposal; the build fails on any hit (rule 1)."""
from __future__ import annotations

import asyncio
import json
import random
import re
from urllib.parse import urldefrag, urlparse

from wa.obs import VIEWPORT

SKIP = re.compile(r"logout|sign_out|signout|/delete|/destroy|\.(png|jpg|jpeg|gif|svg|pdf|zip|tar|gz|css|js)(\?|$)|"
                  r"download|/raw/|/-/archive/|/blame/|/commits?/[0-9a-f]{7,}|export|print", re.I)

SITE_BLURB = {
    "shopping": "a Magento e-commerce storefront (logged in as customer emma.lopez@gmail.com)",
    "shopping_admin": "the Magento admin panel of an online store (logged in as admin)",
    "reddit": "a Postmill (Reddit-like) forum (logged in as MarvelsGrantMan136)",
    "gitlab": "a GitLab instance (logged in as byteblaze)",
}


async def crawl(browser, site: str, base_url: str, state_path, max_pages: int = 300, seed: int = 0) -> list[dict]:
    ctx = await browser.new_context(viewport=VIEWPORT, storage_state=str(state_path) if state_path else None)
    page = await ctx.new_page()
    origin = urlparse(base_url).netloc
    rng = random.Random(seed)
    frontier, seen, pages = [base_url], {base_url}, []
    while frontier and len(pages) < max_pages:
        url = frontier.pop(rng.randrange(len(frontier)) if len(pages) > 20 else 0)  # BFS first, then random
        try:
            await page.goto(url, timeout=30000)
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")
            links = await page.evaluate("() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)")
        except Exception:  # noqa: BLE001
            continue
        pages.append({"site": site, "url": page.url, "title": await page.title(),
                      "text": re.sub(r"\n\s*\n+", "\n", text)[:6000]})
        for link in links:
            link = urldefrag(link)[0]
            if urlparse(link).netloc == origin and link not in seen and not SKIP.search(link):
                seen.add(link)
                frontier.append(link)
    await ctx.close()
    return pages


PROPOSE = """You write practice tasks for a web-browsing agent on {blurb}.
Below is one page of the site (URL, title, visible text). Propose {k} diverse, realistic tasks a user could
complete starting from the site's home page {home}. Ground every task in facts visible on THIS page.
Mix types: RETRIEVE (find a fact; answer copied exactly from the page text), NAVIGATE (reach a page; give a
URL substring that only the target page has), MUTATE (change something: e.g. edit a field, post, comment,
vote, add to cart/wishlist, create an issue; give a URL to open afterwards and a short text that will appear
there only if the change succeeded). Do not ask for deleting accounts, bulk deletes, or logging out.

Return ONLY a JSON list. Each item:
{{"task_type": "RETRIEVE|NAVIGATE|MUTATE", "intent": "<instruction, natural, specific>",
  "answer": ["<exact value from the page>"] (RETRIEVE only),
  "target_url_contains": "<url substring>" (NAVIGATE only),
  "check_url": "<full url>", "check_text": "<text expected after success>" (MUTATE only)}}

URL: {url}
Title: {title}
Text:
{text}"""


def proposal_messages(page: dict, home: str, k: int = 4) -> list[dict]:
    return [{"role": "user", "content": PROPOSE.format(blurb=SITE_BLURB[page["site"]], k=k, home=home, url=page["url"],
                                                       title=page["title"], text=page["text"][:4000])}]


def parse_proposals(raw: str, page: dict, home: str) -> list[dict]:
    m = re.search(r"\[.*\]", re.sub(r"<think>.*?</think>", "", raw, flags=re.S), re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    low = page["text"].lower()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not isinstance(it.get("intent"), str) or len(it["intent"]) < 15:
            continue
        tt = it.get("task_type")
        task = {"site": page["site"], "sites": [page["site"]], "start_urls": [home], "intent": it["intent"].strip(),
                "task_type": tt, "source_url": page["url"]}
        if tt == "RETRIEVE":
            ans = it.get("answer")
            ans = [ans] if isinstance(ans, (str, int, float)) else ans
            # grounding check: every answer must literally occur on the source page
            if not ans or not all(str(a).strip() and str(a).lower().strip() in low for a in ans):
                continue
            task["answer"] = [str(a).strip() for a in ans]
        elif tt == "NAVIGATE":
            t = it.get("target_url_contains")
            if not isinstance(t, str) or len(t) < 4 or t.rstrip("/") in home.rstrip("/"):
                continue
            task["target_url_contains"] = t
        elif tt == "MUTATE":
            cu, ct = it.get("check_url"), it.get("check_text")
            if not (isinstance(cu, str) and isinstance(ct, str) and len(ct) >= 3):
                continue
            if ct.lower() in low:  # text already present before the change -> check proves nothing
                continue
            task["check_url"], task["check_text"] = cu, ct
        else:
            continue
        out.append(task)
    return out


async def propose_all(pages: list[dict], homes: dict, policy_text, k: int = 4, concurrency: int = 16) -> list[dict]:
    """policy_text(messages) -> (raw, usage); runs blocking calls in threads for concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            try:
                raw, _ = await asyncio.to_thread(policy_text, proposal_messages(p, homes[p["site"]], k))
            except Exception:  # noqa: BLE001
                return []
            return parse_proposals(raw, p, homes[p["site"]])

    res = await asyncio.gather(*(one(p) for p in pages))
    seen, out = set(), []
    for lst in res:
        for t in lst:
            key = re.sub(r"\W+", " ", t["intent"].lower()).strip()
            if key not in seen:
                seen.add(key)
                out.append(t)
    return out
