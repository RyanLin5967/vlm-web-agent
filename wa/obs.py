"""Observation: 1280x720 screenshot with a set-of-marks overlay + text element list.

Interactive elements are found from the DOM (visible, in-viewport, not occluded), tagged with
`data-wa-id`, and numbered. The same ids are drawn on the screenshot and listed as text, so a policy
can ground by number. Identical for every model (spec rule 2)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

VIEWPORT = {"width": 1280, "height": 720}
MAX_ELEMENTS = 120
MAX_TEXT_CHARS = 2500

MARK_JS = r"""
() => {
  const SEL = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],' +
    '[role=tab],[role=menuitem],[role=checkbox],[role=radio],[role=option],[role=combobox],[role=textbox],' +
    '[role=switch],[role=treeitem],[role=menuitemcheckbox],[onclick],[contenteditable=true]';
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  document.querySelectorAll('[data-wa-id]').forEach(e => e.removeAttribute('data-wa-id'));
  const W = innerWidth, H = innerHeight, out = [];
  let id = 0;
  for (const el of document.querySelectorAll(SEL)) {
    const r = el.getBoundingClientRect();
    if (r.width < 3 || r.height < 3 || r.bottom <= 0 || r.right <= 0 || r.top >= H || r.left >= W) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || +st.opacity === 0) continue;
    const cx = Math.min(Math.max(r.left + r.width / 2, 1), W - 1), cy = Math.min(Math.max(r.top + r.height / 2, 1), H - 1);
    const top = document.elementFromPoint(cx, cy);
    if (top && !(top === el || el.contains(top) || top.contains(el) || (top.closest('label') && top.closest('label').control === el))) continue;
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    let role = el.getAttribute('role');
    if (!role) role = tag === 'a' ? 'link' : tag === 'select' ? 'combobox' : tag === 'textarea' ? 'textbox' :
      tag === 'input' ? (['checkbox', 'radio'].includes(type) ? type : ['submit', 'button', 'reset', 'image'].includes(type) ? 'button' : 'textbox') :
      tag === 'button' || tag === 'summary' ? 'button' : el.isContentEditable ? 'textbox' : 'clickable';
    let name = el.getAttribute('aria-label') || '';
    if (!name && el.getAttribute('aria-labelledby')) name = el.getAttribute('aria-labelledby').split(' ').map(i => (document.getElementById(i) || {}).innerText || '').join(' ');
    if (!name && el.labels && el.labels.length) name = el.labels[0].innerText;
    if (!name && tag !== 'select') name = el.innerText;
    if (!name) name = el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || el.getAttribute('name') || '';
    if (!name && tag === 'a') name = el.getAttribute('href');
    if (!name) { const img = el.querySelector('img[alt]'); if (img) name = img.getAttribute('alt'); }
    const e = {id, role, name: clean(name).slice(0, 80), x: r.left, y: r.top, w: r.width, h: r.height};
    if (tag === 'input' || tag === 'textarea') {
      if (['checkbox', 'radio'].includes(type)) e.checked = el.checked; else if (el.value) e.value = clean(el.value).slice(0, 60);
    }
    if (tag === 'select') {
      e.value = clean((el.selectedOptions[0] || {}).text || '');
      e.options = Array.from(el.options).slice(0, 15).map(o => clean(o.text).slice(0, 40));
    }
    el.setAttribute('data-wa-id', String(id));
    out.push(e); id++;
    if (id >= %MAX%) break;
  }
  // visible static text (not inside an interactive element), top-to-bottom
  const texts = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let total = 0;
  while (walker.nextNode() && total < %MAXTXT%) {
    const n = walker.currentNode, t = clean(n.textContent);
    if (t.length < 2) continue;
    const p = n.parentElement;
    if (!p || p.closest('[data-wa-id],script,style,noscript')) continue;
    const r = p.getBoundingClientRect();
    if (r.bottom <= 0 || r.top >= H || r.width === 0) continue;
    const st = getComputedStyle(p);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    texts.push(t.slice(0, 200)); total += Math.min(t.length, 200);
  }
  return {elements: out, text: texts, scrollY: Math.round(scrollY), scrollH: document.documentElement.scrollHeight};
}
""".replace("%MAX%", str(MAX_ELEMENTS)).replace("%MAXTXT%", str(MAX_TEXT_CHARS))


def element_line(e: dict) -> str:
    s = f"[{e['id']}] {e['role']} '{e['name']}'"
    if "value" in e:
        s += f" value='{e['value']}'"
    if "checked" in e:
        s += f" checked={str(e['checked']).lower()}"
    if e.get("options"):
        s += " options=" + "|".join(e["options"])
    return s


def draw_som(png: bytes, elements: list[dict], som: bool = True) -> Image.Image:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if not som:
        return img
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    for e in elements:
        x0, y0 = max(e["x"], 0), max(e["y"], 0)
        x1, y1 = min(e["x"] + e["w"], img.width - 1), min(e["y"] + e["h"], img.height - 1)
        d.rectangle([x0, y0, x1, y1], outline=(230, 0, 0), width=2)
        label = str(e["id"])
        tw = 7 * len(label) + 4
        ly = y0 - 14 if y0 >= 14 else y0
        d.rectangle([x0, ly, x0 + tw, ly + 14], fill=(230, 0, 0))
        d.text((x0 + 2, ly + 1), label, fill=(255, 255, 255), font=font)
    return img


async def observe(page, som: bool = True) -> dict:
    """Mark the page, screenshot it, return {image, elements, text, url, title}."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
    except Exception:  # noqa: BLE001
        pass
    data = await page.evaluate(MARK_JS)
    png = await page.screenshot(type="png")
    return {
        "image": draw_som(png, data["elements"], som),
        "elements": data["elements"],
        "text": data["text"],
        "scroll": (data["scrollY"], data["scrollH"]),
        "url": page.url,
        "title": await page.title(),
    }
