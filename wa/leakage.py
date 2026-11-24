"""Spec rule 1: no WebArena test intent, template, or answer may appear in training data.

Checks every training task against all 812 test tasks and exits non-zero on ANY hit:
  exact     normalised intent equals a test intent
  template  intent matches a test intent_template with its {{slots}} as wildcards
  embed     cosine(intent, any test intent or template) > 0.9  (all-MiniLM-L6-v2)
  answer    an expected answer (>= 8 chars normalised) equals a test task's expected answer
Refuses to pass on zero training tasks.

usage: python -m wa.leakage TRAIN.jsonl [--report out.json]   (each line has "intent", optional "answer")
BLIND SPOTS: paraphrases below 0.9 cosine pass the embed check by design (spec threshold); answers under 8
chars (numbers, yes/no) are not compared because they collide by chance. Templates with fewer than 2 literal
words (3 of 190, e.g. "Fork {{repo}}.") get no template-regex check, only exact + embed."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

TEST_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "webarena-verified.json"
THRESH = 0.9
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def template_regex(tpl: str) -> re.Pattern:
    pieces = [norm(p) for p in re.split(r"\{\{\s*\w+\s*\}\}", tpl)]  # split raw: norm() would eat the braces
    return re.compile("^" + r"\s*(.+?)\s*".join(re.escape(p) for p in pieces) + "$")


def answers_of(task: dict) -> list[str]:
    out = []
    for ev in task.get("eval", []):
        exp = ev.get("expected") or {}
        data = exp.get("retrieved_data") if isinstance(exp, dict) else None
        for x in data or []:
            out.append(norm(json.dumps(x) if isinstance(x, (dict, list)) else x))
    return [a for a in out if len(a) >= 8]


def load_test():
    return json.loads(TEST_FILE.read_text())


def check(train: list[dict], test: list[dict] | None = None, embed: bool = True) -> list[dict]:
    if not train:
        raise SystemExit("REFUSE: zero training tasks given — nothing was checked")
    test = test or load_test()
    exact = {norm(t["intent"]): t["task_id"] for t in test}
    templates = {}
    for t in test:
        templates.setdefault(t["intent_template"], t["task_id"])
    # A template with < 2 literal words (e.g. "{{payment_query}}. {{retrieved_data_format_spec}}.") matches any
    # sentence as a regex; those are covered by the exact + embed checks instead.
    tregex = [(template_regex(tpl), tid, tpl) for tpl, tid in templates.items()
              if len(norm(re.sub(r"\{\{\s*\w+\s*\}\}", " ", tpl)).split()) >= 2]
    answers = {}
    for t in test:
        for a in answers_of(t):
            answers.setdefault(a, t["task_id"])
    hits = []
    for i, tr in enumerate(train):
        n = norm(tr["intent"])
        if n in exact:
            hits.append({"i": i, "kind": "exact", "test_id": exact[n], "intent": tr["intent"]})
            continue
        for rx, tid, tpl in tregex:
            if rx.match(n):
                hits.append({"i": i, "kind": "template", "test_id": tid, "intent": tr["intent"], "template": tpl})
                break
        for a in ([tr["answer"]] if isinstance(tr.get("answer"), str) else tr.get("answer") or []):
            na = norm(a)
            if len(na) >= 8 and na in answers:
                hits.append({"i": i, "kind": "answer", "test_id": answers[na], "intent": tr["intent"], "answer": a})
    if embed:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBED_MODEL, device="cpu")
        ref_txt = [t["intent"] for t in test] + list(templates)
        ref_ids = [t["task_id"] for t in test] + list(templates.values())
        R = m.encode(ref_txt, normalize_embeddings=True, batch_size=256, convert_to_tensor=True)
        T = m.encode([t["intent"] for t in train], normalize_embeddings=True, batch_size=256, convert_to_tensor=True)
        for s in range(0, len(train), 2048):
            sims = T[s:s + 2048] @ R.T
            best, idx = sims.max(dim=1)
            for j in (best > THRESH).nonzero().flatten().tolist():
                hits.append({"i": s + j, "kind": "embed", "cos": round(float(best[j]), 4),
                             "test_id": ref_ids[int(idx[j])], "intent": train[s + j]["intent"],
                             "nearest": ref_txt[int(idx[j])]})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--report")
    ap.add_argument("--no-embed", action="store_true")
    a = ap.parse_args()
    train = [json.loads(l) for l in open(a.train) if l.strip()]
    hits = check(train, embed=not a.no_embed)
    rep = {"n_train": len(train), "n_test": len(load_test()), "threshold": THRESH, "embed_model": EMBED_MODEL,
           "n_hits": len(hits), "hits": hits}
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "hits"}))
    for h in hits[:20]:
        print("HIT", json.dumps(h, ensure_ascii=False))
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
