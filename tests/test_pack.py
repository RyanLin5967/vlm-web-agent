import json, pathlib
import kaggle.launch as L

def test_pack_outputs(tmp_path):
    (tmp_path / "out_collect" / "runs" / "a").mkdir(parents=True)
    (tmp_path / "out_collect" / "collect_summary.json").write_text("{}")
    (tmp_path / "out_collect" / "verified.jsonl").write_text("{}")
    (tmp_path / "out_collect" / "runs" / "a" / "step00.png").write_bytes(b"x")
    (tmp_path / "status.json").write_text("{}")
    src = L.KERNEL % {"job": json.dumps({"name": "x"})}
    ns = {}; exec(src.split("T0 = time.time()")[0], ns); ns["OUT"] = str(tmp_path)
    lines = src.splitlines(); i = [k for k, l in enumerate(lines) if "mkdir -p summary" in l][0]
    exec("\n".join(l.strip() for l in lines[i:i + 4]), ns)
    got = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert got == ["out_collect.tar", "status.json", "summary/out_collect/collect_summary.json",
                   "summary/out_collect/verified.jsonl", "summary/status.json"], got
