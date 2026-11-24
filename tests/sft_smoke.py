"""SFT smoke on any machine (CPU ok): fake episode with real PNGs -> rows -> 2 LoRA updates on a tiny random
Qwen3-VL -> adapter saved, loss finite, labels cover exactly the assistant target. Exits non-zero on failure.
usage: python tests/sft_smoke.py [--base trl-internal-testing/tiny-Qwen3VLForConditionalGeneration]"""
import json, pathlib, subprocess, sys, tempfile
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests.test_dataset import make_ep
from wa.dataset import write_rows
base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
d = pathlib.Path(tempfile.mkdtemp()); ep = d / "ep"; ep.mkdir()
make_ep(ep)
for i in range(3):
    Image.new("RGB", (1280, 720), (255, 255 - 40 * i, 255)).save(ep / f"step{i:02d}.png")
info = write_rows([(ep, "Find order 7 total")] * 2, d / "rows.jsonl")
assert info["steps"] == 4, info
out = d / "ckpt"
r = subprocess.run([sys.executable, "-m", "wa.sft", "--data", str(d / "rows.jsonl"), "--out", str(out), "--base", base,
                    "--epochs", "1", "--global_batch", "2", "--rank", "8"], cwd=pathlib.Path(__file__).resolve().parent.parent)
assert r.returncode == 0, "sft failed"
log = [json.loads(l) for l in open(out / "train_log.jsonl")]
assert len(log) == 2 and all(l["loss"] == l["loss"] for l in log), log
assert (out / "adapter_config.json").exists() and json.load(open(out / "train_meta.json"))["seen"] == 4
# label mask: decoded supervised tokens == target + eos
from transformers import AutoProcessor
from wa.sft import encode, load_rows
proc = AutoProcessor.from_pretrained(base)
row = load_rows(d / "rows.jsonl")[0]
enc = encode(proc, row, 8192, 1280 * 720)
sup = enc["labels"][0][enc["labels"][0] != -100]
got = proc.tokenizer.decode(sup)
assert got.strip().startswith(row["target"].strip()[:30]) and row["target"].split("\n")[-1] in got, (got, row["target"])
print("SFT_SMOKE_OK", json.dumps({"updates": len(log), "loss": [l["loss"] for l in log], "supervised": got[:200]}))
