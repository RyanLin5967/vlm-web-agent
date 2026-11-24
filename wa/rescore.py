"""Re-score finished eval episodes with the official evaluator under a corrected config (same episodes, no re-run).
usage: python -m wa.rescore --runs DIR --config cfg.json --model NAME --out results.json"""
import argparse, json, pathlib, shutil, subprocess, sys
from types import SimpleNamespace
from wa.run import summarize

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--model", required=True); ap.add_argument("--work", required=True)
a = ap.parse_args()
work = pathlib.Path(a.work); runs = work / "runs"
shutil.copytree(a.runs, runs, dirs_exist_ok=True)
for f in runs.glob("*/eval_result.json"):
    f.unlink()  # never reuse the old verdicts
subprocess.run(["webarena-verified", "eval-tasks", "--config", a.config, "--output-dir", str(runs)], check=False)
metas = [json.loads(m.read_text()) for m in runs.glob("*/meta.json")]
tasks = [{"task_id": m["task_id"], "sites": json.loads((runs / str(m["task_id"]) / "eval_result.json").read_text())["sites"]}
         for m in metas if (runs / str(m["task_id"]) / "eval_result.json").exists()]
if not tasks:
    sys.exit("REFUSE: zero rescored tasks")
summarize(SimpleNamespace(out=str(work), model=a.model, endpoint="rescore", seed=0, temperature=0.0, no_som=False), tasks, metas, 0)
