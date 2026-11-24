"""Generate + push a Kaggle kernel for one job. Code reaches the kernel by cloning this public repo at a
pinned commit, so a run is reproducible from its commit + job JSON (both saved in the output).

usage: python kaggle/launch.py JOB.json        (JOB: see kaggle/jobs/*.json)
Job keys: name, site, mode ("eval"|"collect"), model, vllm_args, run_args, gpu (bool), commit (default: HEAD of origin/dev)"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
REPO = "https://github.com/RyanLin5967/vlm-web-agent.git"

KERNEL = r'''
import json, os, shlex, subprocess, sys, time, traceback
JOB = json.loads(%(job)r)
OUT = "/kaggle/working"
def sh(c, t=3600):
    print("+", c[:200], flush=True)
    # pipefail: steps run as `cmd | tee log | tail`; without it the step's rc is tail's 0 and a REFUSE reads as ok
    r = subprocess.run(["bash", "-lc", f"timeout {t} bash -o pipefail -lc {shlex.quote(c)}"], capture_output=True, text=True)
    print((r.stdout + r.stderr)[-2000:], flush=True); return r.returncode
status = {"job": JOB, "stages": {}}
def stage(n, ok, **kw):
    status["stages"][n] = {"ok": ok, "t": round(time.time() - T0), **kw}; json.dump(status, open(f"{OUT}/status.json", "w"), indent=1)
T0 = time.time()
try:
    assert sh(f"git clone -q {JOB['repo']} /kaggle/repo && cd /kaggle/repo && git checkout -q {JOB['commit']}") == 0
    sys.path.insert(0, "/kaggle/repo"); os.chdir("/kaggle/repo")
    os.makedirs("/kaggle/in", exist_ok=True)
    for slug in JOB.get("inputs", []):  # mount layout varies; link each input to a fixed /kaggle/in/<slug>
        hits = [r for r, ds, fs in os.walk("/kaggle/input") if os.path.basename(r) == slug and r.count("/") <= 6]
        if not hits:
            raise RuntimeError("input not mounted: " + slug + " ; /kaggle/input has: " + str(os.listdir("/kaggle/input")))
        os.symlink(hits[0], "/kaggle/in/" + slug)
    stage("inputs", True, linked={s: os.path.realpath("/kaggle/in/" + s) for s in JOB.get("inputs", [])})
    sh("pip install -q pillow playwright webarena-verified sentence-transformers 2>&1 | tail -2 && python -m playwright install --with-deps chromium 2>&1 | tail -1", 1800)
    if JOB.get("gpu"):
        sh("pip install -q -U vllm 2>&1 | tail -2" if JOB.get("server", "vllm") == "vllm" else "pip install -q -U transformers peft accelerate 2>&1 | tail -2", 2400)
    if JOB.get("train"):
        sh("pip install -q -U transformers peft accelerate 2>&1 | tail -2", 1800)
    stage("setup", True)
    if JOB.get("site"):
        from wa.rootfs import Site
        site = Site(JOB["site"]); boot_s = site.boot(log=lambda m: print(m, flush=True))
        stage("site", True, boot_s=round(boot_s), pull_s=round(site.pull_s), mode=site.mode, selfcheck=site.selfcheck())
        k, v = site.config_entry()
        json.dump({"environments": {k: v}}, open("/kaggle/wa_config.json", "w"))
    if JOB.get("gpu") and JOB.get("model"):
        vl = open(f"{OUT}/vllm.log", "w")
        if JOB.get("server", "vllm") == "hf":
            srv = f"python -m wa.hf_server --model {JOB['model']} --port 8000 {JOB.get('server_args', '')}"
        else:
            srv = f"vllm serve {JOB['model']} --port 8000 {JOB.get('vllm_args', '')}"
        vp = subprocess.Popen(["bash", "-lc", srv], stdout=vl, stderr=subprocess.STDOUT)
        for _ in range(240):
            if sh("curl -sf localhost:8000/v1/models > /dev/null", 20) == 0: break
            if vp.poll() is not None:  # server died: fail now with its errors, not after the 40-min wait
                errs = [l for l in open(f"{OUT}/vllm.log") if "Error" in l or "error" in l][-15:]
                raise RuntimeError("model server exited rc=" + str(vp.returncode) + "\n" + "".join(errs))
            time.sleep(10)
        else:
            raise RuntimeError("model server did not come up in 40 min")
        stage("vllm", True)
    steps = JOB.get("steps") or [{"mode": JOB["mode"], "run_args": JOB.get("run_args", ""), "out": "out", "hours": 11}]
    for i, st in enumerate(steps):  # several steps share one site boot + one model server
        if "cmd" in st:  # raw step, e.g. building SFT rows or training
            rc = sh(f"{st['cmd']} 2>&1 | tee {OUT}/run_{i}.log | tail -40", int(st.get("hours", 5) * 3600))
            stage(f"run_{i}_cmd", rc == 0, rc=rc)
            continue
        site_args = f"--config /kaggle/wa_config.json --sites {JOB['site']} " if JOB.get("site") else ""
        rc = sh(f"python -m wa.run {st['mode']} {site_args}--model {JOB['model']} "
                f"--endpoint {JOB.get('endpoint', 'http://localhost:8000/v1')} --out {OUT}/{st['out']} {st.get('run_args', '')} "
                f"2>&1 | tee {OUT}/run_{i}.log | tail -40", int(st.get("hours", 5) * 3600))
        stage(f"run_{i}_{st['mode']}", rc == 0, rc=rc)
except Exception as e:
    status["error"] = traceback.format_exc()[-3000:]; json.dump(status, open(f"{OUT}/status.json", "w"), indent=1); print(status["error"])
finally:
    try:
        site.diagnose()  # always keep site logs (writes diagnose_<site>.txt and gitlab_logs/)
    except Exception:
        pass
    sh(f"cd {OUT} && tar czf runs_trimmed.tgz --exclude='*.har' out* 2>/dev/null; du -sh {OUT}/* | tail -20")
'''


def main():
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    job.setdefault("repo", REPO)
    if "commit" not in job:
        subprocess.run(["git", "fetch", "-q", "origin"], check=True)
        job["commit"] = subprocess.check_output(["git", "rev-parse", "origin/dev"], text=True).strip()
    slug = "wa-" + job["name"].replace("_", "-")
    d = HERE / "kernels" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(KERNEL % {"job": json.dumps(job)})
    meta = {"id": f"ryanlin5967/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(job.get("gpu")), "enable_internet": True,
            "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [f"ryanlin5967/{k}" for k in job.get("inputs", [])]}
    # no machine_shape: the default GPU is T4x2 and starts at once; "NvidiaTeslaT4" sat QUEUED 30+ min (2026-11-24)
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    env = dict(os.environ, KAGGLE_API_TOKEN=pathlib.Path("~/.config/kaggle/token").expanduser().read_text().strip())
    r = subprocess.run(["uvx", "--quiet", "--with", "six", "kaggle", "kernels", "push", "-p", str(d)], env=env,
                       capture_output=True, text=True, timeout=180)
    print((r.stdout + r.stderr).strip().splitlines()[-1], "commit", job["commit"][:10])


if __name__ == "__main__":
    main()
